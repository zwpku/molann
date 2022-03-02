
from utils import *
import itertools 

class TrainingTask(object):
    def __init__(self, args, traj_obj,  histogram_feature_mapper=None, output_feature_mapper=None):

        self.learning_rate = args.learning_rate
        self.num_epochs= args.num_epochs
        self.batch_size = args.batch_size 
        self.test_ratio = args.test_ratio
        self.save_model_every_step = args.save_model_every_step
        self.histogram_feature_mapper = histogram_feature_mapper
        self.output_feature_mapper = output_feature_mapper
        self.traj_obj = traj_obj
        self.args = args
        self.k = args.k


        # path to store log data
        prefix = f"{args.sys_name}-" 
        self.model_path = os.path.join(args.model_save_dir, prefix + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
        print ('\nLog directory: {}\n'.format(self.model_path))
        self.writer = SummaryWriter(self.model_path)

        if self.histogram_feature_mapper is not None :
            histogram_feature = self.histogram_feature_mapper(traj_obj.trajectory).detach().numpy()
            feature_names = self.histogram_feature_mapper.feature_all_names()
            df = pd.DataFrame(data=histogram_feature, columns=feature_names) 

            fig, ax = plt.subplots()
            df.hist(ax=ax)
            fig_name = f'{self.model_path}/histogram_feature.png'
            fig.savefig(fig_name, dpi=200, bbox_inches='tight')
            plt.close()
            self.writer.add_image(f'histogram features', cv.cvtColor(cv.imread(fig_name), cv.COLOR_BGR2RGB), dataformats='HWC')

            df.plot(subplots=True) 
            plt.legend(loc='best')
            fig_name = f'{self.model_path}/feature_along_trajectory.png'
            plt.savefig(fig_name, dpi=200, bbox_inches='tight')
            plt.close()
            self.writer.add_image(f'feature along trajectory', cv.cvtColor(cv.imread(fig_name), cv.COLOR_BGR2RGB), dataformats='HWC')

            print (f'Histogram and trajectory plots of features saved.') 

        if self.output_feature_mapper is not None :
            self.output_features = self.output_feature_mapper(traj_obj.trajectory).detach().numpy()
        else :
            self.output_features = None

    def setup_preprocessing_layer(self):
        # read features from file to define preprocessing
        feature_reader = FeatureFileReader(self.args.feature_file, 'Preprocessing', self.traj_obj.u, use_all_positions_by_default=True)
        feature_list = feature_reader.read()
        
        # define the map from positions to features 
        feature_mapper = FeatureMap(feature_list)

        # display information of features used 
        feature_mapper.info('Features in preprocessing layer:\n')

        if 'position' in [f.type_name for f in feature_list] : # if atom positions are used, add alignment to preprocessing layer
            align_atom_ids = self.traj_obj.u.select_atoms(self.args.align_selector).ids
            print ('\nAdd alignment to preprocessing layer.\naligning by atoms:')
            print (traj_obj.atoms_info.loc[traj_obj.atoms_info['id'].isin(align_atom_ids)][['id','name', 'type']])
            align = Align(traj_obj.ref_pos, align_atom_ids)
        else :
            align = torch.nn.Identity()

        # define preprocessing layer: first align (if positions are used), then map to features
        self.preprocessing_layer = Preprocessing(feature_mapper, align)

    def save_model(self):
        #save the model
        trained_model_filename = f'{self.model_path}/trained_model.pt'
        torch.save(self.model.state_dict(), trained_model_filename)  
        print (f'trained model saved at:\n\t{trained_model_filename}\n')

        cv = self.colvar_model()

        trained_cv_script_filename = f'{self.model_path}/trained_cv_scripted.pt'
        torch.jit.script(cv).save(trained_cv_script_filename)

        print (f'script model for CVs saved at:\n\t{trained_cv_script_filename}\n')

    def output_loss(self):
        loss_evol1 = []
        for i in range(len(self.loss_list)):
            loss_evol1.append([torch.mean(self.loss_list[i][0]), torch.mean(self.loss_list[i][1])])
        loss_evol1 = np.array(loss_evol1)

        start_epoch_index = 1
        ax  = plt.axes() 
        ax.plot(range(start_epoch_index, self.num_epochs), loss_evol1[start_epoch_index:, 0], '--', label='train loss', marker='o')
        ax.plot(range(1, num_epochs), loss_evol1[start_epoch_index:, 1], '-.', label='test loss', marker='+')
        ax.legend()
        ax.set_title('losses')

        fig_filename = 'training_loss_%s.jpg' % pot_name
        fig.savefig(fig_filename)
        print ('training loss plotted to file: %s' % fig_filename)

    def plot_encoder_scattered_on_feature_space(self, X, index, epoch): 

        feature_data = self.output_features[index,:]
        cv_vals = self.cv_on_data(X)

        k = cv_vals.size(1)

        for idx in range(k) :
            fig, ax = plt.subplots()
            sc = ax.scatter(feature_data[:,0], feature_data[:,1], s=2.0, c=cv_vals[:,idx].detach().numpy(), cmap='jet')

            ax.set_title(f'{idx+1}th dimension', fontsize=27)
            ax.set_xlabel(r'{}'.format(self.output_feature_mapper.feature_name(0)), fontsize=25, labelpad=3, rotation=0)
            ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_ylabel(r'{}'.format(self.output_feature_mapper.feature_name(1)), fontsize=25, labelpad=-10, rotation=0)

            cax = fig.add_axes([0.92, 0.10, .02, 0.80])
            cbar = fig.colorbar(sc, cax=cax)
            cbar.ax.tick_params(labelsize=20)

            fig_name = f'{self.model_path}/scattered_{self.model_name}_{epoch}_{idx}.png'
            plt.savefig(fig_name, dpi=200, bbox_inches='tight')
            plt.close()

            self.writer.add_image(f'scattered {self.model_name} {idx}', cv.cvtColor(cv.imread(fig_name), cv.COLOR_BGR2RGB), global_step=epoch, dataformats='HWC')

        print (f'scattered {name_prefix} plot for {epoch}th epoch saved.') 


class AutoEncoderTask(TrainingTask):
    def __init__(self, args, traj_obj,  histogram_feature_mapper=None, output_feature_mapper=None):
        super(AutoEncoderTask, self).__init__(args, traj_obj, histogram_feature_mapper, output_feature_mapper)
        self.model_name = 'autoencoder'

    def setup_model(self) :

        self.setup_preprocessing_layer()
        # output dimension of the map 
        feature_dim = self.preprocessing_layer.feature_dim

        # sizes of feedforward neural networks
        e_layer_dims = [feature_dim] + self.args.e_layer_dims + [self.k]
        d_layer_dims = [self.k] + self.args.d_layer_dims + [feature_dim]
        # define autoencoder
        self.model = AutoEncoder(e_layer_dims, d_layer_dims, self.args.activation())
        # print the model
        print ('\nAutoencoder: input dim: {}, encoded dim: {}\n'.format(feature_dim, self.k), self.model)

        if os.path.isfile(self.args.load_model_filename): 
            self.model.load_state_dict(torch.load(self.args.load_model_filename))
            print (f'model parameters loaded from: {self.args.load_model_filename}')

        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def colvar_model(self):
        return AEColVar(self.preprocessing_layer, self.model.encoder)

    def loss_func(self, X):
        # Forward pass to get output
        out = self.model(X)
        # Evaluate loss
        return torch.nn.MSELoss(out, X)

    def cv_on_data(self, X):
        return self.model.encoder(X)

    def train(self):
        """Function to train the model
        """
        #--- prepare the data ---
        weights = self.traj_obj.traj_weights
        traj = self.preprocessing_layer(self.traj_obj.traj)

        # print information of trajectory
        print ( '\nshape of trajectory data array:\n {}'.format(traj.shape) )

        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(traj, weights, torch.arange(self.traj.shape[0], dtype=torch.long), test_size=self.test_ratio)  
        # intialization of the methods to sample with replacement from the data points (needed since weights are present)
        train_sampler = torch.utils.data.WeightedRandomSampler(w_train, len(w_train))
        test_sampler  = torch.utils.data.WeightedRandomSampler(w_test, len(w_test))
        # method to construct data batches and iterate over them
        train_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.TensorDataset(X_train, index_train),
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   sampler=train_sampler)
        test_loader  = torch.utils.data.DataLoader(dataset= torch.utils.data.TensorDataset(X_test, index_test),
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   sampler=test_sampler)
        
        # --- start the training over the required number of epochs ---
        self.loss_list = []
        print ("\ntraining starts, %d epochs in total." % self.num_epochs) 
        for epoch in tqdm(range(self.num_epochs)):
            # Train the model by going through the whole dataset
            self.model.train()
            train_loss = []
            for iteration, [X, index] in enumerate(train_loader):
                # Clear gradients w.r.t. parameters
                self.optimizer.zero_grad()
                # Evaluate loss
                loss = self.loss_func(X)
                # Get gradient with respect to parameters of the model
                loss.backward()
                # Store loss
                train_loss.append(loss)
                # Updating parameters
                self.optimizer.step()
            # Evaluate the test loss on the test dataset
            self.model.eval()
            with torch.no_grad():
                # Evaluation of test loss
                test_loss = []
                for iteration, [X, index] in enumerate(test_loader):
                    loss = loss_func(X)
                    # Store loss
                    test_loss.append(loss)
                self.loss_list.append([torch.tensor(train_loss), torch.tensor(test_loss)])
                
            self.writer.add_scalar('Loss/train', torch.mean(torch.tensor(train_loss)), epoch)
            self.writer.add_scalar('Loss/test', torch.mean(torch.tensor(test_loss)), epoch)

            if self.output_features is not None :
                self.plot_encoder_scattered_on_feature_space(X, index, epoch)

            if epoch % self.save_model_every_step == 0 :
                self.save_model()

        print ("training ends.\n") 

class EigenFunctionTask(TrainingTask):
    def __init__(self, args, traj_obj,  histogram_feature_mapper=None, output_feature_mapper=None):
        super(EigenFunctionTask, self).__init__(args, traj_obj, histogram_feature_mapper, output_feature_mapper)
        self.model_name = 'eigenfunction'

        self.alpha = args.alpha
        self.beta = args.beta
        # diagnoal matrix 
        # the unit of eigenvalues given by Rayleigh quotients is ns^{-1}.
        self.diag_coeff = torch.ones(self.tot_dim).double() * args.diffusion_coeff * 1e7 * self.beta
        self.sort_eigvals_in_training = args.sort_eigvals_in_training
        self.eig_w = args.eig_w
        self.v_in_jac = torch.ones(self.batch_size, dtype=torch.float64)
        # list of (i,j) pairs
        #self.ij_list = list(itertools.combinations_with_replacement(range(Param.k), 2))
        self.ij_list = list(itertools.combinations(range(self.k), 2))
        self.num_ij_pairs = len(self.ij_list)

    def setup_model(self) :

        self.setup_preprocessing_layer()
        # output dimension of the map 
        feature_dim = self.preprocessing_layer.feature_dim

        layer_dims = [feature_dim] + self.args.layer_dims + [1]
        self.model = EigenFunction(layer_dims, self.k, self.preprocessing_layer, self.args.activation())

        print ('\nEigenfunctions:\n', self.model)

        if os.path.isfile(self.args.load_model_filename): 
            self.model.load_state_dict(torch.load(self.args.load_model_filename))
            print (f'model parameters loaded from: {self.args.load_model_filename}')

        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def colvar_model(self):
        return self.model

    def cv_on_data(self, X):
        return self.model(X)

    def loss_func(self, X, b_weights):
        # Evaluate function value on data
        y = self.model(X)

        """
          Apply the Jacobian-vector trick to compute spatial gradients.
          The flag create_graph=True is needed, because later we need to compute
          gradients w.r.t. parameters; Please refer to the torch.autograd.grad function for details.
        """
        y_grad_vec = [torch.autograd.grad(y[:,idx], X, self.v_in_jac, create_graph=True)[0] for idx in range(self.k)]

        # Total weights, will be used for normalization 
        b_tot_weights = b_weights.sum()

        # Mean and variance evaluated on data
        mean_list = [(y[:,idx] * b_weights).sum() / b_tot_weights for idx in range(self.k)]
        var_list = [(y[:,idx]**2 * b_weights).sum() / b_tot_weights - mean_list[idx]**2 for idx in range(self.k)]

        # Compute Rayleigh quotients as eigenvalues
        eig_vals = torch.tensor([1.0 / (b_tot_weights * self.beta) * torch.sum((y_grad_vec[idx]**2 * self.diag_coeff).sum(dim=1) * b_weights) / var_list[idx] for idx in range(self.k)])

        cvec = range(self.k)
        if self.sort_eigvals_in_training :
            cvec = np.argsort(eig_vals)
            # Sort the eigenvalues 
            eig_vals = eig_vals[cvec]

        non_penalty_loss = 1.0 / (b_tot_weights * self.beta) * sum([self.eig_w[idx] * torch.sum((y_grad_vec[cvec[idx]]**2 * self.diag_coeff).sum(dim=1) * b_weights) / var_list[cvec[idx]] for idx in range(self.k)])

        penalty = torch.zeros(1, requires_grad=True).double()

        # Sum of squares of variance for each eigenfunction
        penalty = sum([(var_list[idx] - 1.0)**2 for idx in range(self.k)])

        for idx in range(self.num_ij_pairs):
          ij = self.ij_list[idx]
          # Sum of squares of covariance between two different eigenfunctions
          penalty += ((y[:, ij[0]] * y[:, ij[1]] * b_weights).sum() / b_tot_weights - mean_list[ij[0]] * mean_list[ij[1]])**2

        loss = 1.0 * non_penalty_loss + self.alpha * penalty 

        return loss, eig_vals, non_penalty_loss, penalty, cvec

    def train(self):
        """Function to train the model
        """
        #--- prepare the data ---
        weights = self.traj_obj.traj_weights
        traj = self.traj_obj.traj

        # print information of trajectory
        print ( '\nshape of trajectory data array:\n {}'.format(traj.shape) )

        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(traj, weights, torch.arange(traj.shape[0], dtype=torch.long), test_size=self.test_ratio)  
        # intialization of the methods to sample with replacement from the data points (needed since weights are present)
        train_sampler = torch.utils.data.WeightedRandomSampler(w_train, len(w_train))
        test_sampler  = torch.utils.data.WeightedRandomSampler(w_test, len(w_test))
        # method to construct data batches and iterate over them
        train_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.TensorDataset(X_train, index_train),
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   sampler=train_sampler)
        test_loader  = torch.utils.data.DataLoader(dataset= torch.utils.data.TensorDataset(X_test, index_test),
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   sampler=test_sampler)
        
        # --- start the training over the required number of epochs ---
        self.loss_list = []
        print ("\ntraining starts, %d epochs in total." % self.num_epochs) 
        for epoch in tqdm(range(self.num_epochs)):
            # Train the model by going through the whole dataset
            self.model.train()
            train_loss = []
            for iteration, [X, index] in enumerate(train_loader):
                # Clear gradients w.r.t. parameters
                self.optimizer.zero_grad()
                # Evaluate loss
                loss, eig_vals, non_penalty_loss, penalty, cvec = self.loss_func(X)
                # Get gradient with respect to parameters of the model
                loss.backward()
                # Store loss
                train_loss.append(loss)
                # Updating parameters
                self.optimizer.step()
            # Evaluate the test loss on the test dataset
            self.model.eval()
            with torch.no_grad():
                # Evaluation of test loss
                test_loss = []
                for iteration, [X, index] in enumerate(test_loader):
                    loss = loss_func(X)
                    # Store loss
                    test_loss.append(loss)
                self.loss_list.append([torch.tensor(train_loss), torch.tensor(test_loss)])
                
            self.writer.add_scalar('Loss/train', torch.mean(torch.tensor(train_loss)), epoch)
            self.writer.add_scalar('Loss/test', torch.mean(torch.tensor(test_loss)), epoch)

            if self.output_features is not None :
                self.plot_encoder_scattered_on_feature_space(X, index, epoch)

            if epoch % self.save_model_every_step == 0 :
                self.save_model()

        print ("training ends.\n") 

