#!/usr/bin/env python
# +
import cv2 as cv
from utils import *
# -

# +
class TrainingTask(object):
    def __init__(self, args, traj_obj, preprocessing_layer, ae_model, histogram_feature_mapper=None, output_feature_mapper=None):

        self.ae_model = ae_model
        self.learning_rate = args.learning_rate
        self.preprocessing_layer = preprocessing_layer
        self.num_epochs= args.num_epochs
        self.batch_size = args.batch_size 
        self.test_ratio = args.test_ratio
        self.save_model_every_step = args.save_model_every_step
        self.histogram_feature_mapper = histogram_feature_mapper
        self.output_feature_mapper = output_feature_mapper

        if os.path.isfile(args.load_model_filename): 
            self.ae_model.load_state_dict(torch.load(args.load_model_filename))
            print (f'model parameters loaded from: {args.load_model_filename}')

        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.ae_model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.ae_model.parameters(), lr=self.learning_rate)

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

        self.traj = self.preprocessing_layer(traj_obj.trajectory)

        self.traj_weights = traj_obj.weights

        # print information of trajectory
        print ('\nshape of preprocessed trajectory data array:\n {}'.format(self.traj.shape))

    def save_model(self):
        #save the model
        trained_model_filename = f'{self.model_path}/trained_model.pt'
        torch.save(self.ae_model.state_dict(), trained_model_filename)  
        print (f'trained model saved at:\n\t{trained_model_filename}\n')

        cv = ColVar(self.preprocessing_layer, self.ae_model.encoder)

        trained_cv_script_filename = f'{self.model_path}/trained_cv_scripted.pt'
        torch.jit.script(cv).save(trained_cv_script_filename)

        print (f'script model for CVs saved at:\n\t{trained_cv_script_filename}\n')

# Next, we define the training function 
    def train(self):
        """Function to train an AE model
        """
        #--- prepare the data ---
        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test, index_train, index_test = train_test_split(self.traj, self.traj_weights, torch.arange(self.traj.shape[0], dtype=torch.long), test_size=self.test_ratio)  
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
        
        loss_func = torch.nn.MSELoss()
        # --- start the training over the required number of epochs ---
        self.loss_list = []
        print ("\ntraining starts, %d epochs in total." % self.num_epochs) 
        for epoch in tqdm(range(self.num_epochs)):
            # Train the model by going through the whole dataset
            self.ae_model.train()
            train_loss = []
            for iteration, [X, index] in enumerate(train_loader):
                # Clear gradients w.r.t. parameters
                self.optimizer.zero_grad()
                # Forward pass to get output
                out = self.ae_model(X)
                # Evaluate loss
                loss = loss_func(out, X)
                # Get gradient with respect to parameters of the model
                loss.backward()
                # Store loss
                train_loss.append(loss)
                # Updating parameters
                self.optimizer.step()
            # Evaluate the test loss on the test dataset
            self.ae_model.eval()
            with torch.no_grad():
                # Evaluation of test loss
                test_loss = []
                for iteration, [X, index] in enumerate(test_loader):
                    out = self.ae_model(X)
                    # Evaluate loss
                    loss = loss_func(out,X)
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
        encoded_val = self.ae_model.encoder(X)
        k = encoded_val.size(1)

        for idx in range(k) :
            fig, ax = plt.subplots()
            sc = ax.scatter(feature_data[:,0], feature_data[:,1], s=2.0, c=encoded_val[:,idx].detach().numpy(), cmap='jet')

            ax.set_title(f'{idx+1}th dimension', fontsize=27)
            ax.set_xlabel(r'{}'.format(self.output_feature_mapper.feature_name(0)), fontsize=25, labelpad=3, rotation=0)
            ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_ylabel(r'{}'.format(self.output_feature_mapper.feature_name(1)), fontsize=25, labelpad=-10, rotation=0)

            cax = fig.add_axes([0.92, 0.10, .02, 0.80])
            cbar = fig.colorbar(sc, cax=cax)
            cbar.ax.tick_params(labelsize=20)

            fig_name = f'{self.model_path}/scattered_encoder_{epoch}_{idx}.png'
            plt.savefig(fig_name, dpi=200, bbox_inches='tight')
            plt.close()

            self.writer.add_image(f'scattered encoder {idx}', cv.cvtColor(cv.imread(fig_name), cv.COLOR_BGR2RGB), global_step=epoch, dataformats='HWC')

        print (f'scattered encoder plot for {epoch}th epoch saved.') 

def main():

    # Read configuration parameters
    args = MyArgs()

    # Read trajectory
    traj_obj = Trajectory(args.pdb_filename, args.traj_dcd_filename)

    # Read preprocessing file
    feature_reader = FeatureFileReader(args.feature_file, 'Preprocessing', traj_obj.u, use_all_positions_by_default=True)
    feature_name_list, feature_type_id_list, feature_type_list, feature_ag_list = feature_reader.read()
    
    # Define the map to feature space
    feature_mapper = FeatureMap(feature_name_list, feature_type_id_list, feature_ag_list)
    feature_dim = feature_mapper.feature_total_dimension()

    # Display information of features used 
    print ('Feature List:\nId.\tName\tType\tAtomIDs')
    for idx in range(len(feature_name_list)):
        print ( idx, feature_name_list[idx], feature_type_list[idx], feature_ag_list[idx])

    # Add alignment layer if position is used
    if 'position' in feature_type_list :
        align_atom_ids = traj_obj.u.select_atoms(args.align_selector).ids
        print ('\nAdd Alignment layer in preprocess layer.\naligning by atoms:')
        print (traj_obj.atoms_info.loc[traj_obj.atoms_info['id'].isin(align_atom_ids)][['id','name', 'type']])
        align = Align(traj_obj.ref_pos, align_atom_ids)
    else :
        align = torch.nn.Identity()

    # Define preprocessing layer
    preprocessing_layer = Preprocessing(feature_mapper, align)

    e_layer_dims = [feature_dim] + args.e_layer_dims + [args.k]
    d_layer_dims = [args.k] + args.d_layer_dims + [feature_dim]

    # Define autoencoder
    ae_model = AutoEncoder(e_layer_dims, d_layer_dims, args.activation())

    # Show the model
    print ('\nAutoencoder:\n', ae_model)
    # Encoded dimension
    print ('\nInput dim: {},\tencoded dim: {}\n'.format(feature_dim, args.k))

    # Feature used for h print
    feature_reader = FeatureFileReader(args.feature_file, 'Histogram', traj_obj.u, ignore_position_feature=True)
    feature_name_list, feature_type_id_list, feature_type_list, feature_ag_list = feature_reader.read()

    print ('Features for histogram plot:\nId.\tName\tType\tAtomIDs')
    for idx in range(len(feature_name_list)):
        print (idx, feature_name_list[idx], feature_type_list[idx], feature_ag_list[idx])
    histogram_feature_mapper = FeatureMap(feature_name_list, feature_type_id_list, feature_ag_list, use_angle_value=True)

    assert histogram_feature_mapper.feature_total_dimension() == len(feature_name_list), "Feature map for histogram is incorrect" 

    # Feature used for output
    feature_reader = FeatureFileReader(args.feature_file, 'Output', traj_obj.u, ignore_position_feature=True)
    feature_name_list, feature_type_id_list, feature_type_list, feature_ag_list = feature_reader.read()

    if len(feature_name_list) == 2 : # Use it only if it is 2D
        print ('2d feature List for output:\nId.\tName\tType\tAtomIDs')
        for idx in range(len(feature_name_list)):
            print (idx, feature_name_list[idx], feature_type_list[idx], feature_ag_list[idx])
        output_feature_mapper = FeatureMap(feature_name_list, feature_type_id_list, feature_ag_list, use_angle_value=True)
    else :
        print (f'2d feature required, {len(feature_type_list)} are provided.')
        output_feature_mapper = None

    # Define training task
    train_obj = TrainingTask(args, traj_obj, preprocessing_layer, ae_model, histogram_feature_mapper, output_feature_mapper)

    # Train autoencoder
    train_obj.train()

if __name__ == "__main__":
    main()

