#!/usr/bin/env python
# +
import numpy as np
import matplotlib.pyplot as plt
import torch
import math 
import random
from sklearn.model_selection import train_test_split 

import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals 
import nglview as nv
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import time
import argparse
import json
from datetime import datetime
import configparser
# -

def set_all_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    random.seed(seed)

class MyArgs(object):

    def __init__(self, config_filename='params.cfg'):

        config = configparser.ConfigParser()
        config.read(config_filename)

        self.pdb_filename = config['System']['pdb_filename']
        self.traj_dcd_filename = config['System']['traj_dcd_filename']
        self.sys_name = config['System']['sys_name']
          
        #set training parameters
        self.use_gpu =config['Training'].getboolean('use_gpu')
        self.batch_size = config['Training'].getint('batch_size')
        self.num_epochs = config['Training'].getint('num_epochs')
        self.test_ratio = config['Training'].getfloat('test_ratio')
        self.learning_rate = config['Training'].getfloat('learning_rate')
        self.optimizer = config['Training']['optimizer']
        self.k = config['Training'].getint('encoded_dim')
        self.e_layer_dims = [int(x) for x in config['Training']['encoder_hidden_layer_dims'].split(',')]
        self.d_layer_dims = [int(x) for x in config['Training']['decoder_hidden_layer_dims'].split(',')]
        self.load_model_filename =  config['Training']['load_model_filename']
        self.model_save_dir = config['Training']['model_save_dir'] 
        self.save_model_every_step = config['Training'].getint('save_model_every_step')
        
        self.activation_name = config['Training']['activation'] 
        self.activation = getattr(torch.nn, self.activation_name) 

        self.align_selector = config['Training']['align_mda_selector']
        self.train_atom_selector = config['Training']['train_mda_selector'] 
        self.seed = config['Training'].getint('seed')

        if self.seed:
            set_all_seeds(self.seed)

        # encoded dimension
        # CUDA support
        if torch.cuda.is_available() and self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print (f'Parameters loaded from: {config_filename}\n')

class Trajectory(object):
    def __init__(self, args):
        # load the trajectory data from DCD file
        self.u = mda.Universe(args.pdb_filename, args.traj_dcd_filename)

        # load the reference configuration from the PDB file
        self.ref = mda.Universe(args.pdb_filename) 

        self.atoms_info = pd.DataFrame(
            np.array([self.ref.atoms.ids, self.ref.atoms.names,
                self.ref.atoms.types, self.ref.atoms.masses,
                self.ref.atoms.resids, self.ref.atoms.resnames]).T, 
            columns=['id', 'name', 'type', 'mass', 'resid', 'resname']
            )

        # print information of trajectory
        print ('\nMD system:\n\
        \tno. of atoms: {}\n\
        \tno. of residues: {}\n'.format(self.ref.trajectory.n_atoms, self.ref.residues.n_residues)
              )
        print ('Detailed atom information:\n', self.atoms_info)

        print ('\nSummary:\n', self.atoms_info['type'].value_counts().rename_axis('type').reset_index(name='counts'))

        self.load_traj()

        self.ref_pos = self.ref.atoms.positions

    def load_traj(self):

        print ('\nloading trajectory to numpy array...', end='')
        # load trajectory to torch tensor 
        self.trajectory = torch.from_numpy(self.u.trajectory.timeseries(order='fac')).double()

        print ('done.')

        # print information of trajectory
        print ('\nTrajectory Info:\n\
        \tno. of frames in trajectory data: {}\n\
        \ttimestep: {:.1f}ps\n\
        \ttime length: {:.1f}ps\n\
        \tshape of trajectory data array: {}\n'.format(self.u.trajectory.n_frames, 
                                          self.u.trajectory.time, 
                                          self.u.trajectory.totaltime,
                                          self.trajectory.shape
                                         )
              )
        self.weights = np.ones(self.trajectory.shape[0])

class Preprocessing(torch.nn.Module):
    
    def __init__(self, args, traj_obj):

        super(Preprocessing, self).__init__()

        self.align_atom_ids = traj_obj.u.select_atoms(args.align_selector).ids
        self.train_atom_ids = traj_obj.u.select_atoms(args.train_atom_selector).ids 

        print ('\nInfor of preprocess layer.\naligning by atoms:')
        print (traj_obj.atoms_info.loc[traj_obj.atoms_info['id'].isin(self.align_atom_ids)][['id','name', 'type']])

        self.align_atom_indices = torch.tensor(self.align_atom_ids-1).long() # minus one, such that the index starts from 0
        self.train_atom_indices = torch.tensor(self.train_atom_ids-1).long() 

        self.ref_x = torch.from_numpy(traj_obj.ref_pos[self.align_atom_indices, :]).double()        

        # shift reference state 
        ref_c = torch.mean(self.ref_x, 0) 
        self.ref_x = self.ref_x - ref_c
        
    def show_info(self):
        print ('atom indices used for alignment: ', self.align_atom_indices.numpy())
        print ('atom indices used for input layer: ', self.train_atom_indices.numpy())
        print ('\n\treference state used in aligment:\n', self.ref_x.numpy())

    def align(self, traj):  
        """
        align trajectory by translation and rotation
        """
                         
        traj_selected_atoms = traj[:, self.align_atom_indices, :]
        # centers
        x_c = torch.mean(traj_selected_atoms, 1, True)
        # translation
        x_notran = traj_selected_atoms - x_c 
        
        xtmp = x_notran.permute((0,2,1))
        prod = torch.matmul(xtmp, self.ref_x) # dimension: traj_length x 3 x 3
        u, s, vh = torch.linalg.svd(prod)

        diag_mat = torch.diag(torch.ones(3)).double().unsqueeze(0).repeat(traj.size(0), 1, 1)

        sign_vec = torch.sign(torch.linalg.det(torch.matmul(u, vh))).detach()
        diag_mat[:,2,2] = sign_vec

        rotate_mat = torch.bmm(torch.bmm(u, diag_mat), vh)

        aligned_traj = torch.matmul(traj-x_c, rotate_mat) 
                
        return aligned_traj     
    
    def forward(self, inp):
        inp = self.align(inp)
        inp = torch.flatten(inp[:,self.train_atom_indices,:], start_dim=1)
        return inp


class ColVar(torch.nn.Module):
    def __init__(self, preprocessing_layer, encoder):
        super(ColVar, self).__init__()
        self.preprocessing_layer = preprocessing_layer
        self.encoder = encoder
    def forward(self, inp):
        return self.encoder(self.preprocessing_layer(inp))

#Auto encoders class and functions for training.
def create_sequential_nn(layer_dims, activation=torch.nn.Tanh()):
    layers = []
    for i in range(len(layer_dims)-2) :
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1])) 
        layers.append(activation)
    layers.append(torch.nn.Linear(layer_dims[-2], layer_dims[-1])) 

    return torch.nn.Sequential(*layers).double()

class AutoEncoder(torch.nn.Module):
    def __init__(self, e_layer_dims, d_layer_dims, activation=torch.nn.Tanh()):
        super(AutoEncoder, self).__init__()
        self.encoder = create_sequential_nn(e_layer_dims, activation)
        self.decoder = create_sequential_nn(d_layer_dims, activation)

    def forward(self, inp):
        return self.decoder(self.encoder(inp))

# +
class TrainingTask(object):
    def __init__(self, args, traj_obj, preprocessing_layer, ae_model):

        self.ae_model = ae_model
        self.learning_rate = args.learning_rate
        self.preprocessing_layer = preprocessing_layer
        self.num_epochs= args.num_epochs
        self.batch_size = args.batch_size 
        self.test_ratio = args.test_ratio
        self.save_model_every_step = args.save_model_every_step

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

        self.traj = self.preprocessing_layer(traj_obj.trajectory)

        self.traj_weights = traj_obj.weights

        # print information of trajectory
        print ('\nshape of preprocessed trajectory data array:\n {}'.format(self.traj.shape))
         
    def save_model(self):
        #save the model
        trained_model_filename = f'{self.model_path}/trained_model.pt'
        torch.save(self.ae_model.state_dict(), trained_model_filename)  
        print (f'trained model is saved at:\n\t{trained_model_filename}\n')

        cv = ColVar(self.preprocessing_layer, self.ae_model.encoder)

        trained_cv_script_filename = f'{self.model_path}/trained_cv_scripted.pt'
        torch.jit.script(cv).save(trained_cv_script_filename)

        print (f'script model for CVs is saved at:\n\t{trained_cv_script_filename}\n')

# Next, we define the training function 
    def train(self):
        """Function to train an AE model
        """
        #--- prepare the data ---
        # split the dataset into a training set (and its associated weights) and a test set
        X_train, X_test, w_train, w_test = train_test_split(self.traj, self.traj_weights, test_size=self.test_ratio)  
        # intialization of the methods to sample with replacement from the data points (needed since weights are present)
        train_sampler = torch.utils.data.WeightedRandomSampler(w_train, len(w_train))
        test_sampler  = torch.utils.data.WeightedRandomSampler(w_test, len(w_test))
        # method to construct data batches and iterate over them
        train_loader = torch.utils.data.DataLoader(dataset=X_train,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   sampler=train_sampler)
        test_loader  = torch.utils.data.DataLoader(dataset=X_test,
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
            for iteration, X in enumerate(train_loader):
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
                for iteration, X in enumerate(test_loader):
                    out = self.ae_model(X)
                    # Evaluate loss
                    loss = loss_func(out,X)
                    # Store loss
                    test_loss.append(loss)
                self.loss_list.append([torch.tensor(train_loss), torch.tensor(test_loss)])
                
            self.writer.add_scalar('Loss/train', torch.mean(torch.tensor(train_loss)), epoch)
            self.writer.add_scalar('Loss/test', torch.mean(torch.tensor(test_loss)), epoch)

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
# -

def main():

    args = MyArgs()

    traj_obj = Trajectory(args)

    #preprocessing the trajectory data
    preprocessing_layer = Preprocessing(args, traj_obj)

    input_dim = 3 * len(preprocessing_layer.train_atom_ids)
    e_layer_dims = [input_dim] + args.e_layer_dims + [args.k]
    d_layer_dims = [args.k] + args.d_layer_dims + [input_dim]

    ae_model = AutoEncoder(e_layer_dims, d_layer_dims, args.activation())

    print ('\nAutoencoder:\n', ae_model)
    # encoded dimension
    print ('\nInput dim: {},\tencoded dim: {}\n'.format(input_dim, args.k))
    #input dimension of nn
    print ('\n{} Atoms used in define neural network:\n'.format(len(preprocessing_layer.train_atom_ids)), \
                    traj_obj.atoms_info.loc[traj_obj.atoms_info['id'].isin(preprocessing_layer.train_atom_ids)][['id','name', 'type']])

    train_obj = TrainingTask(args, traj_obj, preprocessing_layer, ae_model)
    train_obj.train()

if __name__ == "__main__":
    main()

