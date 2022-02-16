#!/usr/bin/env python
# +
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import torch
import math 
import torch.nn as nn
from sklearn.model_selection import train_test_split as ttsplit

import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals, rms, align
import nglview as nv
import pandas as pd
# -

# ## Part 1: load MD data

# +
sys_name = 'AlanineDipeptide'
# name of PDB file
pdb_filename = "MD_samplers/AlanineDipeptideOpenMM/vacuum.pdb"
# name of DCD file
output_path = 'MD_samplers/allegro-data/working_dir/Langevin_working_dir' 
#output_path = './allegro-data/working_dir/Langevin_working_dir-test3-plumed/' 
traj_dcd_filename = '%s/traj.dcd' % output_path

# load the trajectory data from DCD file
u = mda.Universe(pdb_filename, traj_dcd_filename)
# load the reference configuration from the PDB file
ref = mda.Universe(pdb_filename) 

atoms_info = pd.DataFrame(
    np.array([ref.atoms.ids, ref.atoms.names, ref.atoms.types, ref.atoms.masses, ref.atoms.resids, ref.atoms.resnames]).T, 
    columns=['id', 'name', 'type', 'mass', 'resid', 'resname']
)

# print information of trajectory
print ('\nSummary:\n\
\tno. of atoms: {}\n\
\tno. of residues: {}\n'.format(u.trajectory.n_atoms, u.residues.n_residues)
      )
print ('Detailed atom information:\n', atoms_info)

 

# +
# load trajectory to numpy array
traj = u.trajectory.timeseries(order='fac')
# print information of trajectory
print ('Trajectory Info:\n\
\tno. of frames in trajectory data: {}\n\
\ttimestep: {:.1f}ps\n\
\ttime length: {:.1f}ps\n\
\tshape of data array: {}'.format(u.trajectory.n_frames, 
                                  u.trajectory.time, 
                                  u.trajectory.totaltime,
                                  traj.shape
                                 )
      )

# display the trajectory
view = nv.show_mdanalysis(u)
view   
# -

# ### Optional: generate Ramachandran plot of two dihedral angles

ax = plt.gca()
r = dihedrals.Ramachandran(u.select_atoms('resid 2')).run()
r.plot(ax, color='black', marker='.') #, ref=True)

# ## Alignment

# +
head_frames = 10
selector = "type C or type O or type N"
rmsd_list = []
for ts in u.trajectory[:head_frames]:
    rmsd_ret = rms.rmsd(u.select_atoms(selector).positions, ref.select_atoms(selector).positions, superposition=False)
    rmsd_list.append(rmsd_ret)
print ('First {} RMSD values before alignment:\n\t'.format(head_frames), rmsd_list)

selected_ids = u.select_atoms(selector).ids
print ('\nAligning by atoms:')
print (atoms_info.loc[atoms_info['id'].isin(selected_ids)])

'''
align.AlignTraj(u,  # trajectory to align
                ref,  # reference
                select=selector,  # selection of atoms to align
                filename=None,  # file to write the trajectory to
                in_memory=True,
                match_atoms=True,  # whether to match atoms based on mass
               ).run()
'''
rmsd_list_aligned = []
for ts in u.trajectory[:head_frames]:
    rmsd_ret = rms.rmsd(u.select_atoms(selector).positions, ref.select_atoms(selector).positions, superposition=False)
    rmsd_list_aligned.append(rmsd_ret)
    
print ('\nFirst {} RMSD values after alignment:\n\t'.format(head_frames), rmsd_list_aligned)


# -

# ## Part 2: define neural network model and training function

# +
#We now define the Auto encoders classes and useful functions for the training.

class DeepAutoEncoder(nn.Module):
    def __init__(self, encoder_dims, decoder_dims):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param encoder_dims: list, List of dimensions for encoder, including input/output layers
        :param decoder_dims: list, List of dimensions for decoder, including input/output layers
        """
        super(DeepAutoEncoder, self).__init__()
        layers = []
        for i in range(len(encoder_dims)-2) :
            layers.append(torch.nn.Linear(encoder_dims[i], encoder_dims[i+1])) 
            layers.append( torch.nn.Tanh() )
        layers.append(torch.nn.Linear(encoder_dims[-2], encoder_dims[-1])) 

        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims)-2) :
            layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i+1])) 
            layers.append( torch.nn.Tanh() )
        layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1])) 

        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, inp):
        encoded = self.encoder(inp)
        decoded = self.decoder(encoded)
        return decoded

def xi_ae(model,  x):
    """Collective variable defined through an auto encoder model

    :param model: Neural network model build with PyTorch
    :param x: np.array, position, ndim = 2, shape = (1,1)

    :return: xi: np.array
    """
    model.eval()
    if torch.is_tensor(x) == False :
        x = torch.from_numpy(x).float()
    return model.encoder(x).detach().numpy()


# Next, we define the training function 
def train(model, optimizer, traj, weights, num_epochs=10, batch_size=32, test_size=0.2):
    """Function to train an AE model
    
    :param model: Neural network model built with PyTorch,
    :param loss_function: Function built with PyTorch tensors or built-in PyTorch loss function
    :param optimizer: PyTorch optimizer object
    :param traj: np.array, physical trajectory (in the potential pot), ndim == 2, shape == T // save + 1, pot.dim
    :param weights: np.array, weights of each point of the trajectory when the dynamics is biased, ndim == 1, shape == T // save + 1, 1
    :param num_epochs: int, number of times the training goes through the whole dataset
    :param batch_size: int, number of data points per batch for estimation of the gradient
    :param test_size: float, between 0 and 1, giving the proportion of points used to compute test loss

    :return: model, trained neural net model
    :return: loss_list, list of lists of train losses and test losses; one per batch per epoch
    """
    #--- prepare the data ---
    # split the dataset into a training set (and its associated weights) and a test set
    X_train, X_test, w_train, w_test = ttsplit(traj, weights, test_size=test_size)
    X_train = torch.tensor(X_train.astype('float32'))
    X_test = torch.tensor(X_test.astype('float32'))
    w_train = torch.tensor(w_train.astype('float32'))
    w_test = torch.tensor(w_test.astype('float32'))
    # intialization of the methods to sample with replacement from the data points (needed since weights are present)
    train_sampler = torch.utils.data.WeightedRandomSampler(w_train, len(w_train))
    test_sampler  = torch.utils.data.WeightedRandomSampler(w_test, len(w_test))
    # method to construct data batches and iterate over them
    train_loader = torch.utils.data.DataLoader(dataset=X_train,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=train_sampler)
    test_loader  = torch.utils.data.DataLoader(dataset=X_test,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=test_sampler)
    
    # --- start the training over the required number of epochs ---
    loss_list = []
    print ("\ntraining starts, %d epochs in total." % num_epochs) 
    for epoch in range(num_epochs):
        # Train the model by going through the whole dataset
        model.train()
        train_loss = []
        for iteration, X in enumerate(train_loader):
            # Set gradient calculation capabilities
            X.requires_grad_()
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output
            out = model(X)
            # Evaluate loss
            loss = nn.MSELoss(out, X)
            # Get gradient with respect to parameters of the model
            loss.backward()
            # Store loss
            train_loss.append(loss)
            # Updating parameters
            optimizer.step()
            
        # Evaluate the test loss on the test dataset
        model.eval()
        with torch.no_grad():
            # Evaluation of test loss
            test_loss = []
            for iteration, X in enumerate(test_loader):
                out = model(X)
                # Evaluate loss
                loss = nn.MSELoss(out, X)
                # Store loss
                test_loss.append(loss)
            loss_list.append([torch.tensor(train_loss), torch.tensor(test_loss)])

    print ("training ends.\n") 
    return model, loss_list
# -

# ## Part 3: train autoencoder

#All the parameters are set in the cell below. 
seed = None 
# for training
batch_size = 10000
num_epochs = 500
learning_rate = 0.005
n_bins_z = 20          # number of bins in the encoded dimension
optimizer_algo='Adam'  # Adam by default, otherwise SGD
#dimensions
ae1 = DeepAutoEncoder([2, 20, 20, 1], [1, 20, 20, 2]) 
print("test using NN:", ae1) 
save_fig_to_file = False

# Training the NN

# +
# Define the optimizer
if optimizer_algo == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

(
    ae1,
    loss_list 
) = train(ae1, 
          optimizer, 
          trajectory, 
          np.ones(trajectory.shape[0]), 
          batch_size=batch_size, 
          num_epochs=num_epochs
          )

#--- Compute average train per epoch ---
loss_evol1 = []
for i in range(len(loss_list)):
    loss_evol1.append([torch.mean(loss_list[i][0]), torch.mean(loss_list[i][1])])
loss_evol1 = np.array(loss_evol1)
# -

# Plot the results 

# +
start_epoch_index = 1
fig, (ax0, ax1, ax2)  = plt.subplots(1,3, figsize=(12,4)) 
ax0.plot(range(start_epoch_index, num_epochs), loss_evol1[start_epoch_index:, 0], '--', label='train loss', marker='o')
ax0.plot(range(1, num_epochs), loss_evol1[start_epoch_index:, 1], '-.', label='test loss', marker='+')
ax0.legend()
ax0.set_title('losses')

if save_fig_to_file :
    fig_filename = 'training_loss_%s.jpg' % pot_name
    fig.savefig(fig_filename)
    print ('training loss plotted to file: %s' % fig_filename)
