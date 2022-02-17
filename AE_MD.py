#!/usr/bin/env python
# +
import numpy as np
import matplotlib.pyplot as plt
import torch
import math 
import random
from sklearn.model_selection import train_test_split 

import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals, rms, align
import nglview as nv
import pandas as pd
# -

# ## Part 1: prepare MD data
# #### 1.1. show some information

# +
sys_name = 'AlanineDipeptide'

# name of PDB file
pdb_filename = "MD_samplers/AlanineDipeptideOpenMM/vacuum.pdb"
# name of DCD file
output_path = 'MD_samplers/allegro-data/working_dir/Langevin_working_dir' 
#output_path = './allegro-data/working_dir/Langevin_working_dir-test3-plumed/' 
traj_dcd_filename = '%s/traj.dcd' % output_path

# load the reference configuration from the PDB file
ref = mda.Universe(pdb_filename) 

atoms_info = pd.DataFrame(
    np.array([ref.atoms.ids, ref.atoms.names, ref.atoms.types, ref.atoms.masses, ref.atoms.resids, ref.atoms.resnames]).T, 
    columns=['id', 'name', 'type', 'mass', 'resid', 'resname']
)

# print information of trajectory
print ('\nMD system:\n\
\tno. of atoms: {}\n\
\tno. of residues: {}\n'.format(ref.trajectory.n_atoms, ref.residues.n_residues)
      )
print ('Detailed atom information:\n', atoms_info)

print ('\nSummary:\n', atoms_info['type'].value_counts().rename_axis('type').reset_index(name='counts'))


# -


# #### 1.2 load trajectory, and align with respect to refenrence

# +
def align(traj, ref_pos, align_atom_indices):
    
        traj_selected_atoms = traj[:, align_atom_indices, :]
        # translation
        x_notran = traj_selected_atoms - ref_pos 
        
        xtmp = x_notran.permute((0,2,1)).reshape((-1, self.ref_num_atoms))
        prod = torch.matmul(xtmp, self.ref_x).reshape((-1, 3, 3))
        u, s, vh = torch.linalg.svd(prod)

        diag_mat = torch.diag(torch.ones(3)).double().unsqueeze(0).repeat(self.batch_size, 1, 1)

        sign_vec = torch.sign(torch.linalg.det(torch.matmul(u, vh))).detach()
        diag_mat[:,2,2] = sign_vec

        rotate_mat = torch.bmm(torch.bmm(u, diag_mat), vh)

        return torch.matmul(x-ref_pos, rotate_mat).reshape((-1, self.tot_dim) )        


# load the trajectory data from DCD file
u = mda.Universe(pdb_filename, traj_dcd_filename)

print ('\n[Task 1/2] load trajectory to numpy array...', end='')
# load trajectory to numpy array
trajectory = u.trajectory.timeseries(order='fac')
print ('done.')

# print information of trajectory
print ('\nTrajectory Info:\n\
\tno. of frames in trajectory data: {}\n\
\ttimestep: {:.1f}ps\n\
\ttime length: {:.1f}ps\n\
\tshape of trajectory data array: {}'.format(u.trajectory.n_frames, 
                                  u.trajectory.time, 
                                  u.trajectory.totaltime,
                                  trajectory.shape
                                 )
      )

head_frames = 5
align_selector = "type C or type O or type N"
selected_ids = u.select_atoms(align_selector).ids
print ('\n[Task 2/2] aligning by atoms:')
print (atoms_info.loc[atoms_info['id'].isin(selected_ids)][['id','name', 'type']])

ref_pos = ref.atoms.positions
rmsd_list = []
for idx in range(head_frames):
    rmsd_ret = rms.rmsd(trajectory[idx,selected_ids-1,:], ref_pos[selected_ids-1,:], superposition=False)
    rmsd_list.append(rmsd_ret)

align.AlignTraj(u,  # trajectory to align
                ref,  # reference
                select=align_selector,  # selection of atoms to align
                filename=None,  # file to write the trajectory to
                in_memory=True,
                match_atoms=True,  # whether to match atoms based on mass
               ).run()

print ('\n[Task 1/2] done.')

rmsd_list_aligned = []
for ts in u.trajectory[:head_frames]:
    rmsd_ret = rms.rmsd(u.select_atoms(align_selector).positions, ref.select_atoms(align_selector).positions, superposition=False)
    rmsd_list_aligned.append(rmsd_ret)
    

# -

# #### (optional) display information

# +
#print RMSD values before and after alignment
print ('First {} RMSD values before alignment:\n\t'.format(head_frames), rmsd_list)
print ('\nFirst {} RMSD values after alignment:\n\t'.format(head_frames), rmsd_list_aligned)

#generate Ramachandran plot of two dihedral angles
ax = plt.axes()
r = dihedrals.Ramachandran(u.select_atoms('resid 2')).run()
r.plot(ax, color='black', marker='.') #, ref=True)

# display the trajectory
view = nv.show_mdanalysis(u)
view   


# -

# ## Part 2: Training
#
# #### define neural network model and training function

# +
#Auto encoders class and functions for training.

def create_seqential_nn(layer_dims, activation=torch.nn.Tanh()):
    layers = []
    for i in range(len(layer_dims)-2) :
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1])) 
        layers.append(activation)
    layers.append(torch.nn.Linear(layer_dims[-2], layer_dims[-1])) 
    
    return torch.nn.Sequential(*layers)
       
class Encoder(nn.Module):
    def __init__(self, encoder_dims, atom_indices=None, activation=torch.nn.Tanh()):
        """Initialise auto encoder

        :param encoder_dims: list, List of dimensions for encoder, including input/output layers
        """
        super(Encoder, self).__init__()
        self.atom_indices = atom_indices
        self.encoder = create_seqential_nn(encoder_dims, activation)

    def forward(self, inp):
        # flatten the data
        if self.atom_indices is None: # use all atoms
            inp = torch.flatten(inp, start_dim=1)            
        else: # use selected atoms
            inp = torch.flatten(inp[:,self.atom_indices,:], start_dim=1)
        encoded = self.encoder(inp)
        return encoded

def xi_ae(encoder, x):
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
def train(model, optimizer, traj, weights, train_atom_indices, num_epochs=10, batch_size=32, test_ratio=0.2):
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
    X_train, X_test, w_train, w_test = train_test_split(traj, weights, test_size=test_ratio)  
    X_train = torch.tensor(X_train.astype('float32'))
    X_test = torch.tensor(X_test.astype('float32'))
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
    
    loss_func = torch.nn.MSELoss()
    # --- start the training over the required number of epochs ---
    loss_list = []
    print ("\ntraining starts, %d epochs in total." % num_epochs) 
    for epoch in range(num_epochs):
        # Train the model by going through the whole dataset
        model.train()
        train_loss = []
        for iteration, X in enumerate(train_loader):
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output
            out = model(X)
            # Evaluate loss
            loss = loss_func(out, torch.flatten(X[:,train_atom_indices,:],start_dim=1))
            # Get gradient with respect to parameters of the model
            loss.backward()
            # Store loss
            train_loss.append(loss)
            # Updating parameters
            optimizer.step()
            print (epoch, iteration)
        # Evaluate the test loss on the test dataset
        model.eval()
        with torch.no_grad():
            # Evaluation of test loss
            test_loss = []
            for iteration, X in enumerate(test_loader):
                out = model(X)
                # Evaluate loss
                loss = loss_func(out, torch.flatten(X[:,train_atom_indices,:],start_dim=1))
                # Store loss
                test_loss.append(loss)
            loss_list.append([torch.tensor(train_loss), torch.tensor(test_loss)])

    print ("training ends.\n") 
    return model, loss_list
# -

# #### set training parameters 

# +
def set_seed_all(seed=-1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

#All the parameters are set in the cell below. 
seed = None 
if seed:
    set_seed_all(seed)
    
#set training parameters
batch_size = 10000
num_epochs = 500
learning_rate = 0.005
optimizer_algo = 'Adam'  # Adam by default, otherwise SGD
#dimensions

train_atom_selector = "type C or type O or type N"
train_atom_ids = u.select_atoms(align_selector).ids 
train_atom_indices = train_atom_ids - 1 # minus one, such that the index starts from 0

#input dimension
input_dim = 3 * len(train_atom_ids)
print ('{} Atoms used in define neural network:\n'.format(len(train_atom_ids)), atoms_info.loc[atoms_info['id'].isin(train_atom_ids)][['id','name', 'type']])

# encoded dimension
k = 1
e_layer_dims = [input_dim, 20, 20, k]
d_layer_dims = [k, 20, 20, input_dim]
print ('\nInput dim: {},\tencoded dim: {}\n'.format(input_dim, k))

activation = torch.nn.Tanh()
encoder = Encoder(e_layer_dims, train_atom_indices, activation)
decoder = create_seqential_nn(d_layer_dims, activation)

ae_model = torch.nn.Sequential(encoder, decoder) 

print ('Autoencoder:\n', ae_model)
save_fig_to_file = False
# -

# #### start training 

# +
# Define the optimizer
if optimizer_algo == 'Adam':
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=learning_rate)
else:
    optimizer = torch.optim.SGD(ae_model.parameters(), lr=learning_rate)

ae_model, loss_list = train(ae_model, 
                            optimizer, 
                            trajectory, 
                            np.ones(trajectory.shape[0]), 
                            train_atom_indices,
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
