#!/usr/bin/env python
# +
import numpy as np
import torch
import random

import MDAnalysis as mda
# -


class Align(torch.nn.Module):
    def __init__(self, ref_pos, align_atom_ids):
        super(Align, self).__init__()
        self.align_atom_ids = align_atom_ids 
        self.align_atom_indices = torch.tensor(self.align_atom_ids-1).long() # minus one, such that the index starts from 0
        self.ref_x = torch.from_numpy(ref_pos[self.align_atom_indices, :]).double()        

        # shift reference state 
        ref_c = torch.mean(self.ref_x, 0) 
        self.ref_x = self.ref_x - ref_c

    def show_info(self):
        print ('atom indices used for alignment: ', self.align_atom_indices.numpy())
        print ('\n\treference state used in aligment:\n', self.ref_x.numpy())

    def forward(self, traj):  
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

class Preprocessing(torch.nn.Module):
    
    def __init__(self, feature_mapper, align_layer=torch.nn.Identity()):

        super(Preprocessing, self).__init__()

        self.align = align_layer
        self.feature_mapper = feature_mapper 
        self.feature_dim = feature_mapper.feature_total_dimension()

    def forward(self, inp):
        return self.feature_mapper(self.align(inp))

def create_sequential_nn(layer_dims, activation=torch.nn.Tanh()):
    layers = []
    for i in range(len(layer_dims)-2) :
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1])) 
        layers.append(activation)
    layers.append(torch.nn.Linear(layer_dims[-2], layer_dims[-1])) 

    return torch.nn.Sequential(*layers).double()

class ColVar(torch.nn.Module):
    def __init__(self, preprocessing_layer, layer):
        super(ColVar, self).__init__()
        self.preprocessing_layer = preprocessing_layer
        self.layer = layer
    def forward(self, inp):
        return self.layer(self.preprocessing_layer(inp))

# autoencoder class 
class AutoEncoder(torch.nn.Module):
    def __init__(self, e_layer_dims, d_layer_dims, activation=torch.nn.Tanh()):
        super(AutoEncoder, self).__init__()
        self.encoder = create_sequential_nn(e_layer_dims, activation)
        self.decoder = create_sequential_nn(d_layer_dims, activation)

    def forward(self, inp):
        """TBA
        """
        return self.decoder(self.encoder(inp))

# eigenfunction class
class EigenFunction(torch.nn.Module):
    def __init__(self, layer_dims, k, activation=torch.nn.Tanh()):
        super(EigenFunction, self).__init__()
        assert layer_dims[-1] == 1, "each eigenfunction must be one-dimensional"

        self.eigen_funcs = torch.nn.ModuleList([create_sequential_nn(layer_dims, activation) for idx in range(k)])

    def forward(self, inp):
        """TBA"""

        return torch.cat([nn(inp) for nn in self.eigen_funcs], dim=1)

