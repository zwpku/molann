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
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import time
from datetime import datetime
# -

class Trajectory(object):
    def __init__(self, pdb_filename, traj_dcd_filename, beta=1.0, weight_filename=None):
        # load the trajectory data from DCD file
        self.u = mda.Universe(pdb_filename, traj_dcd_filename)

        # load the reference configuration from the PDB file
        self.ref = mda.Universe(pdb_filename) 

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

        if weight_filename :
            self.weights = self.load_weights(weight_filename)
            # normalize
            self.weights = self.weights / np.mean(self.weights)
        else :
            self.weights = np.ones(self.n_frames)

    def load_traj(self):

        print ('\nloading trajectory to numpy array...', end='')
        # load trajectory to torch tensor 
        self.trajectory = torch.from_numpy(self.u.trajectory.timeseries(order='fac')).double()

        print ('done.')

        self.start_time = self.u.trajectory.time
        self.dt = self.u.trajectory.dt
        self.n_frames = self.u.trajectory.n_frames

        # print information of trajectory
        print ('\nTrajectory Info:\n\
        \tno. of frames in trajectory data: {}\n\
        \tstepsize: {:.1f}ps\n\
        \ttime of first frame: {:.1f}ps\n\
        \ttime length: {:.1f}ps\n\
        \tshape of trajectory data array: {}\n'.format(self.n_frames, 
                                          self.dt, 
                                          self.start_time, 
                                          self.u.trajectory.totaltime,
                                          self.trajectory.shape
                                         )
              )


    def load_weights(self, weight_filename):
        print ('\nloading weights from file: ', weight_filename)
        time_weight_vec = pd.read_csv(weight_filename)
        time_weight_vec['weight'] /= time_weight_vec['weight'].mean()
        print ('\n', time_weight_vec.head(8))
        time_weight_vec = time_weight_vec.to_numpy()
        if self.start_time - time_weight_vec[0,0] > 0.01 or self.n_frames != time_weight_vec.shape[0] :
            print ('Error: time in weight file does match the trajectory data!\n')
            exit(0)
        # weights are in the second column
        return time_weight_vec[:,1]

class Feature(object):
    def __init__(self, name, feature_type, ag):

        if feature_type not in ['angle', 'bond', 'dihedral', 'position']:
            raise NotImplementedError(f'feature {feature_type} not implemented')

        if feature_type == 'angle': 
            assert len(ag)==3, '3 atoms are needed to define an angle, {} provided'.format(len(ag))
            type_id = 0 
        if feature_type == 'bond': 
            assert len(ag)==2, '2 atoms are needed to define a bond length, {} provided'.format(len(ag))
            type_id = 1 
        if feature_type == 'dihedral': 
            assert len(ag)==4, '4 atoms are needed to define a dihedral angle, {} provided'.format(len(ag))
            type_id = 2 
        if feature_type == 'position':
            type_id = 3 

        self.name = name
        self.type_name = feature_type
        self.atom_group = ag
        self.type_id = type_id

class FeatureFileReader(object):
    def __init__(self, feature_file, section_name, universe, ignore_position_feature=False, use_all_positions_by_default=False):

        self.feature_file = feature_file
        self.section_name = section_name
        self.use_all_positions_by_default = use_all_positions_by_default
        self.ignore_position_feature = ignore_position_feature
        self.u = universe

    def read(self):

        feature_list = []

        pp_cfg_file = open(self.feature_file, "r")
        in_section = False

        for line in pp_cfg_file:
            line = line.strip()

            if not line or line.startswith("#") : 
                continue 

            if line.startswith("["):
                if line.strip('[]') == self.section_name :
                    in_section = True
                    continue 
                if in_section and line.strip('[]') == 'End':
                    break

            if in_section :
                feature_name, feature_type, selector = line.split(',')

                ag = self.u.select_atoms(selector).ids
                feature = Feature(feature_name.strip(), feature_type.strip(), ag)

                if feature.type_name == 'position' and self.ignore_position_feature :
                    print (f'Position feature in section {self.section_name} ignored:\t {line}')
                    continue

                feature_list.append(feature)

        pp_cfg_file.close()

        if len(feature_list) == 0 and self.use_all_positions_by_default : ## in this case, positions of all atoms will be used.
            print ("No valid features found, use positions of all atoms.\n") 
            ag = self.u.atoms.ids 
            feature = Feature('all', 'position', ag)
            feature_list.append(feature)

        return feature_list

class FeatureMap(torch.nn.Module):
    def __init__(self, feature_list, use_angle_value=False):
        super(FeatureMap, self).__init__()
        self.num_features = len(feature_list)
        self.name_list = [f.name for f in feature_list]
        self.type_list = [f.type_name for f in feature_list]
        self.type_id_list = [f.type_id for f in feature_list]
        self.ag_list = [torch.tensor(f.atom_group-1) for f in feature_list] # minus one, so that it starts from 0
        self.use_angle_value = use_angle_value

    def info(self, info_title):
        print (f'{info_title}Id.\tName\tType\tAtomIDs')
        for idx in range(self.num_features):
            print ('{}\t{}\t{}\t{}'.format(idx, self.name_list[idx], self.type_list[idx], self.ag_list[idx].numpy()))

    def feature_name(self, idx):
        return self.name_list[idx]

    def feature_all_names(self):
        return self.name_list

    def feature_total_dimension(self):
        output_dim = 0
        for i in range(len(self.type_id_list)) :
            feature_id = self.type_id_list[i]
            if feature_id == 0 : 
                output_dim += 1 
            if feature_id == 1 : 
                output_dim += 1 
            if feature_id == 2 : 
                if self.use_angle_value == True :
                    output_dim += 1 
                else :
                    output_dim += 2 
            if feature_id == 3 :
                output_dim += 3 * len(self.ag_list[i])
        return output_dim 

    def map_to_feature(self, x, idx: int):
        feature_id = self.type_id_list[idx]
        ag = self.ag_list[idx]

        if feature_id == 0 : # angle
            r21 = x[:, ag[0], :] - x[:, ag[1], :]
            r23 = x[:, ag[2], :] - x[:, ag[1], :]
            r21l = torch.norm(r21, dim=1, keepdim=True)
            r23l = torch.norm(r23, dim=1, keepdim=True)
            cos_angle = (r21 * r23).sum(dim=1, keepdim=True) / (r21l * r23l)
            if self.use_angle_value :
                return torch.acos(cos_angle)
            else :
                return cos_angle

        if feature_id == 1 : # bond length
            r12 = x[:, ag[1], :] - x[:, ag[0], :]
            return torch.norm(r12, dim=1, keepdim=True)

        if feature_id == 2 : # dihedral angle
            r12 = x[:, ag[1], :] - x[:, ag[0], :]
            r23 = x[:, ag[2], :] - x[:, ag[1], :]
            r34 = x[:, ag[3], :] - x[:, ag[2], :]
            n1 = torch.cross(r12, r23)
            n2 = torch.cross(r23, r34)
            cos_phi = (n1*n2).sum(dim=1, keepdim=True)
            sin_phi = (n1 * r34).sum(dim=1, keepdim=True) * torch.norm(r23, dim=1, keepdim=True)
            radius = torch.sqrt(cos_phi**2 + sin_phi**2)

            if self.use_angle_value :
                return torch.atan2(sin_phi, cos_phi)
            else :
                return torch.cat((cos_phi / radius, sin_phi / radius), dim=1)

        if feature_id == 3: # atom_position 
            return x[:, ag, :].reshape((-1, len(ag) * 3))

        raise RuntimeError()

    def forward(self, x):
        xf = self.map_to_feature(x, 0)
        for i in range(1, len(self.type_id_list)) :
            # Features are stored in columns 
            xf = torch.cat((xf, self.map_to_feature(x, i)), dim=1)
        return xf

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
        return self.decoder(self.encoder(inp))

# eigenfunction class
class EigenFunction(torch.nn.Module):
    def __init__(self, layer_dims, k, activation=torch.nn.Tanh()):
        super(EigenFunction, self).__init__()
        assert layer_dims[-1] == 1, "each eigenfunction must be one-dimensional"

        self.eigen_funcs = torch.nn.ModuleList([create_sequential_nn(layer_dims, activation) for idx in range(k)])

    def forward(self, inp):
        return torch.cat([nn(inp) for nn in self.eigen_funcs], dim=1)

