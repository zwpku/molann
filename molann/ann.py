#!/usr/bin/env python
# +
import torch
# -

def create_sequential_nn(layer_dims, activation=torch.nn.Tanh()):
    layers = []
    for i in range(len(layer_dims)-2) :
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1])) 
        layers.append(activation)
    layers.append(torch.nn.Linear(layer_dims[-2], layer_dims[-1])) 

    return torch.nn.Sequential(*layers).double()

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

class PreprocessingANN(torch.nn.Module):
    
    def __init__(self, feature_mapper, align_layer=torch.nn.Identity()):

        super(PreprocessingANN, self).__init__()

        self.align = align_layer
        self.feature_mapper = feature_mapper 
        self.feature_dim = feature_mapper.feature_total_dimension()

    def forward(self, inp):
        return self.feature_mapper(self.align(inp))

class MolANN(torch.nn.Module):
    def __init__(self, preprocessing_layer, ann_layers):
        super(MolANN, self).__init__()
        self.preprocessing_layer = preprocessing_layer
        self.ann_layers = ann_layers
    def forward(self, inp):
        return self.ann_layers(self.preprocessing_layer(inp))

