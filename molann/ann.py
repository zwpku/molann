#!/usr/bin/env python
# +
import torch
# -

def create_sequential_nn(layer_dims, activation=torch.nn.Tanh()):
    """
    TBA
    """
    layers = []
    for i in range(len(layer_dims)-2) :
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1])) 
        layers.append(activation)
    layers.append(torch.nn.Linear(layer_dims[-2], layer_dims[-1])) 

    return torch.nn.Sequential(*layers).double()

class AlignmentLayer(torch.nn.Module):
    """ ANN Layer that performs alignment 
    """

    def __init__(self, ref_pos, align_atom_ids):
        """
        TBA
        """

        super(AlignmentLayer, self).__init__()
        self.align_atom_ids = align_atom_ids 
        self.align_atom_indices = torch.tensor(self.align_atom_ids-1).long() # minus one, such that the index starts from 0
        self.ref_x = torch.from_numpy(ref_pos[self.align_atom_indices, :]).double()        

        # shift reference state 
        ref_c = torch.mean(self.ref_x, 0) 
        self.ref_x = self.ref_x - ref_c

    def show_info(self):
        """
        TBA
        """
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

class FeatureMap(torch.nn.Module):
    """ANN that maps coordinates to a feature 
    """

    def __init__(self, feature, use_angle_value=False):
        """
        TBA
        """
        super(FeatureMap, self).__init__()
        self.feature = feature
        self.type_id = feature.type_id() 
        self.ag = torch.tensor(feature.atom_group()-1)  # minus one, so that it starts from 0
        self.use_angle_value = use_angle_value

    def dim(self):
        r"""return total dimension of features
        """
        output_dim = 0
        self.type_id = self.type_id_list[i]
        if self.type_id == 0 or self.type_id == 1 : # angle or bond
            output_dim = 1 
        if self.type_id == 2 : # dihedral angle
            if self.use_angle_value == True :
                output_dim = 1 
            else :
                output_dim = 2 
        if self.type_id == 3 : # position 
            output_dim = 3 * len(self.feature.atom_group())
        return output_dim 

    def forward(self, x):
        r"""map position to feature 

        Parameters
        ----------
        x : torch tensor 
            coordinates of state

        """

        ag = self.ag
        ret = None

        if self.type_id == 0 : # angle
            r21 = x[:, ag[0], :] - x[:, ag[1], :]
            r23 = x[:, ag[2], :] - x[:, ag[1], :]
            r21l = torch.norm(r21, dim=1, keepdim=True)
            r23l = torch.norm(r23, dim=1, keepdim=True)
            cos_angle = (r21 * r23).sum(dim=1, keepdim=True) / (r21l * r23l)
            if self.use_angle_value :
                ret = torch.acos(cos_angle)
            else :
                ret = cos_angle

        if self.type_id == 1 : # bond length
            r12 = x[:, ag[1], :] - x[:, ag[0], :]
            ret = torch.norm(r12, dim=1, keepdim=True)

        if self.type_id == 2 : # dihedral angle
            r12 = x[:, ag[1], :] - x[:, ag[0], :]
            r23 = x[:, ag[2], :] - x[:, ag[1], :]
            r34 = x[:, ag[3], :] - x[:, ag[2], :]
            n1 = torch.cross(r12, r23)
            n2 = torch.cross(r23, r34)
            cos_phi = (n1*n2).sum(dim=1, keepdim=True)
            sin_phi = (n1 * r34).sum(dim=1, keepdim=True) * torch.norm(r23, dim=1, keepdim=True)
            radius = torch.sqrt(cos_phi**2 + sin_phi**2)

            if self.use_angle_value :
                ret = torch.atan2(sin_phi, cos_phi)
            else :
                ret = torch.cat((cos_phi / radius, sin_phi / radius), dim=1)

        if self.type_id == 3: # atom_position 
            ret = x[:, ag, :].reshape((-1, len(ag) * 3))

        return ret 

    def forward(self, x):
        """forward map
        """
        xf = self.map_to_feature(x, 0)
        for i in range(1, len(self.type_id_list)) :
            # Features are stored in columns 
            xf = torch.cat((xf, self.map_to_feature(x, i)), dim=1)
        return xf

class FeatureLayer(torch.nn.Module):
    """
    ANN layer that map coordinates to all features
    """

    def __init__(self, feature_list, use_angle_value=False):
        """
        TBA
        """
        super(FeatureLayer, self).__init__()
        self.feature_list = feature_list
        self.feature_map_list = [FeatureMap(f, use_angle_value) for f in feature_list]

    def info(self, info_title):
        r"""display information of features 

        Parameters
        ----------
        info_title : str
            texts to print before displaying information of features
        """

        print (f'{info_title}Id.\tName\tType\tAtomIDs')
        for idx, f in enumerate(self.feature_list) :
            print ( '{}\t{}\t{}\t{}'.format(idx, f.name(), f.type(), f.atom_group()) )

    def feature_name(self, idx):
        r"""return the name of feature 

        Parameters
        ----------
        idx : int
            index of feature
        """
        return self.feature_list[idx].name()

    def all_feature_names(self):
        r"""return the list of all feature names 
        """
        return [f.name() for f in self.feature_list]

    def output_dimension(self):
        r"""return total dimension of features
        """
        return np.sum([f_map.dim() for f_map in self.feature_map_list])

    def forward(self, x):
        """forward map
        """
        xf = self.feature_map_list[0](x)
        for i in range(1, len(self.feature_map_list)) :
            # Features are stored in columns 
            xf = torch.cat((xf, self.feature_map_list[i](x)), dim=1)
        return xf

class IdentityFeatureLayer(torch.nn.Module):
    """
    Feature layer that corresponds to the identity map
    """

    def __init__(self, dim):
        """
        TBA
        """
        self.dim = dim

    def output_dimension(self):
        """
        return the dimension of the output layer
        """
        return self.dim

    def forward(self, x):
        """
        forward map
        """
        return x

class PreprocessingANN(torch.nn.Module):
    """Preprocessing ANN
    """
    
    def __init__(self, align_layer, feature_layer):
        """
        TBA
        """

        super(PreprocessingANN, self).__init__()

        if align_layer is not None :
            self.align_layer = align_layer
        else :
            self.align_layer = torch.nn.Identity()

        self.feature_layer = feature_layer

    def output_dimension(self):
        """
        Return the dimension of the output layer
        """
        return self.feature_layer.output_dimension() 

    def forward(self, inp):
        """
        forward map
        """

        return self.feature_layer(self.align_layer(inp))

class MolANN(torch.nn.Module):
    """
    ANN that incoorporates alignment and feature 
    """
    def __init__(self, preprocessing_layer, ann_layers):
        """
        TBA
        """
        super(MolANN, self).__init__()
        self.preprocessing_layer = preprocessing_layer
        self.ann_layers = ann_layers

    def preprocessing_layer(self):
        """
        return the preprocessing_layer 
        """
        return self.preprocessing_layer 

    def forward(self, inp):
        """
        forward map
        """
        return self.ann_layers(self.preprocessing_layer(inp))

