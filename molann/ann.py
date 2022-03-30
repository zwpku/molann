r"""Artificial Neural networks for Molecular System --- :mod:`molann.ann`
========================================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements several PyTorch artificial neural network (ANN)
classes, i.e. derived classes of :external+pytorch:class:`torch.nn.Module`, 
which take into acount alignment, as well as features of molecular system.

Classes
-------
.. autoclass:: AlignmentLayer
    :members:

.. autoclass:: FeatureMap
    :members:

.. autoclass:: FeatureLayer
    :members:

.. autoclass:: PreprocessingANN
    :members:

.. autoclass:: MolANN
    :members:

.. autofunction:: create_sequential_nn

"""

import torch
import pandas as pd

def create_sequential_nn(layer_dims, activation=torch.nn.Tanh()):
    r""" Construct a feedforward Pytorch neural network

    :param layer_dims: dimensions of layers 
    :type layer_dims: list of int
    :param activation: PyTorch non-linear activation function

    :raises AssertionError: if length of **layer_dims** is not larger than 1.

    Example
    -------

    .. code-block:: python

        from molann.ann import create_sequential_nn
        import torch

        nn1 = create_sequential_nn([10, 5, 1])
        nn2 = create_sequential_nn([10, 2], activation=torch.nn.ReLU())
    """

    assert len(layer_dims) >= 2, 'Error: at least 2 layers are needed to define a neural network (length={})!'.format(len(layer_dims))
    layers = []
    for i in range(len(layer_dims)-2) :
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1])) 
        layers.append(activation)
    layers.append(torch.nn.Linear(layer_dims[-2], layer_dims[-1])) 

    return torch.nn.Sequential(*layers)

class AlignmentLayer(torch.nn.Module):
    """ ANN layer that performs alignment based on Kabsch algorithm 

    Args:
        align_atom_group (:external+mdanalysis:class:`MDAnalysis.core.groups.AtomGroup`): atom
                    group. Specifies coordinates of reference atoms to perform alignment. 

    Example:

    .. code-block:: python

        import torch
        import MDAnalysis as mda
        from molann.ann import AlignmentLayer

        # pdb file of the system
        pdb_filename = '/path/to/system.pdb'
        ref = mda.Universe(pdb_filename) 
        ag=ref.select_atoms('bynum 1 2 3')

        align = AlignmentLayer(ag)
        align.show_info()

        # for illustration, use the state in the pdb file (length 1)
        x = torch.tensor(ref.atoms.positions).unsqueeze(0)
        print (align(x))

        # save the model to file
        align_model_name = 'algin.pt'
        torch.jit.script(align).save(align_model_name)
    """

    def __init__(self, align_atom_group):
        """
        """

        super(AlignmentLayer, self).__init__()
        self.align_atom_indices = torch.tensor(align_atom_group.ids - 1).long() # minus one, such that the index starts from 0
        self.ref_x = torch.from_numpy(align_atom_group.positions)        

        # shift reference state 
        ref_c = torch.mean(self.ref_x, 0) 
        self.ref_x = self.ref_x - ref_c

    def show_info(self):
        """
        display indices and positions of reference atoms that are used to perform alignment
        """
        print ('\natom indices used for alignment: \n', self.align_atom_indices.numpy())
        print ('\npositions of reference state used in aligment:\n', self.ref_x.numpy())

    def forward(self, x):  
        """
        align states by translation and rotation. 

        Args: 
            x (:external+pytorch:class:`torch.Tensor`): states to be aligned

        Returns:
            :external+pytorch:class:`torch.Tensor` that stores the aligned states

        **x** should be a 3d tensor, whose shape is :math:`[l, N, 3]`,
        where :math:`l` is the number of states in **x** and :math:`N` is the total number of atoms in the system.
        The returned tensor has the same shape.

        This method implements the Kabsch algorithm.
        """
                         
        traj_selected_atoms = x[:, self.align_atom_indices, :]
        # centers
        x_c = torch.mean(traj_selected_atoms, 1, True)
        # translation
        x_notran = traj_selected_atoms - x_c 
        
        xtmp = x_notran.permute((0,2,1))
        prod = torch.matmul(xtmp, self.ref_x) # dimension: traj_length x 3 x 3
        u, s, vh = torch.linalg.svd(prod)

        diag_mat = torch.diag(torch.ones(3)).unsqueeze(0).repeat(x.size(0), 1, 1)

        sign_vec = torch.sign(torch.linalg.det(torch.matmul(u, vh))).detach()
        diag_mat[:,2,2] = sign_vec

        rotate_mat = torch.bmm(torch.bmm(u, diag_mat), vh)

        aligned_x = torch.matmul(x-x_c, rotate_mat) 
                
        return aligned_x

class FeatureMap(torch.nn.Module):
    """ANN that maps coordinates to a feature 

    Args:
        feature (:class:`molann.feature.Feature`): feature that defines the map
        use_angle_value (boolean): if true, use angle value in radians, else
            use sine and/or cosine values. It does not play a role if the
            type of **feature** is 'position'.

    Example:

    .. code-block:: python

        import MDAnalysis as mda
        from molann.ann import FeatureMap
        from molann.feature import Feature

        # pdb file of the system
        pdb_filename = '/path/to/system.pdb'
        ref = mda.Universe(pdb_filename) 

        f = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))
        fmap = FeatureMap(f, use_angle_value=False)
        print ('dim=', fmap.dim())

        x = torch.tensor(ref.atoms.positions).unsqueeze(0)
        print (fmap(x))
        feature_model_name = 'feature_map.pt'
        torch.jit.script(fmap).save(feature_model_name)

    """

    def __init__(self, feature, use_angle_value=False):
        """
        """
        super(FeatureMap, self).__init__()
        self.feature = feature
        self.type_id = feature.get_type_id() 
        self.atom_indices = torch.tensor(feature.get_atom_indices()-1)  # minus one, so that it starts from 0
        self.use_angle_value = use_angle_value

    def dim(self):
        r"""
        Return: 
            int, total dimension of features or, equivalently, the dimension of the output layer of the ANN.

        The dimension equals 1 for 'angle' and 'bond', as well as for
        'dihedral' when **use_angle_value** =True.

        The dimension equals 2 for 'dihedral', when **use_angle_value** =False.

        The dimension equals :math:`3\times n` for 'position', where :math:`n`
        is the number of atoms involved in **feature** .
        """
        output_dim = 0
        if self.type_id == 0 or self.type_id == 1 : # angle or bond
            output_dim = 1 
        if self.type_id == 2 : # dihedral angle
            if self.use_angle_value == True :
                output_dim = 1 
            else :
                output_dim = 2 
        if self.type_id == 3 : # position 
            output_dim = 3 * len(self.feature.get_atom_indices())
        return output_dim 

    def forward(self, x):
        r"""map position to feature 

        Args:
            x (:external+pytorch:class:`torch.Tensor`): 3d tensor that contains coordinates of states

        Returns:
            :external+pytorch:class:`torch.Tensor`, 2d tensor that contains features of the states

        The shape of the return tensor is :math:`[l, d]`, where :math:`l` is
        the number of states in **x** and :math:`d` is the dimension returned by :meth:`dim`.

        For 'angle', if use_angle_value=True, it returns angle values in
        :math:`[0, \pi]`; otherwise, it retuns the cosine values of the angles.  

        For 'dihedral', if use_angle_value=True, it returns angle values in
        :math:`[-\pi, \pi]`; otherwise, it retuns [cosine, sine] of the angles.  

        For 'position', it returns the coordinates of all the atoms in the feature.

        """

        atom_indices = self.atom_indices

        ret = torch.tensor(0.0)

        if self.type_id == 0 : # angle
            r21 = x[:, atom_indices[0], :] - x[:, atom_indices[1], :]
            r23 = x[:, atom_indices[2], :] - x[:, atom_indices[1], :]
            r21l = torch.norm(r21, dim=1, keepdim=True)
            r23l = torch.norm(r23, dim=1, keepdim=True)
            cos_angle = (r21 * r23).sum(dim=1, keepdim=True) / (r21l * r23l)
            if self.use_angle_value :
                ret = torch.acos(cos_angle)
            else :
                ret = cos_angle

        if self.type_id == 1 : # bond length
            r12 = x[:, atom_indices[1], :] - x[:, atom_indices[0], :]
            ret = torch.norm(r12, dim=1, keepdim=True)

        if self.type_id == 2 : # dihedral angle
            r12 = x[:, atom_indices[1], :] - x[:, atom_indices[0], :]
            r23 = x[:, atom_indices[2], :] - x[:, atom_indices[1], :]
            r34 = x[:, atom_indices[3], :] - x[:, atom_indices[2], :]
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
            ret = x[:, atom_indices, :].reshape((-1, len(atom_indices) * 3))

        return ret 

class FeatureLayer(torch.nn.Module):
    """
    ANN layer that map coordinates to all features
    """

    def __init__(self, feature_list, use_angle_value=False):
        """
        TBA
        """
        super(FeatureLayer, self).__init__()

        assert len(feature_list) > 0, 'Error: feature list is empty!'

        self.feature_list = feature_list
        self.feature_map_list = torch.nn.ModuleList([FeatureMap(f, use_angle_value) for f in feature_list])

    def get_feature_info(self):
        r"""display information of features 
        """
        df = pd.DataFrame()
        for f in self.feature_list:
            df = df.append(f.get_feature_info(), ignore_index=True)
        return df

    def get_feature(self, idx):
        r"""return the name of feature 

        Parameters
        ----------
        idx : int
            index of feature
        """
        return self.feature_list[idx]

    def output_dimension(self):
        r"""return total dimension of features
        """
        return sum([f_map.dim() for f_map in self.feature_map_list])

    def forward(self, x):
        """forward map
        """
        xf_vec = [fmap(x) for fmap in self.feature_map_list]
        # Features are stored in columns 
        xf = torch.cat(xf_vec, dim=1)
        return xf

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

    def get_preprocessing_layer(self):
        """
        return the preprocessing_layer 
        """
        return self.preprocessing_layer 

    def forward(self, inp):
        """
        forward map
        """
        return self.ann_layers(self.preprocessing_layer(inp))

