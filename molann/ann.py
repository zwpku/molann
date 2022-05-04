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
    r"""ANN layer that performs alignment based on `Kabsch algorithm <http://en.wikipedia.org/wiki/kabsch_algorithm>`__

    Args:
        align_atom_group (:external+mdanalysis:class:`MDAnalysis.core.groups.AtomGroup`): atom
                    group. Specifies coordinates of reference atoms that are used to perform alignment. 


    Let :math:`x_{ref}\in \mathbb{R}^{n_r\times 3}` be the coordinates of the
    reference atoms, where :math:`n_r` is the number of atoms in the atom group. Then, this class defines the map


    .. math::

        x \in \mathbb{R}^{n \times 3} \longrightarrow (x-c(x))A(x) \in \mathbb{R}^{n \times 3}\,,

    where, given a state :math:`x`, :math:`A(x)\in \mathbb{R}^{3\times 3}` and
    :math:`c(x)\in \mathbb{R}^{n\times 3}` are respectively the optimal
    rotation and translation determined (with respect to :math:`x_{ref}`) using the Kabsch algorithm.

    Note that :math:`x_{ref}` will be shifted to have zero mean before it is used to align states.

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

        self.register_buffer('align_atom_indices', torch.tensor(align_atom_group.ids - 1)) # minus one, such that the index starts from 0

        ref_x = torch.from_numpy(align_atom_group.positions)        
        # shift reference state 
        ref_c = torch.mean(ref_x, 0) 
        ref_x = ref_x - ref_c

        self.register_buffer('ref_x', ref_x)

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

        diag_mat = torch.diag(torch.ones(3)).unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)

        sign_vec = torch.sign(torch.linalg.det(torch.matmul(u, vh))).detach()
        diag_mat[:,2,2] = sign_vec

        rotate_mat = torch.bmm(torch.bmm(u, diag_mat), vh)

        aligned_x = torch.matmul(x-x_c, rotate_mat) 
                
        return aligned_x

class FeatureMap(torch.nn.Module):
    r"""ANN that maps coordinates to a feature 

    Args:
        feature (:class:`molann.feature.Feature`): feature that defines the map
        use_angle_value (boolean): if true, use angle value in radians, else
            use sine and/or cosine values. It does not play a role if the
            type of **feature** is 'position'.

    This class defines the feature map

    .. math::

       f: x \in \mathbb{R}^{n \times 3} \longrightarrow f(x) \in \mathbb{R}^{d}\,,

    corresponding to the input feature. 

    Example:

    .. code-block:: python

        import MDAnalysis as mda
        from molann.ann import FeatureMap
        from molann.feature import Feature
        import torch

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
        self.use_angle_value = use_angle_value

        self.register_buffer('atom_indices', torch.tensor(feature.get_atom_indices()-1))

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
    r""" ANN layer that maps coordinates to all features

    Args:
        feature_list (list of :class:`molann.feature.Feature`): list of features 
        use_angle_value (boolean): whether to use angle value in radians 

    This class encapsulates :class:`FeatureMap` and maps coordinates to multiple features.
    More concretely, it defines the map

    .. math::

       x \in \mathbb{R}^{n \times 3} \longrightarrow (f_1(x), f_2(x), \dots, f_l(x))\,,

    where :math:`l` is the number of features in the feature list, and each
    :math:`f_i` is the feature map defined by the class :class:`FeatureMap`.

    Raises:
        AssertionError: if feature_list is empty.

    Example:

    .. code-block:: python

        import MDAnalysis as mda
        from molann.ann import FeatureLayer
        from molann.feature import Feature
        import torch

        # pdb file of the system
        pdb_filename = '/path/to/system.pdb'
        ref = mda.Universe(pdb_filename) 

        f1 = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))
        f2 = Feature('name', 'angle', ref.select_atoms('bynum 1 3 2'))
        f3 = Feature('name', 'bond', ref.select_atoms('bynum 1 3'))

        # define feature layer using features f1, f2 and f3
        f_layer = FeatureLayer([f1, f3, f2], use_angle_value=False)

        print ('output dim=', f_layer.output_dimension())
        x = torch.tensor(ref.atoms.positions).unsqueeze(0)
        print (f_layer(x))
        ff = f_layer.get_feature(0)
        print (f_layer.get_feature_info())

        feature_layer_model_name = 'feature_layer.pt'
        torch.jit.script(f_layer).save(feature_layer_model_name)

    The following code defines an identity feature layer.

    .. code-block:: python

        f4 = Feature('identity', 'position', ref.atoms)
        identity_f_layer = FeatureLayer([f4], use_angle_value=False)
    """

    def __init__(self, feature_list, use_angle_value=False):
        """
        """
        super(FeatureLayer, self).__init__()

        assert len(feature_list) > 0, 'Error: feature list is empty!'

        self.feature_list = feature_list
        self.feature_map_list = torch.nn.ModuleList([FeatureMap(f, use_angle_value) for f in feature_list])

    def get_feature_info(self):
        r"""display information of features 

        Returns:
            :external+pandas:class:`pandas.DataFrame`, information of features
        """
        df = pd.DataFrame()
        for f in self.feature_list:
            df = df.append(f.get_feature_info(), ignore_index=True)
        return df

    def get_feature(self, idx):
        r"""
        Args: 
            idx (int): index of feature in feature list
        Returns:
             :class:`molann.feature.Feature`, the feature in the feature list
        """
        return self.feature_list[idx]

    def output_dimension(self):
        r"""
        Returns: 
            int, total dimension of features in the feature list, or, equivalently,
            the dimension of the output layer of ANN
        """
        return sum([f_map.dim() for f_map in self.feature_map_list])

    def forward(self, x):
        """forward map

        Args:
            x (:external+pytorch:class:`torch.Tensor`): 3d tensor that contains coordinates of states

        Returns:
            :external+pytorch:class:`torch.Tensor`, 2d tensor that contains all features (in the feature list) of states

        This function simply calls :meth:`FeatureMap.forward` for each feature
        in the feature list and then concatenates the tensors.
            
        """
        xf_vec = [fmap(x) for fmap in self.feature_map_list]
        # Features are stored in columns 
        xf = torch.cat(xf_vec, dim=1)
        return xf

class PreprocessingANN(torch.nn.Module):
    """ANN that performs preprocessing of states 

    Args:
        align_layer (:class:`AlignmentLayer` or None): alignment layer 
        feature_layer (:class:`FeatureLayer`): feature layer

    Example:

    .. code-block:: python

        import MDAnalysis as mda
        from molann.ann import FeatureLayer, PreprocessingANN
        from molann.feature import Feature
        import torch

        # pdb file of the system
        pdb_filename = '/path/to/system.pdb'
        ref = mda.Universe(pdb_filename) 

        ag=ref.select_atoms('bynum 1 2 3')

        # define alignment layer
        align = AlignmentLayer(ag)

        # features are just positions of atoms 1,2 and 3.
        f1 = Feature('name', 'position', ag)
        f_layer = FeatureLayer([f1], use_angle_value=False)

        # put together to get the preprocessing layer
        pp_layer = PreprocessingANN(align, f_layer)

        x = torch.tensor(ref.atoms.positions).unsqueeze(0)
        print (pp_layer(x))

    When feature is both translation- and rotation-invariant, alignment is not neccessary:

    .. code-block:: python

        # define feature as dihedral angle 
        f1 = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))
        f_layer = FeatureLayer([f1], use_angle_value=False)

        # since feature is both translation- and rotation-invariant, alignment is not neccessary
        pp_layer = PreprocessingANN(None, f_layer)

    If only alignment is desired, one can provide an identity feature layer when defining :class:`PreprocessingANN`.

    .. code-block:: python

        f = Feature('identity', 'position', ref.atoms)
        identity_f_layer = FeatureLayer([f], use_angle_value=False)
        pp_layer = PreprocessingANN(align, identity_f_layer)

    """
    
    def __init__(self, align_layer, feature_layer):
        """
        """

        super(PreprocessingANN, self).__init__()

        if align_layer is not None :
            self.align_layer = align_layer
        else :
            self.align_layer = torch.nn.Identity()

        self.feature_layer = feature_layer

    def output_dimension(self):
        """
        Return:
            int, the dimension of the output layer
        """
        return self.feature_layer.output_dimension() 

    def forward(self, x):
        """
        forward map that aligns states and then maps to features 

        Args:
            x (:external+pytorch:class:`torch.Tensor`): 3d tensor that contains coordinates of states

        Returns:
            2d :external+pytorch:class:`torch.Tensor` 

        """

        return self.feature_layer(self.align_layer(x))

class MolANN(torch.nn.Module):
    """
    ANN that incoorporates preprocessing layer and the remaining layers which contains training parameters.

    Args:
        preprocessing_layer (:class:`PreprocessingANN`): preprocessing layer
        ann_layers (:external+pytorch:class:`torch.nn.Module`): remaining layers


    Example:

    .. code-block:: python

        import MDAnalysis as mda
        from molann.ann import FeatureLayer, PreprocessingANN, MolANN, create_sequential_nn
        from molann.feature import Feature

        # pdb file of the system
        pdb_filename = '/path/to/system.pdb'
        ref = mda.Universe(pdb_filename) 

        f1 = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))
        f_layer = FeatureLayer([f1], use_angle_value=False)
        pp_layer = PreprocessingANN(None, f_layer)

        output_dim = pp_layer.output_dimension()

        # neural networks layers which contains training parameters 
        nn = create_sequential_nn([output_dim, 5, 3])

        model = MolANN(pp_layer, nn)

    Attributes:
        preprocessing_layer (:class:`PreprocessingANN`)
        ann_layers (:external+pytorch:class:`torch.nn.Module`)

    """
    def __init__(self, preprocessing_layer, ann_layers):
        """
        """
        super(MolANN, self).__init__()
        self.preprocessing_layer = preprocessing_layer
        self.ann_layers = ann_layers

    def get_preprocessing_layer(self):
        """
        Returns:
            :class:`PreprocessingANN`, the preprocessing_layer 
        """
        return self.preprocessing_layer 

    def forward(self, x):
        """
        the forward map
        """
        return self.ann_layers(self.preprocessing_layer(x))

