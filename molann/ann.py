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

    layers = torch.nn.Sequential()

    for i in range(len(layer_dims)-2) :
        layers.add_module('%dth_layer' % (i+1), torch.nn.Linear(layer_dims[i], layer_dims[i+1])) 
        layers.add_module('activation of %dth_layer' % (i+1), activation)
    layers.add_module('%dth_layer' % (len(layer_dims)-1), torch.nn.Linear(layer_dims[-2], layer_dims[-1])) 

    return layers

class AlignmentLayer(torch.nn.Module):
    r"""ANN layer that performs alignment based on `Kabsch algorithm <http://en.wikipedia.org/wiki/kabsch_algorithm>`__

    Args:
        align_atom_group (:external+mdanalysis:class:`MDAnalysis.core.groups.AtomGroup`): Specifies coordinates of reference atoms that are used to perform alignment. 
        input_atom_group (:external+mdanalysis:class:`MDAnalysis.core.groups.AtomGroup`): Specifies those atoms that are used as input. 


    Let :math:`x_{ref}\in \mathbb{R}^{n_r\times 3}` be the coordinates of the
    reference atoms, where :math:`n_r` is the number of atoms in the atom group `align_atom_group`. Then, this class defines the map

    .. math::

        x \in \mathbb{R}^{n_{inp} \times 3} \longrightarrow (x-c(x))A(x) \in \mathbb{R}^{n_{inp} \times 3}\,,

    where, given coordinates :math:`x` of :math:`n_{inp}` atoms, :math:`A(x)\in \mathbb{R}^{3\times 3}` and :math:`c(x)\in \mathbb{R}^{n_{inp}\times 3}` are respectively the optimal rotation and translation determined (with respect to :math:`x_{ref}`) using the Kabsch algorithm.

    Note that :math:`x_{ref}` will be shifted to have zero mean before it is used to align states.


    Example:

    .. code-block:: python

        import torch
        import MDAnalysis as mda
        from molann.ann import AlignmentLayer

        # pdb file of the system
        pdb_filename = '/path/to/system.pdb'
        ref = mda.Universe(pdb_filename) 
        ag=ref.select_atoms('bynum 1 2 5')
        input_ag = ref.atoms

        align = AlignmentLayer(ag, input_ag)
        align.show_info()

        # for illustration, use the state in the pdb file (length 1)
        x = torch.tensor(ref.atoms.positions).unsqueeze(0)
        print (align(x))

        # save the model to file
        align_model_name = 'algin.pt'
        torch.jit.script(align).save(align_model_name)


    Attributes:
        align_atom_indices (list of int): indices of atoms used to align coordinates.
        input_atom_indices (list of int): indices of atoms in the input tensor.
        input_atom_num (int): atom number (i.e. :math:`n_{inp}`) in the input tensor.


    """

    def __init__(self, align_atom_group, input_atom_group):
        r"""
        Raises:
            ValueError: if some reference atom is not in the atom group *input_atom_group*.
        """

        super(AlignmentLayer, self).__init__()

        self.align_atom_indices = (align_atom_group.ids - 1).tolist() # minus one, such that the index starts from 0
        self.input_atom_indices = (input_atom_group.ids - 1).tolist()
        self.input_atom_num = len(input_atom_group)

        self.ref_x = torch.from_numpy(align_atom_group.positions)        
        # shift reference state 
        ref_c = torch.mean(self.ref_x, 0) 
        self.ref_x = self.ref_x - ref_c

        try:
            self._local_align_atom_indices = [self.input_atom_indices.index(idx) for idx in self.align_atom_indices]
        except ValueError :
            raise ValueError("Atoms used for alignment must be among the input")

    def show_info(self):
        """
        display indices of input atoms, indices and positions of the reference atoms that are used to perform alignment
        """
        print (f'\n{self.input_atom_num} atoms used for input, (0-based) global indices: \n', self.input_atom_indices)
        print (f'\n{len(self._local_align_atom_indices)} atoms used for alignment, with (0-based) global indices: \n', self.align_atom_indices)
        print ('local indices\n', self._local_align_atom_indices)
        print ('\npositions of reference state used in aligment:\n', self.ref_x.numpy())

    def forward(self, x):  
        """
        align states by translation and rotation. 

        Args: 
            x (:external+pytorch:class:`torch.Tensor`): states to be aligned

        Returns:
            :external+pytorch:class:`torch.Tensor` that stores the aligned states

        Raises:
            AssertionError: if `x` is not a Torch tensor with sizes :math:`[*, n_{inp},3]`.

        **x** should be a 3d tensor, whose shape is :math:`[l, n_{inp}, 3]`, where :math:`l` is the number of states in **x** and :math:`n_{inp}` is the total number of atoms in the atom group *input_atom_group*. The returned tensor has the same shape.

        This method implements the Kabsch algorithm.
        """

        assert isinstance(x, torch.Tensor), 'Input x is not a torch tensor'

        assert x.size(1) == self.input_atom_num and x.size(2) == 3, f'Input should be a 3d torch tensor, with sizes [*, {self.input_atom_num}, 3]. Actual sizes: {x.shape}'
                         
        traj_selected_atoms = x[:, self._local_align_atom_indices, :]
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
        input_atom_group (:external+mdanalysis:class:`MDAnalysis.core.groups.AtomGroup`): Specifies those atoms that are used as input. 
        use_angle_value (boolean): if true, use angle value in radians, else
            use sine and/or cosine values. It does not play a role if the
            type of **feature** is 'position'.

    This class defines the feature map

    .. math::

       f: x \in \mathbb{R}^{n_{inp} \times 3} \longrightarrow f(x) \in \mathbb{R}^{d}\,,

    corresponding to the input feature, where :math:`n_{inp}` is the number of atoms provided in the input.

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
        input_ag = ref.select_atoms('bynum 1 2 3 4 5')
        fmap = FeatureMap(f, input_ag, use_angle_value=False)
        print ('dim=', fmap.dim())

        x = torch.tensor(input_ag.positions).unsqueeze(0)
        print (fmap(x))
        feature_model_name = 'feature_map.pt'
        torch.jit.script(fmap).save(feature_model_name)

    """

    def __init__(self, feature, input_atom_group, use_angle_value=False):
        """
        Raises:
            ValueError: if some atom used to define feature is not in the atom group for input.
        """
        super(FeatureMap, self).__init__()

        self.feature = feature
        self.type_id = feature.get_type_id()
        self.use_angle_value = use_angle_value

        self.input_atom_indices = (input_atom_group.ids - 1).tolist()
        self.input_atom_num = len(input_atom_group)

        atom_indices = feature.get_atom_indices()-1

        try:
            self._local_atom_indices = [self.input_atom_indices.index(idx) for idx in atom_indices]
        except ValueError :
            raise ValueError("Atoms used in feature must be among the input")

    def dim(self):
        r"""
        Return: 
            int, total dimension of features or, equivalently, the dimension of the output layer of the ANN.

        The dimension equals 1 for 'angle' and 'bond', as well as for
        'dihedral' when **use_angle_value** =True.

        The dimension equals 2 for 'dihedral', when **use_angle_value** =False.

        The dimension equals :math:`3n` for 'position', where :math:`n`
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


        **x** should be a 3d tensor, whose shape is :math:`[l, n_{inp}, 3]`, where :math:`l` is the number of states in **x** and :math:`n_{inp}` is the total number of atoms in the atom group *input_atom_group*. 

        The shape of the return tensor is :math:`[l, d]`, where :math:`l` is
        the number of states in **x** and :math:`d` is the dimension returned by :meth:`dim`.

        For 'angle', if use_angle_value=True, it returns angle values in
        :math:`[0, \pi]`; otherwise, it retuns the cosine values of the angles.  

        For 'dihedral', if use_angle_value=True, it returns angle values in
        :math:`[-\pi, \pi]`; otherwise, it retuns [cosine, sine] of the angles.  

        For 'position', it returns the coordinates of all the atoms in the feature.

        Raises:
            AssertionError: if `x` is not a Torch tensor with sizes :math:`[*, n_{inp},3]`.

        """

        assert isinstance(x, torch.Tensor), 'Input x is not a torch tensor'

        assert x.size(1) == self.input_atom_num and x.size(2) == 3, f'Input should be a 3d torch tensor, with sizes [*, {self.input_atom_num}, 3]. Actual sizes: {x.shape}'

        atom_indices = self._local_atom_indices

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
    r""" ANN layer that maps coordinates to all features in a feature list

    Args:
        feature_list (list of :class:`molann.feature.Feature`): list of features 
        input_atom_group (:external+mdanalysis:class:`MDAnalysis.core.groups.AtomGroup`): Specifies those atoms that are used as input. 
        use_angle_value (boolean): whether to use angle value in radians 

    This class encapsulates :class:`FeatureMap` and maps input coordinates to multiple features.
    More concretely, it defines the map

    .. math::

       x \in \mathbb{R}^{n_{inp} \times 3} \longrightarrow (f_1(x), f_2(x), \dots, f_l(x))\,,

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

        input_ag = ref.select_atoms('bynum 1 2 3 4 5 6')
        # define feature layer using features f1, f2 and f3
        f_layer = FeatureLayer([f1, f3, f2], input_ag, use_angle_value=False)

        print ('output dim=', f_layer.output_dimension())
        x = torch.tensor(input_ag.positions).unsqueeze(0)
        print (f_layer(x))
        ff = f_layer.get_feature(0)
        print (f_layer.get_feature_info())

        feature_layer_model_name = 'feature_layer.pt'
        torch.jit.script(f_layer).save(feature_layer_model_name)

    The following code defines an identity feature layer (for the first three atoms).

    .. code-block:: python

        ag = ref.select_atoms('bynum 1 2 3')
        f4 = Feature('identity', 'position', ag)
        identity_f_layer = FeatureLayer([f4], ag, use_angle_value=False)
    """

    def __init__(self, feature_list, input_atom_group, use_angle_value=False):
        """
        """
        super(FeatureLayer, self).__init__()

        assert len(feature_list) > 0, 'Error: feature list is empty!'

        self.feature_list = feature_list
        self.feature_map_list = torch.nn.ModuleList([FeatureMap(f, input_atom_group, use_angle_value) for f in feature_list])

        self.input_atom_num = len(input_atom_group)

    def get_feature_info(self):
        r"""display information of features 

        Returns:
            :external+pandas:class:`pandas.DataFrame`, information of features
        """
        return pd.concat([f.get_feature_info() for f in self.feature_list], ignore_index=True)

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

        assert isinstance(x, torch.Tensor), 'Input x is not a torch tensor'

        assert x.size(1) == self.input_atom_num and x.size(2) == 3, f'Input should be a 3d torch tensor, with sizes [*, {self.input_atom_num}, 3]. Actual sizes: {x.shape}'

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
        input_ag=ref.select_atoms('bynum 1 2 3 4 5 6 7')

        # define alignment layer
        align = AlignmentLayer(ag, input_ag)

        # features are just positions of atoms 1,2 and 3.
        f1 = Feature('name', 'position', ag)
        f_layer = FeatureLayer([f1], input_ag, use_angle_value=False)

        # put together to get the preprocessing layer
        pp_layer = PreprocessingANN(align, f_layer)

        x = torch.tensor(input_ag.positions).unsqueeze(0)
        print (pp_layer(x))

    When feature is both translation- and rotation-invariant, alignment is not neccessary:

    .. code-block:: python

        # define feature as dihedral angle 
        f1 = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))
        f_layer = FeatureLayer([f1], input_ag, use_angle_value=False)

        # since feature is both translation- and rotation-invariant, alignment is not neccessary
        pp_layer = PreprocessingANN(None, f_layer)

    If only alignment is desired, one can provide an identity feature layer when defining :class:`PreprocessingANN`.

    .. code-block:: python

        f = Feature('identity', 'position', input_ag)
        identity_f_layer = FeatureLayer([f], input_ag, use_angle_value=False)
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
        input_ag=ref.select_atoms('bynum 1 2 3 4 5 6 7')

        f_layer = FeatureLayer([f1], input_ag, use_angle_value=False)
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

