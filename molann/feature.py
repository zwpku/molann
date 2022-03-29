r"""Features of Molecuar System --- :mod:`molann.feature`
==========================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

Classes
-------
.. autoclass:: Feature
    :members:

.. autoclass:: FeatureFileReader
    :members:

"""

import torch
import pandas as pd

class Feature(object):
    r"""Feature information 

    Parameters
    ----------
    name : str
        name of the feature
    feature_type : str
        type of feature. Currently supported value ares: 'angle', 'bond', 'dihedral', and 'position'
    atom_group : AtomGroup
        AtomGroup. Atoms used in defining a feature 

    Attributes
    ----------
    name : str
    type : int


    Note
    ----
    :math:`D`

    .. math::

       \frac{30}{\mathbb{E}}

    Returns
    -------
    a : int

    Example
    -------

    >>> f = Feature(fname, ftype, ag)
    >>> exit(0)

    .. code-block:: python
    
        import os

    """

    def __init__(self, feature_name, feature_type, atom_group):

        if feature_type not in ['angle', 'bond', 'dihedral', 'position']:
            raise NotImplementedError(f'feature {feature_type} not implemented!')

        if len(set(atom_group)) < len(atom_group) :
            raise IndexError(f'atom group contains repeated elements!')

        if feature_type == 'angle': 
            assert len(atom_group)==3, '3 atoms are needed to define an angle feature, {} provided'.format(len(atom_group))
            type_id = 0 
        if feature_type == 'bond': 
            assert len(atom_group)==2, '2 atoms are needed to define a bond length feature, {} provided'.format(len(atom_group))
            type_id = 1 
        if feature_type == 'dihedral': 
            assert len(atom_group)==4, '4 atoms are needed to define a dihedral angle feature, {} provided'.format(len(atom_group))
            type_id = 2 
        if feature_type == 'position':
            type_id = 3 

        self.name = feature_name
        self.type_name = feature_type
        self.atom_group = atom_group
        self.type_id = type_id

    def get_name(self):
        """
        return feature's name
        """
        return self.name

    def get_type(self):
        """
        return feature's type
        """
        return self.type_name

    def get_atom_indices(self):
        """
        return indices of atoms in the atom group
        """
        return self.atom_group.ids

    def get_type_id(self):
        """
        return feature id
        """
        return self.type_id

    def get_feature_info(self):
        """
        TBA
        """
        return pd.DataFrame({'name': self.name, 'type': self.type_name, 'type_id': self.type_id, 'atom indices': [self.get_atom_indices()]})

class FeatureFileReader(object):
    r"""Read features from file

    Parameters
    ----------
    feature_file : str
        name of the feature file
    """

    def __init__(self, feature_file, section_name, universe):
        """ init
        """

        self.feature_file = feature_file
        self.section_name = section_name
        self.u = universe

    def read(self):
        """
        read features from file
        """

        self.feature_list = []

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
                ag = None
                feature_name, feature_type, *selector_list = line.split(',')
                for selector in selector_list:
                    if ag is None :
                        ag = self.u.select_atoms(selector)
                    else :
                        ag = ag + self.u.select_atoms(selector)

                feature = Feature(feature_name.strip(), feature_type.strip(), ag)
                self.feature_list.append(feature)

        pp_cfg_file.close()

        if len(self.feature_list) == 0 : 
           print ("Warning: no feature found! \n") 
        else :
           print ("\n{} features loaded\n".format(len(self.feature_list)) )

        return self.feature_list

    def get_feature_list(self):
        """return list of features 
        """
        return self.feature_list

    def get_num_of_features(self):
        """return number of features
        """
        return len(self.feature_list)

    def get_feature_info(self):
        """return a pandas DataFrame including information of all features
        """
        df = pd.DataFrame()
        for f in self.feature_list:
            f_info = f.get_feature_info()
            df = df.append(f_info, ignore_index=True)
        return df

