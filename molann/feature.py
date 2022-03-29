"""Feature of Molecular System --- :mod:`molann.feature`
==========================================================

:Author: Wei Zhang
:Year: 2022
:Copyright: GNU Public License v3

This module implements a class that defines a feature of molecular system
(:class:`molann.feature.Feature`), and a class that constructs a list of
features from a feature file (:class:`molann.feature.FeatureFileReader`).

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
    r"""Feature of system

    :param str name: feature's name
    :param str feature_type: feature's type; supported values are 'angle', 'bond', 'dihedral', and 'position'
    :param atom_group: atom group used to define a feature 
    :type atom_group: :external+mdanalysis:class:`MDAnalysis.core.groups.AtomGroup`

    .. note::

        When `feature_type` ='angle', then :`atom_group` must contain 3 atoms; 
        when `feature_type` ='bond', then `atom_group` must contain 2 atoms; 
        when `feature_type` ='dihedral', then `atom_group` must contain 4 atoms. 

    Example
    -------

    .. code-block:: python

        # package MDAnalysis is required
        import MDAnalysis as mda
        from feature import Feature 

        # pdb file of the system
        pdb_filename = '/path/to/system.pdb'
        ref = mda.Universe(pdb_filename) 

        # feature that is the angle formed by the atoms whose ids are 1, 2 and 3.
        f1 = Feature('some name', 'angle', ref.select_atoms('bynum 1 2 3'))
        print (f1.get_feature_info())
        # feature that is the bond distance between the two atoms whose ids are 1 and 2.
        f2 = Feature('some name', 'bond', ref.select_atoms('bynum 1 2'))
        # dihedral angle formed by atoms whose ids are 1, 2, 3, and 4.
        f3 = Feature('some name', 'dihedral', ref.select_atoms('bynum 1 2 3 4'))
        # coordinates of atoms whose ids are 3 and 4.
        f4 = Feature('some name', 'position', ref.select_atoms('bynum 3 5'))

    .. note::
        :external+mdanalysis:meth:`MDAnalysis.core.universe.Universe.select_atoms()` returns an atom group that does not preserve the orders of atoms.
        To construct a feature that is the dihedral angle formed by atoms whose ids
        are 2,1,3, and 4, we can define the atom group by concatenation:
        
        .. code-block:: python

            ag = ref.select_atoms('bynum 2') + ref.select_atoms('bynum 1') + ref.select_atoms('bynum 3 and 4')  
            f = Feature('some name', 'dihedral', ag)

    Attributes
    ----------
    name : str
        feature name
    type_name : str
        feature type: 'angle', 'bond', 'dihedral', or 'position'
    type_id : int
        0 for 'angle'; 1 for 'bond'; 2 for 'dihedral'; 3 for 'position'

    """

    def __init__(self, name, feature_type, atom_group):

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

        self.name = name
        self.type_name = feature_type
        self.atom_group = atom_group
        self.type_id = type_id

    def get_name(self):
        """
        :returns: :attr:`name`
        :rtype: str
        """
        return self.name

    def get_type(self):
        """
        :return: :attr:`type_name`
        :rtype:  str
        """
        return self.type_name

    def get_atom_indices(self):
        """
        :rtype:  list of int
        :return: indices of atoms in the atom group. The indices start from 1.
        """
        return self.atom_group.ids

    def get_type_id(self):
        """
        :return: :attr:`type_id`
        :rtype: int
        """
        return self.type_id

    def get_feature_info(self):
        """
        :return: feature's information
        :rtype: :class:`pandas.DataFrame`  
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

