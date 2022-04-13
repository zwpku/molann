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

    :raises AssertionError: if the number of atoms in **atom_group** does not match **feature_type**.


    Note:

        For **feature_type** = 'angle', 'bond', and 'dihedral', **atom_group** must contain 3 atoms, 2 atoms, and 4 atoms, respectively. 

    Examples:

    .. code-block:: python

        # package MDAnalysis is required
        import MDAnalysis as mda
        from molann.feature import Feature 

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

    Note:
        :external+mdanalysis:meth:`MDAnalysis.core.universe.Universe.select_atoms()` returns an atom group that does not preserve the orders of atoms.
        To construct a feature that is the dihedral angle formed by atoms whose ids
        are 2,1,3, and 4, we can define the atom group by concatenation:
        
        .. code-block:: python

            ag = ref.select_atoms('bynum 2') + ref.select_atoms('bynum 1') + ref.select_atoms('bynum 3 and 4')  
            f = Feature('some name', 'dihedral', ag)

    Attributes:
        name: string that indicates the name of feature
        type_name : string whose value is among 'angle', 'bond', 'dihedral', or 'position'

        type_id : integer, 0 if type_name='angle', 1 if type_name='bond', 2 if type_name='dihedral', 3 if type_name='position'

        atom_group : :external+mdanalysis:class:`MDAnalysis.core.groups.AtomGroup` used to define the feature

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
        Returns:
            :attr:`name`
        """
        return self.name

    def get_type(self):
        """
        Returns:
            :attr:`type_name`
        """
        return self.type_name

    def get_atom_indices(self):
        """
        Returns:
            list of int, indices of atoms in the atom group. The indices start from 1.
        """
        return self.atom_group.ids

    def get_type_id(self):
        """
        Returns:
            :attr:`type_id`
        """
        return self.type_id

    def get_feature_info(self):
        """
        Returns:
            :external+pandas:class:`pandas.DataFrame`, which contains feature's information
        """
        return pd.DataFrame({'name': self.name, 'type': self.type_name, 'type_id': self.type_id, 'atom indices': [self.get_atom_indices()]})

class FeatureFileReader(object):
    r"""Read features from file

    :param str feature_file: name of the feature file
    :param str section_name: name of the section in the file from which to read features
    :param universe: universe that defines the system
    :type universe: :external+mdanalysis:class:`MDAnalysis.core.universe.Universe`

    Note:

        A feature file is a normal text file. 

        Each section describes a list of features.
        The begining of a section is marked by a line with content '[section_name]', and its end is marked by a line with
        content '[End]'. :meth:`read` constructs a list of :class:`Feature` from the section with
        the corresponding section_name.

        Lines that describe a feature contain several fields, seperated by
        comma. The first and the second fields specify the `name` and the
        `type_name` of the feature, respectively. The remaining fields specify
        the strings for selection  to define an atom group (by concatenation). See :external+mdanalysis:mod:`MDAnalysis.core.selection`.

        Lines starting with '#' are comment lines and they are not processed. 

    Examples:

       Below is an example of feature file, named as *feature.txt*.

    .. code-block:: text

        # This is a comment line. 

        # Lines that describe a feature contain several fields, seperated by comma. 

        # The first and the second fields specify the name and the
        # type_name of the feature, respectively. The remaining fields specify
        # the strings for selection to define an atom group by concatenation.

        # Note: to keep the order of atoms, use 'bynum 5, bynum 2', instead of 'bynum 5 2'

        [Preprocessing]
        #position, type C or type O or type N
        p1, position, resid 2 
        [End]
        [Histogram]
        d1, dihedral, bynum 5, bynum 7, bynum 9, bynum 15 
        d2, dihedral, bynum 7, bynum 9, bynum 15, bynum 17
        b1, bond, bynum 2 5
        b2, bond, bynum 5 6
        a1, angle, bynum 20, bynum 19, bynum 21
        a2, angle, bynum 16, bynum 15, bynum 17
        [End]
        [Output]
        d1, dihedral, bynum 5 7 9 15 
        d2, dihedral, bynum 7 9 15 17
        [End]

    The following code constructs a list of features from the section 'Histogram', and a list of features from section 'Preprocessing'.

    .. code-block:: python

        # package MDAnalysis is required
        import MDAnalysis as mda
        from molann.feature import FeatureFileReader

        # pdb file of the system
        pdb_filename = '/path/to/system.pdb'
        ref = mda.Universe(pdb_filename) 

        # read features from the section 'Histogram' 
        feature_reader = FeatureFileReader('feature.txt', 'Histogram', ref)
        feature_list_1 = feature_reader.read()

        # read features from the section 'Preprocessing'
        feature_reader = FeatureFileReader('feature.txt', 'Preprocessing', ref)
        feature_list_2 = feature_reader.read()

     """

    def __init__(self, feature_file, section_name, universe):

        self.feature_file = feature_file
        self.section_name = section_name
        self.u = universe

    def read(self):
        """
        read features from file

        Returns:

            list of :class:`Feature`, a list of features constructed from the feature file
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

        return self.feature_list

    def get_feature_list(self):
        """
        Returns:
            list of :class:`Feature`, feature list constructed by calling :meth:`read` 
        """
        return self.feature_list

    def get_num_of_features(self):
        """
        Returns:
            int, number of features in the feature list 
        """
        return len(self.feature_list)

    def get_feature_info(self):
        """
        Returns:
            :external+pandas:class:`pandas.DataFrame`, which contains information of all features (each row describes a feature)
        """
        df = pd.DataFrame()
        for f in self.feature_list:
            f_info = f.get_feature_info()
            df = df.append(f_info, ignore_index=True)
        return df

