import torch

class Feature(object):
    """Feature information 

    Parameters
    ----------
    name : str
        name of the feature 
    feature_type : str
        type of feature; supported value ares: 'angle', 'bond', 'dihedral', and 'position'
    ag : list of int
        atom group, list of atom indices

    """

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

    def name(self):
        """
        return feature's name
        """
        return self.name

    def type(self):
        """
        return feature's type
        """
        return self.type_name

    def atom_group(self):
        """
        return atom group
        """
        return self.atom_group

    def type_id(self):
        """
        return atom group
        """
        return self.type_id


class FeatureFileReader(object):
    r"""Read features from file

    Parameters
    ----------
    feature_file : str
        name of the feature file
    """

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


