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
        r"""display information of features 

        Parameters
        ----------
        info_title : str
            texts to print before displaying information of features
        """

        print (f'{info_title}Id.\tName\tType\tAtomIDs')
        for idx in range(self.num_features):
            print ('{}\t{}\t{}\t{}'.format(idx, self.name_list[idx], self.type_list[idx], self.ag_list[idx].numpy()))

    def feature_name(self, idx):
        r"""return the name of feature 

        Parameters
        ----------
        idx : int
            index of feature
        """

        return self.name_list[idx]

    def feature_all_names(self):
        r"""return the list of feature names 
        """
        return self.name_list

    def feature_total_dimension(self):
        r"""return total dimension of features
        """

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
        r"""map position to certain feature 

        Parameters
        ----------
        x : torch tensor 
            coordinates of state
        idx : int
            index of feature

        """

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
        """forward map
        """
        xf = self.map_to_feature(x, 0)
        for i in range(1, len(self.type_id_list)) :
            # Features are stored in columns 
            xf = torch.cat((xf, self.map_to_feature(x, i)), dim=1)
        return xf

class IdentityFeatureMap(torch.nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        return x
