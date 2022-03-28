import MDAnalysis as mda
import sys
import torch

sys.path.append('../molann')

from feature import Feature, FeatureFileReader
from ann import create_sequential_nn, AlignmentLayer, FeatureMap, FeatureLayer

def test_Feature():
    f = Feature('name', 'bond', ref.select_atoms('bynum 1 2'))
    print (f.get_feature_info())

def test_FeatureFileReader():
    # read features for histogram plot
    feature_reader = FeatureFileReader('feature.txt', 'Histogram', ref)
    feature_list = feature_reader.read()

    print (feature_reader.get_feature_info())

def test_create_sequential_nn():
    nn1 = create_sequential_nn([10, 5, 1])
    nn2 = create_sequential_nn([10, 5, 1], activation=torch.nn.ReLU())
    nn3 = create_sequential_nn([10])
    print (nn1, nn2, nn3)

def test_FeatureMap():
#    f = Feature('name', 'position', ref.select_atoms('bynum 1 3 2'))
    f = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))
    fmap = FeatureMap(f, use_angle_value=False)
    print ('dim=', fmap.dim())
    pos = torch.tensor(ref.atoms.positions).unsqueeze(0)
    print (pos)
    print (fmap(pos))
    feature_model_name = 'feature_map.pt'
    torch.jit.script(fmap).save(feature_model_name)

def test_AlignmentLayer():
    ag=ref.select_atoms('bynum 1 2 3')
    align = AlignmentLayer(ag)
    align.show_info()
    align_model_name = 'algin.pt'
    torch.jit.script(align).save(align_model_name)

def test_FeatureLayer():
#    f = Feature('name', 'position', ref.select_atoms('bynum 1 3 2'))
    f1 = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))
    f2 = Feature('name', 'angle', ref.select_atoms('bynum 1 3 2'))
    f3 = Feature('name', 'bond', ref.select_atoms('bynum 1 3'))
    f_layer = FeatureLayer([f1, f3, f2], use_angle_value=False)
    print ('output dim=', f_layer.output_dimension())
    pos = torch.tensor(ref.atoms.positions).unsqueeze(0)
    print (pos)
    print (f_layer(pos))
    feature_layer_model_name = 'feature_layer.pt'
    torch.jit.script(f_layer).save(feature_layer_model_name)

pdb_filename = '../../openmm_traj_samplers/system/AlanineDipeptideOpenMM/vacuum.pdb'
# load the reference configuration from the PDB file
ref = mda.Universe(pdb_filename) 

#test_Feature()
#test_FeatureFileReader()
#test_create_sequential_nn()
#test_AlignmentLayer()
#test_FeatureMap()
test_FeatureLayer()

