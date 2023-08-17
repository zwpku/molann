import MDAnalysis as mda
import sys
import torch

sys.path.append('../molann')

from feature import Feature, FeatureFileReader
from ann import create_sequential_nn, AlignmentLayer, FeatureMap, FeatureLayer, PreprocessingANN, MolANN

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
    input_ag =ref.select_atoms('bynum 1 2 3 4 5')
    pos = torch.tensor(input_ag.positions).unsqueeze(0)
    fmap = FeatureMap(f, input_ag, use_angle_value=False)
    print ('dim=', fmap.dim())
    print (fmap(pos))
    feature_model_name = 'feature_map.pt'
    torch.jit.script(fmap).save(feature_model_name)

def test_AlignmentLayer():
    ag=ref.select_atoms('bynum 1 2 5')
    #input_ag=ref.select_atoms('bynum 1 2 3 5')
    input_ag=ref.atoms
    align = AlignmentLayer(ag, input_ag)
    align.show_info()
    print (align(pos))
    align_model_name = 'algin.pt'
    torch.jit.script(align).save(align_model_name)

def test_FeatureLayer():
#    f = Feature('name', 'position', ref.select_atoms('bynum 1 3 2'))
    f1 = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))
    f2 = Feature('name', 'angle', ref.select_atoms('bynum 1 3 2'))
    f3 = Feature('name', 'bond', ref.select_atoms('bynum 1 3'))

    input_ag =ref.select_atoms('bynum 1 2 3 4 5')
    pos = torch.tensor(input_ag.positions).unsqueeze(0)

    f_layer = FeatureLayer([f1, f3, f2], input_ag, use_angle_value=False)

    print ('output dim=', f_layer.output_dimension())
    print (f_layer(pos))
    feature_layer_model_name = 'feature_layer.pt'
    torch.jit.script(f_layer).save(feature_layer_model_name)

    # To define an identity feature layer
    #ag = ref.atoms
    ag =ref.select_atoms('bynum 1 2 3 4 5')
    f4 = Feature('identity', 'position', ag)
    identity_f_layer = FeatureLayer([f4], ag, use_angle_value=False)
    pos = torch.tensor(ag.positions).unsqueeze(0)
    print ('identity map:\n', identity_f_layer(pos))
    print ('output dim=', identity_f_layer.output_dimension())
    ff = identity_f_layer.get_feature(0)
    print (ff.get_feature_info())
    feature_layer_model_name = 'identity_feature_layer.pt'
    torch.jit.script(identity_f_layer).save(feature_layer_model_name)

def test_PreprocessingANN():

    input_ag =ref.select_atoms('bynum 1 2 3 4 5')
    pos = torch.tensor(input_ag.positions).unsqueeze(0)

    ag=ref.select_atoms('bynum 1 2 3')
    align = AlignmentLayer(ag, input_ag)
    f1 = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))
    f_layer = FeatureLayer([f1], input_ag, use_angle_value=False)
    pp_layer = PreprocessingANN(align, f_layer)
    print ('output_dim=', pp_layer.output_dimension())
    print (pp_layer(pos))
    pp_layer = PreprocessingANN(None, f_layer)
    print (pp_layer(pos))

    f1 = Feature('name', 'position', ref.select_atoms('bynum 1 2'))
    f_layer = FeatureLayer([f1], input_ag, use_angle_value=False)
    pp_layer = PreprocessingANN(align, f_layer)
    print (pp_layer(pos))

    pp_layer = PreprocessingANN(None, f_layer)
    print (pp_layer(pos))

    model_name = 'pp_layer.pt'
    torch.jit.script(pp_layer).save(model_name)

def test_MolANN():
    input_ag =ref.select_atoms('bynum 1 2 3 4 5')
    pos = torch.tensor(input_ag.positions).unsqueeze(0)
    f1 = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))
    f_layer = FeatureLayer([f1], input_ag, use_angle_value=False)
    pp_layer = PreprocessingANN(None, f_layer)
    output_dim = pp_layer.output_dimension()
    nn = create_sequential_nn([output_dim, 5, 3])
    molann = MolANN(pp_layer, nn)
    print (molann(pos))
    model_name = 'ann_layer.pt'
    torch.jit.script(molann).save(model_name)


pdb_filename = 'alanine-dipeptide-vacuum.pdb'
# load the reference configuration from the PDB file
ref = mda.Universe(pdb_filename) 

pos = torch.tensor(ref.atoms.positions).unsqueeze(0)
#print (pos)

#test_Feature()
#test_FeatureFileReader()
#test_create_sequential_nn()
#test_AlignmentLayer()
#test_FeatureMap()
#test_FeatureLayer()
#test_PreprocessingANN()
test_MolANN()

