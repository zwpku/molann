Quick Start
===========

Example 1: Define an ANN that includes alignment layer.

.. code-block:: python

    import MDAnalysis as mda
    from molann.ann import AlignmentLayer, FeatureLayer, PreprocessingANN, MolANN, create_sequential_nn
    from molann.feature import Feature

    # pdb file of the system
    pdb_filename = '/path/to/system.pdb'
    ref = mda.Universe(pdb_filename) 

    # define alignment layer
    align = AlignmentLayer(ref.select_atoms('bynum 1 2 3'))

    f = Feature('identity', 'position', ref.atoms)
    identity_f_layer = FeatureLayer([f], use_angle_value=False)
    pp_layer = PreprocessingANN(align, identity_f_layer)

    output_dim = pp_layer.output_dimension()
    # neural networks layers which contains training parameters 
    nn = create_sequential_nn([output_dim, 5, 3])

    model = MolANN(pp_layer, nn)

Example 2: Define an ANN as a function of a bond distance and a dihedral angle.

.. code-block:: python

    import MDAnalysis as mda
    from molann.ann import FeatureLayer, PreprocessingANN, MolANN, create_sequential_nn
    from molann.feature import Feature

    # pdb file of the system
    pdb_filename = '/path/to/system.pdb'
    ref = mda.Universe(pdb_filename) 

    f1 = Feature('name', 'bond', ref.select_atoms('bynum 5 6'))
    f2 = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))

    f_layer = FeatureLayer([f1,f2], use_angle_value=False)
    # alignment not needed, since both features are translation- and rotation-invariant.
    pp_layer = PreprocessingANN(None, f_layer)

    output_dim = pp_layer.output_dimension()
    # neural networks layers which contains training parameters 
    nn = create_sequential_nn([output_dim, 5, 3])

    model = MolANN(pp_layer, nn)


