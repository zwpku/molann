Quick Start
===========

Example 1: Define an ANN that includes alignment layer. 
This gives a neural network function that is invariant under both rotation and translation.

.. code-block:: python

    import MDAnalysis as mda
    from molann.ann import AlignmentLayer, FeatureLayer, PreprocessingANN, MolANN, create_sequential_nn
    from molann.feature import Feature

    # pdb file of the system
    pdb_filename = '/path/to/system.pdb'
    ref = mda.Universe(pdb_filename) 

    # define the alignment layer
    # use coordinates of the first three atoms to align the system 
    align = AlignmentLayer(ref.select_atoms('bynum 1 2 3'))

    # define a feature that is the identity map
    f = Feature('identity', 'position', ref.atoms)
    # use the identity feature to define the feature layer
    identity_f_layer = FeatureLayer([f], use_angle_value=False)
    # the preprocessing layer consists of the alignment and the feature layers
    pp_layer = PreprocessingANN(align, identity_f_layer)

    output_dim = pp_layer.output_dimension()
    # define neural network layers that contain training parameters 
    nn = create_sequential_nn([output_dim, 5, 3])

    # define the final network 
    model = MolANN(pp_layer, nn)

Example 2: Define an ANN as a function of a bond distance and a dihedral angle.

.. code-block:: python

    import MDAnalysis as mda
    from molann.ann import FeatureLayer, PreprocessingANN, MolANN, create_sequential_nn
    from molann.feature import Feature

    # pdb file of the system
    pdb_filename = '/path/to/system.pdb'
    ref = mda.Universe(pdb_filename) 

    # define a feature that describes the bond between atoms 5 and 6.
    f1 = Feature('name', 'bond', ref.select_atoms('bynum 5 6'))
    # define a feature that describes the dihedral angle formed by the first 4 atoms.
    f2 = Feature('name', 'dihedral', ref.select_atoms('bynum 1 3 2 4'))

    # define the feature layer using the above two features.
    f_layer = FeatureLayer([f1,f2], use_angle_value=False)
    # define the preprocessing layer. 
    # we do not need alignment, since both features f1 and f2 are translation- and rotation-invariant.
    pp_layer = PreprocessingANN(None, f_layer)

    output_dim = pp_layer.output_dimension()
    # define neural network layers which contains training parameters 
    nn = create_sequential_nn([output_dim, 5, 3])

    # define the final network
    model = MolANN(pp_layer, nn)


