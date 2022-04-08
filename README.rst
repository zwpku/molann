MolANN
======

Artificial Neural Networks (ANNs) for Molecular Systems

This package implements PyTorch ANN classes that allow to define neural network functions that take into account alignment, or functions that depend on molecular features (bonds, angles, etc.). 

Installation 
============

The package can be installed via `pip`:

.. code-block:: console

   pip install molann

The installation from source is described in the `Installation`_ page.

Simple example 
==============

The following code defines an ANN as a function of a bond distance and a dihedral angle.

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

    # define feature layer using the above two features.
    f_layer = FeatureLayer([f1,f2], use_angle_value=False)

    # define the preprocessing layer. 
    # we do not need alignment, since both features f1 and f2 are translation- and rotation-invariant.
    pp_layer = PreprocessingANN(None, f_layer)

    output_dim = pp_layer.output_dimension()
    # define neural network layers that contain training parameters 
    nn = create_sequential_nn([output_dim, 5, 3])

    # define the final network
    model = MolANN(pp_layer, nn)

    torch.jit.script(model).save('model.pt')

More examples for each class can be found in the `MolANN docs`_ .

Documentataion
==============

Please refer to `MolANN docs`_.


.. _`Installation`:
  https://molann.readthedocs.io/en/latest/installation.html
.. _`MolANN docs`:
  https://molann.readthedocs.io/en/latest
