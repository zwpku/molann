MolANN
======

Artificial Neural Networks (ANNs) for Molecular Systems

This package implements PyTorch ANN classes that include alignment layers and feature layers. 

Installation 
============

The package can be installed via `pip`:

.. code-block:: console

   pip install molann

The installation from source is described in the `Installation`_

Simple example 
==============

.. code-block:: python

    import MDAnalysis as mda
    from molann.ann import AlignmentLayer, FeatureLayer, PreprocessingANN, MolANN, create_sequential_nn
    from molann.feature import Feature
    import torch

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
    torch.jit.script(model).save('model.pt')

More examples for each classes can be found in the `MolANN docs`_

Documentataion
==============

Please refer to `MolANN docs`_.


.. _`Installation`:
  https://molann.readthedocs.io/en/latest/installation.html
.. _`MolANN docs`:
  https://molann.readthedocs.io/en/latest
