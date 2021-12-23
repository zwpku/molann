#!/usr/bin/env python
# ## Generate MD trajectory data using OpenMM ad openmmtools packages

# +
# import from openmm and openmmtools
from openmm import *
from openmm.app import *
from openmmtools import testsystems, integrators
from openmmtools import mcmc
from openmmtools import states
from openmmtools.multistate import ParallelTemperingSampler
from openmmtools.multistate import MultiStateReporter

import math
from random import random, randint
import tempfile
import numpy as np
import time
import datetime
import mdtraj
import os
import warnings


# -

# construct a TestSystem object from a PDB file
def MyAlanineDipeptideVacuum(pdb_filename):
    pdb = PDBFile(pdb_filename)
    forcefield = ForceField('amber14-all.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedCutoff=2*unit.nanometer, constraints=HBonds)
    test_system = testsystems.TestSystem()
    test_system.system = system
    test_system.positions = pdb.positions
    test_system.topology = pdb.topology
    return test_system

# ### Set parameters and define the MD system

# +
n_steps = 500 # Total simulation steps, i.e. number of states.
n_ghmc_step = 10
n_ckpt_interval = 10
n_replicas = 1  # Number of temperature replicas.
T_min = 298.0 * unit.kelvin  # Minimum temperature.
T_max = 500.0 * unit.kelvin  # Maximum temperature.
is_read_from_storage = True
 
suffix_string = "1"
storage_path = 'tmpfile_%s.nc' % suffix_string
if is_read_from_storage == False:
    if os.path.exists(storage_path):
        os.remove(storage_path)
        warnings.warn("storage file already exists: %s, and is deleted." % storage_path)
  
# This PDB file describes AlanineDipeptide in vacuum. 
# It is from the OpenMM source code.
pdb_filename = './AlanineDipeptideOpenMM/vacuum.pdb'
test_system = MyAlanineDipeptideVacuum(pdb_filename)
# Alanatively, one can also use the testsystem provided by openmmtools package.
#test_system = testsystems.AlanineDipeptideVacuum()

topology = test_system.mdtraj_topology # Topology in the format of mdtraj package.
# -

# ### Generate trajectory using parallel tempering sampler

# +
if is_read_from_storage :
    simulation = ParallelTemperingSampler.from_storage(storage_path)
    print ('restart from checkpoint: %s' % storage_path)
    print ('current iteration=%d' % simulation.iteration)
else :
    # The following steps are standard.
    print ('filename of checkpoints: %s' % storage_path)      
    reference_state = states.ThermodynamicState(system=test_system.system, temperature=T_min)
    move = mcmc.GHMCMove(timestep=2.0*unit.femtoseconds, n_steps=n_ghmc_step)
    simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=n_steps, online_analysis_interval=None)
    reporter = MultiStateReporter(storage_path, checkpoint_interval=n_ckpt_interval)
    simulation.create(reference_state, states.SamplerState(test_system.positions), reporter, min_temperature=T_min, max_temperature=T_max, n_temperatures=n_replicas)

print ('Simulation starts...')
start = time.time()
simulation.extend(n_steps)
print ('final iteration=%d' % simulation.iteration)
end = time.time()
print ( 'Simulation ends, %d sec. elapsed.' % (end - start) )
# -

# ### Save trajectory data in DCD format

# +
start = time.time()
# Read the checkpoint file and get the trajectory.
# See the source file of the class Multistatereporter for details.
reporter = MultiStateReporter(storage_path, 'r', checkpoint_interval=n_ckpt_interval)
storage = reporter._storage_dict['checkpoint']
x = storage.variables['positions'][:,0,:,:].astype(np.float64)

#energy_list = reporter.read_energies()[0][:,0,0]
#print ('List of energies:', energy_list)

# construct a Trajectory object 
traj = mdtraj.Trajectory(x, topology)
# save the trajectory in dcd format 
traj_filename = "./traj.dcd"
traj.save_dcd(traj_filename)
end = time.time()

print ( 'trajectory (of length %d) saved to file: %s.\n%d sec. elapsed.' % (traj.n_frames, traj_filename, end - start) )
# -


