#!/usr/bin/env python

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
import mdtraj

# Construct a TestSystem object 
def MyAlanineDipeptideVacuum():
    psf = CharmmPsfFile('vacuum.psf')
    pdb = PDBFile('vacuum.pdb')
    params = CharmmParameterSet('par_all22_prot.inp', 'top_all22_prot.inp')
    system = psf.createSystem(params, nonbondedMethod=NoCutoff, nonbondedCutoff=1*unit.nanometer, constraints=HBonds)

    test_system = testsystems.TestSystem()
    test_system.system = system
    test_system.positions = pdb.positions
    test_system.topology = pdb.topology
    return test_system

test_system = MyAlanineDipeptideVacuum()

#test_system = testsystems.AlanineDipeptideVacuum()

topology = test_system.mdtraj_topology

n_steps = 4
n_replicas = 3  # Number of temperature replicas.
T_min = 298.0 * unit.kelvin  # Minimum temperature.
T_max = 500.0 * unit.kelvin  # Maximum temperature.

reference_state = states.ThermodynamicState(system=test_system.system, temperature=T_min)

move = mcmc.GHMCMove(timestep=2.0*unit.femtoseconds, n_steps=2)

simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=n_steps)

storage_path = tempfile.NamedTemporaryFile(delete=False).name + '.nc'

print ('filename of checkpoints: %s' % storage_path)

reporter = MultiStateReporter(storage_path, checkpoint_interval=2)

simulation.create(reference_state, states.SamplerState(test_system.positions), reporter, min_temperature=T_min, max_temperature=T_max, n_temperatures=n_replicas)

print ('Simulation starts...')

start = time.time()
simulation.run(n_steps)
end = time.time()

print ( 'Simulation ends, %d sec. elapsed.' % (end - start) )

start = time.time()
# Read the checkpoint file and get the trajectory.
# See the source file of the class Multistatereporter for details.
storage = reporter._storage_dict['checkpoint']
x = storage.variables['positions'][:,0,:,:].astype(np.float64)
# positions = unit.Quantity(x, unit.nanometers)

# construct a Trajectory object 
traj = mdtraj.Trajectory(x, topology)
# save the trajectory in dcd format 
traj_filename = "./traj.dcd"
traj.save_dcd(traj_filename)
end = time.time()

print ( 'trajectory save to file: %s.\n%d sec. elapsed.' % (traj_filename, end - start) )

