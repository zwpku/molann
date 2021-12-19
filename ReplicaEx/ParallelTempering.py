#!/usr/bin/env python

import openmm
from openmm import unit
from openmmtools import testsystems, integrators
from openmmtools import mcmc
from openmmtools import states
from openmmtools import cache
from openmmtools.multistate import ParallelTemperingSampler
from openmmtools.multistate import MultiStateReporter

import math
from random import random, randint
import tempfile
import numpy as np

test_system = testsystems.AlanineDipeptideVacuum()
topology = test_system.mdtraj_topology

n_steps = 3
n_replicas = 3  # Number of temperature replicas.
T_min = 298.0 * unit.kelvin  # Minimum temperature.
T_max = 500.0 * unit.kelvin  # Maximum temperature.

reference_state = states.ThermodynamicState(system=test_system.system, temperature=T_min)

move = mcmc.GHMCMove(timestep=2.0*unit.femtoseconds, n_steps=50)

simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=n_steps)

storage_path = tempfile.NamedTemporaryFile(delete=False).name + '.nc'

print ('filename of checkpoints: %s' % storage_path)

reporter = MultiStateReporter(storage_path, checkpoint_interval=1)

simulation.create(reference_state, states.SamplerState(test_system.positions), reporter, min_temperature=T_min, max_temperature=T_max, n_temperatures=n_replicas)

simulation.run(n_steps)

# Read the checkpoint file and get the trajectory.
# See the source file of the class Multistatereporter for details.
storage = reporter._storage_dict['checkpoint']
x = storage.variables['positions'][:,0,:,:].astype(np.float64)
positions = unit.Quantity(x, unit.nanometers)

print (positions.shape)

