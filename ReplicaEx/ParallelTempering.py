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
def MyAlanineDipeptideVacuum(filepath):
    #psf = CharmmPsfFile('%s/vacuum.psf' % filepath)
    #params = CharmmParameterSet('%s/par_all22_prot.inp' % filepath, '%s/top_all22_prot.inp' % filepath)

    pdb = PDBFile('%s/vacuum.pdb' % filepath)
    forcefield = ForceField('amber14-all.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedCutoff=2*unit.nanometer, constraints=HBonds)
    test_system = testsystems.TestSystem()
    test_system.system = system
    test_system.positions = pdb.positions
    test_system.topology = pdb.topology
    return test_system

test_system = MyAlanineDipeptideVacuum('./AlanineDipeptideOpenMM')

# test_system = testsystems.AlanineDipeptideVacuum()

topology = test_system.mdtraj_topology

n_steps = 10000
n_replicas = 1  # Number of temperature replicas.
T_min = 398.0 * unit.kelvin  # Minimum temperature.
T_max = 500.0 * unit.kelvin  # Maximum temperature.

reference_state = states.ThermodynamicState(system=test_system.system, temperature=T_min)

move = mcmc.GHMCMove(timestep=2.0*unit.femtoseconds, n_steps=10)

simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=n_steps, online_analysis_interval=None)

storage_path = tempfile.NamedTemporaryFile(delete=False).name + '.nc'

print ('filename of checkpoints: %s' % storage_path)

reporter = MultiStateReporter(storage_path, checkpoint_interval=100)

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

energy_list = reporter.read_energies()[0][:,0,0]
print ('List of energies:', energy_list)

# positions = unit.Quantity(x, unit.nanometers)

# construct a Trajectory object 
traj = mdtraj.Trajectory(x, topology)
# save the trajectory in dcd format 
traj_filename = "./traj.dcd"
traj.save_dcd(traj_filename)
end = time.time()

print ( 'trajectory (of length %d) saved to file: %s.\n%d sec. elapsed.' % (traj.n_frames, traj_filename, end - start) )

