#!/usr/bin/env python

import openmm
from openmm import unit
from openmmtools import testsystems, integrators
from openmmtools import mcmc
from openmmtools import states
from openmmtools import cache
from openmmtools.multistate import ParallelTemperingSampler

import math
from random import random, randint

class ReplicaExchange:

    def __init__(self, thermodynamic_states, sampler_states, mcmc_move):
        self._thermodynamic_states = thermodynamic_states
        self._replicas_sampler_states = sampler_states
        self._mcmc_move = mcmc_move

    def run(self, n_iterations=1):
        for iteration in range(n_iterations):
            self._mix_replicas()
            self._propagate_replicas()

    def _propagate_replicas(self):
        # _thermodynamic_state[i] is associated to the replica configuration in _replicas_sampler_states[i].
        for thermo_state, sampler_state in zip(self._thermodynamic_states, self._replicas_sampler_states):
            self._mcmc_move.apply(thermo_state, sampler_state)

    def _mix_replicas(self, n_attempts=1):
        # Attempt to switch two replicas at random. Obviously, this scheme can be improved.
        for attempt in range(n_attempts):
            # Select two replicas at random.
            i = randint(0, len(self._thermodynamic_states)-1)
            j = randint(0, len(self._thermodynamic_states)-1)
            sampler_state_i, sampler_state_j = (self._replicas_sampler_states[k] for k in [i, j])
            thermo_state_i, thermo_state_j = (self._thermodynamic_states[k] for k in [i, j])

            # Compute the energies.
            energy_ii = self._compute_reduced_potential(sampler_state_i, thermo_state_i)
            energy_jj = self._compute_reduced_potential(sampler_state_j, thermo_state_j)
            energy_ij = self._compute_reduced_potential(sampler_state_i, thermo_state_j)
            energy_ji = self._compute_reduced_potential(sampler_state_j, thermo_state_i)

            # Accept or reject the swap.
            log_p_accept = - (energy_ij + energy_ji) + energy_ii + energy_jj
            if log_p_accept >= 0.0 or random() < math.exp(log_p_accept):
                # Swap states in replica slots i and j.
                self._thermodynamic_states[i] = thermo_state_j
                self._thermodynamic_states[j] = thermo_state_i

    def _compute_reduced_potential(self, sampler_state, thermo_state):
        # Obtain a Context to compute the energy with OpenMM. Any integrator will do.
        context, integrator = cache.global_context_cache.get_context(thermo_state)
        # Compute the reduced potential of the sampler_state configuration
        # in the given thermodynamic state.
        sampler_state.apply_to_context(context)
        return thermo_state.reduced_potential(context)

n_steps = 20

test_system = testsystems.AlanineDipeptideVacuum()
protocol = {'temperature': [300, 310, 330, 370, 450] * unit.kelvin}

"""
thermo_states = states.create_thermodynamic_state_protocol(mymolsys.system, protocol)

# Initialize replica initial configurations.
sampler_states = [states.SamplerState(positions=mymolsys.positions) for _ in thermo_states]

# Propagate the replicas with Langevin dynamics.
langevin_move = mcmc.LangevinSplittingDynamicsMove(timestep=2.0*unit.femtosecond, n_steps=n_steps)

# Run the parallel tempering simulation.
parallel_tempering = ReplicaExchange(thermo_states, sampler_states, langevin_move)

print ('before:', parallel_tempering._replicas_sampler_states[0].positions)
"""

#parallel_tempering.run(n_steps)

#print ('before:', parallel_tempering._replicas_sampler_states[0].positions)

n_replicas = 3  # Number of temperature replicas.
T_min = 298.0 * unit.kelvin  # Minimum temperature.
T_max = 600.0 * unit.kelvin  # Maximum temperature.

reference_state = states.ThermodynamicState(system=test_system.system, temperature=T_min)

move = mcmc.GHMCMove(timestep=2.0*unit.femtoseconds, n_steps=50)

simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=2)

storage_path = tempfile.NamedTemporaryFile(delete=False).name + '.nc'

reporter = MultiStateReporter(storage_path, checkpoint_interval=10)

simulation.create(reference_state, states.SamplerState(test_system.positions), reporter, min_temperature=T_min, max_temperature=T_max, n_temperatures=n_replicas)

simulation.run(n_iterations=1)

