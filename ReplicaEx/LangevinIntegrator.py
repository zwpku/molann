#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Generate MD trajectory data using OpenMM 

# +
import math
from random import random, randint
import numpy as np
import time
import datetime
import mdtraj
import os
import warnings
from sys import stdout
import matplotlib.pyplot as plt

# import openmm
from openmm import *
from openmm.app import *
# -

# ### MD simulation

# +
# This PDB file describes AlanineDipeptide in vacuum. 
# It is from the OpenMM source code.
pdb_filename = './AlanineDipeptideOpenMM/vacuum.pdb'

n_steps = 20000 # Total simulation steps, i.e. number of states.
Temp = 498.0 * unit.kelvin  # temperature.
output_path = './Langevin_output'
traj_dcd_filename = '%s/traj.dcd' % output_path
csv_filename = '%s/state_data.csv' % output_path
report_interval_dcd = 100
report_interval_stdout = 1000
report_interval_csv = 100

print ( 'trajectory will be saved to file: %s' % traj_dcd_filename )

# prepare before simulation
pdb = PDBFile(pdb_filename)
forcefield = ForceField('amber14-all.xml')
system = forcefield.createSystem(pdb.topology, nonbondedCutoff=2*unit.nanometer, constraints=HBonds)
integrator = LangevinIntegrator(Temp, 1/unit.picosecond, 2*unit.femtoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

# registrate reporter for output
simulation.reporters = []
simulation.reporters.append(DCDReporter(traj_dcd_filename, report_interval_dcd))
simulation.reporters.append(StateDataReporter(stdout, report_interval_stdout, step=True,
                                              temperature=True, elapsedTime=True))
simulation.reporters.append(StateDataReporter(csv_filename, report_interval_csv, time=True,
                                              potentialEnergy=True, totalEnergy=True, temperature=True))

# run the simulation
print ('Simulation starts...')
start = time.time()
simulation.step(n_steps)
end = time.time()
print ( 'Simulation ends, %d sec. elapsed.' % (end - start) )

del simulation
