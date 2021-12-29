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
import os, sys
import warnings
from sys import stdout
import matplotlib.pyplot as plt
import configparser

# import openmm
from openmm import *
from openmm.app import *
from openmmplumed import PlumedForce
# -

# ### set parameters

config = configparser.ConfigParser()
config.read('params.cfg')
pdb_filename = config['Default']['pdb_filename']
n_steps = config['Default'].getint('n_steps')
Temp = config['Default'].getfloat('Temperature') * unit.kelvin
traj_dcd_filename = config['Default']['traj_dcd_filename']
csv_filename = config['Default']['csv_filename']
report_interval_dcd = config['Default'].getint('report_interval_dcd')
report_interval_stdout = config['Default'].getint('report_interval_stdout')
report_interval_csv = config['Default'].getint('report_interval_csv')

# ### MD simulation

# +
print ( 'trajectory will be saved to file: %s' % traj_dcd_filename )

# prepare before simulation
pdb = PDBFile(pdb_filename)
forcefield = ForceField('amber14-all.xml')

system = forcefield.createSystem(pdb.topology, nonbondedCutoff=2*unit.nanometer, constraints=HBonds)

script = """
d: DISTANCE ATOMS=1,10
METAD ARG=d SIGMA=0.2 HEIGHT=0.3 PACE=500"""
system.addForce(PlumedForce(script))

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
# -

