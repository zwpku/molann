#!/usr/bin/env python

import openmm
from openmm import unit
from openmmtools import testsystems, integrators

water_box = testsystems.WaterBox(box_edge=2.0*unit.nanometer)
system = water_box.system  # An OpenMM System object.
positions = water_box.positions 

mymolsys = testsystems.LysozymeImplicit()

#mymolsys = testsystems.AlanineDipeptideVacuum()
molsystem = mymolsys.system

"""
pxylene_dsl = '(resname TMP) and (mass > 1.5)'  # Select heavy atoms of p-xylene.
binding_site_dsl = ('(resi 77 or resi 86 or resi 101 or resi 110 or '
                    ' resi 117 or resi 120) and (mass > 1.5)')

pxylene_atom_indices = lysozyme_pxylene.mdtraj_topology.select(pxylene_dsl).tolist()
binding_site_atom_indices = lysozyme_pxylene.mdtraj_topology.select(binding_site_dsl).tolist()
"""

integrator = integrators.LangevinIntegrator(temperature=298.0*unit.kelvin,
                                            collision_rate=1.0/unit.picoseconds,
                                            timestep=1.0*unit.femtoseconds)

context = openmm.Context(molsystem, integrator)
context.setPositions(mymolsys.positions)

print ('name of system: ', mymolsys.name)
print ('topology of system: ', mymolsys.topology)

n_steps = 1
integrator.step(n_steps)

#print (ADVacuum.positions)
