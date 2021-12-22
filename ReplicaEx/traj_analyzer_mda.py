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

import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals, rms
import matplotlib.pyplot as plt
import numpy as np
import nglview as nv


u = mda.Universe("./AlanineDipeptideOpenMM/vacuum.pdb", 'traj.dcd')
ref = mda.Universe("./AlanineDipeptideOpenMM/vacuum.pdb")
print (u.residues)
print (u.trajectory)
print (ref.trajectory)

# +
nv.show_mdanalysis(u)
res = u.residues[1]

phi = res.phi_selection()
res
#protein = u.select_atoms('protein')
#protein
# -

r = dihedrals.Ramachandran(u.select_atoms('resid 2')).run()
r.plot()

selector = 'name N or name CA or name C'
#sel = u.select_atoms(selector)
#print (sel)
R = rms.RMSD(u, ref, select=selector)          
R.run()

rmsd = R.results.rmsd.T   # transpose makes it easier for plotting
print (rmsd.shape)
fig = plt.figure(figsize=(4,4))
plt.plot(rmsd[0,:], rmsd[2,:], 'k-')
#ax.legend(loc="best")
plt.xlabel("time (fs)")
plt.ylabel(r"RMSD ($\AA$)")
#plt.xlim([0,2000])
#fig.savefig("rmsd_all_CORE_LID_NMP_ref1AKE.pdf")


