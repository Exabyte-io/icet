# import modules
import matplotlib.pyplot as plt
import numpy as np
from ase.db import connect
from ase.build import bulk
from icetdev import (ClusterSpace,
                     StructureContainer,
                     Optimizer,
                     ClusterExpansion)
from icetdev.tools import enumerate_structures

# step 1: Set up the basic structure and a cluster space
prim = bulk('Au')
cutoffs = [6.0, 5.0, 4.0]
subelements = ['Ag', 'Au']
cs = ClusterSpace(prim, cutoffs, subelements)
print(cs)

# step 2: Parse the input structures and set up a structure container
db = connect('structures.db')
# get reference energies for the elements
eref = {}
for elem in subelements:
    for row in db.select('{}=1'.format(elem), natoms=1):
        eref[elem] = row.energy / row.natoms
        break
# compile the structures into a structure container and add the mixing energies
atoms_list = []
properties = []
for row in db.select():
    conc = float(row.count_atoms().get('Ag', 0)) / row.natoms
    emix = row.energy / row.natoms
    emix -= conc * eref['Ag'] + (1.0 - conc) * eref['Au']
    properties.append({'energy': emix})
    atoms = row.toatoms()
    atoms.set_positions(row.data['original_positions'])
    atoms_list.append(atoms)
sc = StructureContainer(cs, atoms_list, properties)
print(sc)

# step 3: Train parameters
opt = Optimizer(sc.get_fit_data())
opt.train()
print(opt)

# step 4: Compare predicted and target data
ce = ClusterExpansion(cs, opt.parameters)
data = []
for row in db.select():
    conc = float(row.count_atoms().get('Ag', 0)) / row.natoms
    emix = row.energy / row.natoms
    emix -= conc * eref['Ag'] + (1.0 - conc) * eref['Au']
    atoms = row.toatoms()
    atoms.set_positions(row.data['original_positions'])
    emix_ce = ce.predict(atoms)
    data.append([conc, emix, emix_ce])
data = np.array(data).T
# plot the results
fig, ax = plt.subplots()
ax.set_xlabel(r'Ag concentration')
ax.set_ylabel(r'Mixing energy (meV/atom)')
ax.set_xlim([0, 1])
ax.scatter(data[0], 1e3 * data[1], marker='o')
ax.scatter(data[0], 1e3 * data[2], marker='x')
plt.savefig('mixing-energy-comparison.png', bbox_inches='tight')

# step 5: Predict energies for, multiple, enumerated structures
# enumerate the structures and compile the predicted energies
data = []
for atoms in enumerate_structures(prim, range(1, 12), subelements):
    conc = float(atoms.get_chemical_symbols().count('Ag')) / len(atoms)
    emix = ce.predict(atoms)
    data.append([conc, emix])
print('Predicted energies for {} structures'.format(len(data)))
data = np.array(data).T
# plot the results
fig, ax = plt.subplots()
ax.set_xlabel(r'Ag concentration')
ax.set_ylabel(r'Mixing energy (meV/atom)')
ax.set_xlim([0, 1])
ax.scatter(data[0], 1e3 * data[1], marker='x')
plt.savefig('mixing-energy-predicted.png', bbox_inches='tight')
