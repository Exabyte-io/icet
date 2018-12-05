import numpy as np

from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
from os import remove
from os.path import exists

fname = 'AlGa.traj'
if exists(fname):
    remove(fname)

# Set up the structure we want to simulate
atoms = bulk('Al').repeat(3)
for i, atom in enumerate(atoms):
    if i % 2 == 0:
        atom.symbol = 'Ga'

# Create a cluster space
cutoffs = [5, 3]
elements = ['Al', 'Ga']
chemical_potentials = {'Al': 5, 'Ga': 0}
cs = ClusterSpace(atoms, cutoffs, elements)

# Create some parameters and make a cluster expansion
parameters = np.array([1.2] * len(cs))
ce = ClusterExpansion(cs, parameters)

# Set up a calculator
calculator = ClusterExpansionCalculator(atoms, ce)

# Finally we have the ensemble
ensemble = SemiGrandCanonicalEnsemble(
    calculator=calculator, atoms=atoms,
    random_seed=42, temperature=100.0,
    chemical_potentials=chemical_potentials,
    ensemble_data_write_interval=10,
    trajectory_write_interval=10)

# Let's take it for a spin
ensemble.run(100)

print("Acceptance ratio {}".format(ensemble.acceptance_ratio))

print(ensemble.data_container.data)

# Write trajectory to file that serves to visualize with ASE GUI
# the trajectory with energy for each frame
ensemble.data_container.write_trajectory(fname)
