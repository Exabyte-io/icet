from ase.build import bulk
from ase.build import cut

import itertools

from icet import ClusterSpace
import unittest

tc = unittest.TestCase('__init__')

# initialize cluster space and get the internal primitive atoms object
prim = bulk('Au', a=4.0, crystalstructure='hcp')
subelements = ['Au', 'Pd']
cutoffs = [0.0]
cs = ClusterSpace(prim, cutoffs, subelements)

atoms_prim = cs.primitive_structure


# Create a supercell using permutation matrix
p_trial = [[1, 0, 0], [0, 1, 5], [0, 0, 2]]
atoms2 = cut(atoms_prim, p_trial[0], p_trial[1], p_trial[2])


# Setup cartesian input to generate a random population
cartesian_product_input = []
for i in range(len(atoms2)):
    cartesian_product_input.append(['Pd', 'Au'])

# Loop over element combinations and assert expected singlet value
for subset in itertools.product(*cartesian_product_input):
    for atom, element in zip(atoms2, subset):
        atom.symbol = element
    atoms = atoms2
    cv = cs.get_cluster_vector(atoms)
    expected_singlet = (-atoms.get_chemical_symbols().count("Pd") +
                        atoms.get_chemical_symbols().
                        count("Au")) / len(atoms)
    tc.assertAlmostEqual(cv[1], expected_singlet)
