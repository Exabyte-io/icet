'''
This example generate a permutation map for a structure
'''

# Import modules
import numpy as np
from ase.build import bulk
from icet.core.permutation_map import permutation_matrix_from_atoms

# Create a prototype Al structure
atoms = bulk('Al', 'fcc', a=2.0)

# Generate a permutation map (matrix) for all neighbors inside the cutoff
neighbor_cutoff = 2.0
permutation_map, prim_structure, neighbor_list = \
    permutation_matrix_from_atoms(atoms, neighbor_cutoff, verbosity=3)

# Extract the permutated, indexed and unique positions.
perm_pos = permutation_map.get_permutated_positions()
ind_pos, unique_pos = permutation_map.get_indiced_positions()

# Print the permutated, indexed and unique positions.
print('Permutated fractional coordinates')
for pp in perm_pos:
    unique_rows = np.vstack({tuple(row) for row in pp})
    for el in unique_rows:
        print(el, end=' ')
    print('')
print('Permutated indices and positions')
for i, pos in enumerate(ind_pos):
    print(i, len(set(pos)), pos)
print('Unique permutated indices and positions')
for index, dist in enumerate(unique_pos):
    print(index, dist)
