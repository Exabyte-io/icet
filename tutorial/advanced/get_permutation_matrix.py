'''
This example generate a permutation matrix for a structure
'''

# Import modules
import numpy as np
from ase.build import bulk
from icet.core.permutation_matrix import permutation_matrix_from_atoms

# Create a prototype Al structure
atoms = bulk('Al', 'fcc', a=2.0)

# Generate a permutation matrix for all neighbors inside the cutoff
neighbor_cutoff = 2.0
permutation_matrix, prim_structure, neighbor_list = \
    permutation_matrix_from_atoms(atoms, neighbor_cutoff)

# Extract the permuted, indexed and unique positions.
perm_pos = permutation_matrix.get_permuted_positions()
ind_pos, unique_pos = permutation_matrix.get_indexed_positions()

# Print the permuted, indexed and unique positions.
print('Permutated fractional coordinates')
for pp in perm_pos:
    unique_rows = np.vstack({tuple(row) for row in pp})
    for el in unique_rows:
        print(el, end=' ')
    print('')
print('Permutated indices and positions')
for i, pos in enumerate(ind_pos):
    print(i, len(set(pos)), pos)
print('Unique permuted indices and positions')
for index, dist in enumerate(unique_pos):
    print(index, dist)
