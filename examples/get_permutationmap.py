"""
This example generate a permutation map for a structure
"""
# Start import
import numpy as np

from ase.build import bulk

from icetdev.permutation_map import permutation_matrix_from_atoms
# End import

# Create a prototype Al structure in the form of a 1x1x1 unit cell.
# Start setup
atoms = bulk("Al", "fcc", a=2.0).repeat(1)
# End setup

# Generate a permutation map (matrix) for all neighbors within the cutoff
# (2.0 A).
# Start permutation
neighbor_cutoff = 2.0
permutation_map, prim_structure, neighbor_list =\
    permutation_matrix_from_atoms(atoms, neighbor_cutoff, verbosity=3)
# End permutation

# Extract the permutated positions.
# Start perm_pos
perm_pos = permutation_map.get_permutated_positions()
ind_pos, unique_pos = permutation_map.get_indiced_positions()
# End perm_pos

# Print the fractional coordinates for the permutated positions.
# Start frac_coor
print("Permutated fractional coordinates")
for pp in perm_pos:
    unique_rows = np.vstack({tuple(row) for row in pp})
    for el in unique_rows:
        print(el, end=' ')
    print("")
# End frac_coor

# Print the permutated indices and positions.
# Start ind_pos
print("Permutated indices and positions")
for i, pos in enumerate(ind_pos):
    print(i, len(set(pos)), pos)
# End ind_pos

# Print the permutated indices and positions.
# Start uni_ind_pos
print("Unique permutated indices and positions")
for index, dist in enumerate(unique_pos):
    print(index, dist)
# End uni_ind_pos
