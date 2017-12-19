'''
This example demonstrates how to obtain the list of neighbors for a structure.
'''
# Start import
from ase.build import bulk
from icetdev.neighbor_list import get_neighbor_lists

from icetdev.structure import Structure

# End import

# Generate an iceT structure from a 2x2x2 Al fcc supercell.
# Start setup
atoms = bulk("Al", "fcc", a=2).repeat(2)
atoms.pbc = [True, True, True]
structure = Structure.from_atoms(atoms)
# End setup

# Construct a list of all neighbors within the cutoff (1.5 A).
# Start neighbor
neighbor_cutoff = [1.5]
nl = get_neighbor_lists(structure, neighbor_cutoff)[0]
# End neighbor

# Loop over all atomic indices and print all of the neighbors.
# Start results
for index in range(len(atoms)):
    neighbors = nl.get_neighbors(index)
    print("Neighbors of atom with index {}".format(index))
    for neighbor in neighbors:
        neighbor_index = neighbor.index
        neighbor_offset = neighbor.unitcell_offset
        distance_to_neighbor = structure.get_distance(
            index, neighbor_index, [0, 0, 0], neighbor_offset)
        print("{0} {1} {2:1.5f}".format(neighbor_index,
                                        neighbor_offset, distance_to_neighbor))
    print("")
print("fcc has {} nearest neighbors".format(len(neighbors)))
# End results
