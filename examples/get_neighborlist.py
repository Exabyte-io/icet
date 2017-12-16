'''
This examples demonstrates how to obtain the list of neighbors for a structure.
'''

# Start import
from ase.build import bulk
<<<<<<< HEAD
from icetdev.neighbor_list import get_neighbor_lists
from icetdev.structure import Structure
# End import

=======
<<<<<<< HEAD
from icetdev.neighbor_list import get_neighbor_lists 
=======
from icetdev.neighborlist import Neighborlist
>>>>>>> fef092fd0ad3f79c8bb00d4560a9ac025f4dae6b
from icetdev.structure import Structure
# End import


>>>>>>> 6844742a734f72f18f511b85533342dd47d6efe7
# Generate an iceT structure from a 2x2x2 Al fcc supercell.
# Start setup
atoms = bulk("Al", "fcc", a=2).repeat(2)
atoms.pbc = [True, True, True]
<<<<<<< HEAD
structure = Structure.from_atoms(atoms)
# End setup

# Construct a list of all neighbors within the cutoff (1.5 A).
# Start neighbor
neighbor_cutoff = [1.5]
nl = get_neighbor_lists(structure, neighbor_cutoff)[0]
=======
<<<<<<< HEAD
structure = Structure.from_atoms(atoms)
=======
structure = Structure().from_atoms(atoms)
>>>>>>> fef092fd0ad3f79c8bb00d4560a9ac025f4dae6b
# End setup


# Construct a list of all neighbors within the cutoff (1.5 A).
# Start neighbor
<<<<<<< HEAD
neighbor_cutoff = [1.5]
nl = get_neighbor_lists(structure, neighbor_cutoff)[0]
=======
neighbor_cutoff = 1.5
nl = Neighborlist(neighbor_cutoff)
nl.build(structure)
>>>>>>> fef092fd0ad3f79c8bb00d4560a9ac025f4dae6b
>>>>>>> 6844742a734f72f18f511b85533342dd47d6efe7
# End neighbor

# Loop over all atomic indices and print all of the neighbors.
# Start results
for index in range(len(atoms)):
    neighbors = nl.get_neighbors(index)
    print("Neighbors of atom with index {}".format(index))
    for neighbor in neighbors:
        neighbor_index = neighbor.index
<<<<<<< HEAD
        neighbor_offset = neighbor.unitcell_offset
        distance_to_neighbor = structure.get_distance(
            index, neighbor_index, [0, 0, 0], neighbor_offset)
=======
<<<<<<< HEAD
        neighbor_offset = neighbor.unitcell_offset
        distance_to_neighbor = structure.get_distance(
            index, neighbor.index, [0, 0, 0], neighbor.unitcell_offset)
=======
        neighbor_offset = neighbor.unitcellOffset
        distance_to_neighbor = structure.get_distance(
            index, neighbor.index, [0, 0, 0], neighbor.unitcellOffset)
>>>>>>> fef092fd0ad3f79c8bb00d4560a9ac025f4dae6b
>>>>>>> 6844742a734f72f18f511b85533342dd47d6efe7
        print("{0} {1} {2:1.5f}".format(neighbor_index,
                                        neighbor_offset, distance_to_neighbor))
    print("")
print("fcc has {} nearest neighbors".format(len(neighbors)))
# End results
