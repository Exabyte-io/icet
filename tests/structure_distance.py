from icetdev import structure
from ase import Atoms
import numpy as np
from ase.db import connect
from icetdev import *
from icetdev.structure import *
from ase.neighborlist import NeighborList

#Distance tolerance for comparing distances 
DISTTOL = 1e-8

# TODO: Will mic functionality be included in a near future. Otherwise testing with 
#       chain HOC is senseless
# TODO: Notation should be explicit to avoid conflicts between ase and icet namespace
# TODO: Improve the names for some icet functions, e.g. get_distance2

db = connect('structures_for_testing.db')

# Test distance calculator with offsets
for row in db.select():
    atoms_row = row.toatoms()
    structure = structure_from_atoms(atoms_row)
    nl = NeighborList(len(atoms_row)*[10])
    nl.update(atoms_row)
    for index in range(len(atoms_row)):
        indices, offsets = nl.get_neighbors(index)
        for i, offset in zip(indices, offsets):
            dist = np.linalg.norm(atoms_row.positions[index] - (atoms_row.positions[i] + np.dot(offset, atoms_row.get_cell())))
            dist_struct = structure.get_distance2(index, [0,0,0], i, offset)
            assert (dist - dist_struct) < DISTTOL, "Testing distance calculator failed for structure {}".format(row.tag)