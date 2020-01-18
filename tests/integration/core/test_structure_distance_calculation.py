import numpy as np
from icet import Structure
from ase.db import connect
from ase.neighborlist import NeighborList as ASENeighborList

"""
Testing the calculation of distances with offsets
TODO:
    Delete this after edit unittest/test_structure
"""

""" Fetch structures from database """
db = connect('structures_for_testing.db')
for row in db.select():
    structure = row.toatoms()
    nl = ASENeighborList(len(structure) * [2.6], self_interaction=False)
    nl.update(structure)
    for index in range(len(structure)):
        indices, offsets = nl.get_neighbors(index)
        for i, offset in zip(indices, offsets):
            dvec = structure.positions[index] - structure.positions[i]
            dvec -= np.dot(offset, structure.cell)
            dist_ase = np.linalg.norm(dvec)
            dist_struct = Structure.from_atoms(structure).get_distance(
                index,
                i,
                [0, 0, 0],
                offset)
            msg = 'Testing distance calculator failed'
            msg += ' for structure {}'.format(row.tag)
            assert dist_ase - dist_struct < 1e-8, msg
