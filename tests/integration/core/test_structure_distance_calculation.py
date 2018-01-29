import numpy as np
from ase.db import connect
from ase.neighborlist import NeighborList as ASENeighborList
import icet

'''
Testing the calculation of distances with offsets
TODO:
    Delete this after edit unittest/test_structure
'''


''' Tolerance for comparing distances '''
DISTTOL = 1e-8

''' Fetch structures from database '''
db = connect('structures_for_testing.db')

print('')
for row in db.select():
    print(' structure: {}'.format(row.tag))

    atoms_row = row.toatoms()
    structure = icet.Structure.from_atoms(atoms_row)
    nl = ASENeighborList(len(atoms_row)*[2.6], self_interaction=False,)
    nl.update(atoms_row)
    for index in range(len(atoms_row)):
        indices, offsets = nl.get_neighbors(index)
        for i, offset in zip(indices, offsets):
            dvec = atoms_row.positions[index] - atoms_row.positions[i]
            dvec -= np.dot(offset, atoms_row.get_cell())
            dist_ase = np.linalg.norm(dvec)
            dist_struct = structure.get_distance(index, i, [0, 0, 0], offset)
            msg = 'Testing distance calculator failed'
            msg += ' for structure {}'.format(row.tag)
            assert dist_ase - dist_struct < DISTTOL, msg

    ''' Test interface to get_distance function '''
    try:
        dr = structure.get_distance(index, i, offset2=offset)
    except:  # NOQA
        assert False, 'get_distance function failed interface test (1)'

    try:
        dr = structure.get_distance(index, i, offset1=offset)
    except:  # NOQA
        assert False, 'get_distance function failed interface test (2)'
