import numpy as np
from ase.db import connect
import icet

_tolerance = 1e-9

'''
Testing icet structure against ASE
TODO:
    Delete this after edit unittest/test_structure
'''

''' Fetch structures from database '''
db = connect('structures_for_testing.db')

for row in db.select():

    atoms_row = row.toatoms()

    ''' Convert ASE atoms to icet structures '''
    structure = icet.Structure.from_atoms(atoms_row)

    ''' Test that structures have the same length '''
    msg = 'Test of len failed for structure {}'.format(row.tag)
    assert len(atoms_row) == len(structure), msg

    ''' Test that positions are equal '''
    for ase_pos, struct_pos in zip(atoms_row.positions,
                                   structure.positions):
        msg = 'Test for positions failed for structure {}'.format(row.tag)
        assert (np.abs(ase_pos - struct_pos) < _tolerance).all(), msg

    ''' Test that chemical symbols are equal '''
    for ase_symbol, struct_symbol in zip(atoms_row.get_atomic_numbers(),
                                         structure.get_atomic_numbers()):
        msg = 'Test for atomic numbers failed for structure {}'.format(row.tag)
        msg += '; {} != {}'.format(ase_symbol, struct_symbol)
        assert ase_symbol == struct_symbol, msg

    ''' Test periodic boundary conditions '''
    msg = 'Test for periodic boundary conditions failed'
    msg += ' for structure {}'.format(row.tag)
    assert (structure.pbc == atoms_row.pbc).all(), msg

    ''' Test unit cell equality '''
    msg = 'Test for cell failed for structure {}'.format(row.tag)
    assert (structure.cell == atoms_row.cell).all(), msg

    ''' Assert that structure return as an equal ASE atoms object '''
    atoms_icet = structure.to_atoms()
    msg = 'Test for conversion back to atoms failed'
    msg += ' for structure {}'.format(row.tag)
    assert atoms_icet == atoms_row, msg
