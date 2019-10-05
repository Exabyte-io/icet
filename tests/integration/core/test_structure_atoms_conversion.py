import numpy as np
from ase.db import connect
import icet

_tolerance = 1e-9

"""
Testing icet structure against ASE
TODO:
    Delete this after edit unittest/test_structure
"""

""" Fetch structures from database """
db = connect('structures_for_testing.db')

for row in db.select():

    ase_atoms = row.toatoms()

    """ Convert ASE atoms to icet structures """
    icet_structure = icet.Structure.from_atoms(ase_atoms)

    """ Test that structures have the same length """
    msg = 'Test of len failed for structure {}'.format(row.tag)
    assert len(ase_atoms) == len(icet_structure), msg

    """ Test that positions are equal """
    for ase_pos, struct_pos in zip(ase_atoms.positions,
                                   icet_structure.positions):
        msg = 'Test for positions failed for structure {}'.format(row.tag)
        assert (np.abs(ase_pos - struct_pos) < _tolerance).all(), msg

    """ Test that chemical symbols are equal """
    for ase_symbol, struct_symbol in zip(ase_atoms.get_chemical_symbols(),
                                         icet_structure.get_chemical_symbols()):
        msg = 'Test for atomic numbers failed for structure {}'.format(row.tag)
        msg += '; {} != {}'.format(ase_symbol, struct_symbol)
        assert ase_symbol == struct_symbol, msg

    """ Test periodic boundary conditions """
    msg = 'Test for periodic boundary conditions failed'
    msg += ' for structure {}'.format(row.tag)
    assert (icet_structure.pbc == ase_atoms.pbc).all(), msg

    """ Test unit cell equality """
    msg = 'Test for cell failed for structure {}'.format(row.tag)
    assert (icet_structure.cell == ase_atoms.cell).all(), msg

    """ Assert that structure return as an equal ASE atoms object """
    atoms_from_icet = icet_structure.to_atoms()
    msg = 'Test for conversion back to atoms failed'
    msg += ' for structure {}'.format(row.tag)
    assert atoms_from_icet == ase_atoms, msg
