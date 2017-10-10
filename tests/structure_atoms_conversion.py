from ase import Atoms
from ase.db import connect
from ase.io import read
import numpy as np
from icetdev import *
from icetdev.structure import *

"""
Testing icet structure against ASE 

"""

""" Set up structures from database """
db = connect("structures_for_testing.db")

for row in db.select():

    atoms_row = row.toatoms()

    """ Convert ASE atoms to icet structures """
    structure = structure_from_atoms(atoms_row)

    """ Test that each position is equal """
    assert len(atoms_row.positions) == len(structure.get_positions()), "Test for len of positions failed for structure {}".format(raw.tag) 
    
    for ase_pos, struct_pos in zip(atoms_row.positions, structure.get_positions()):
        assert (ase_pos == struct_pos).all(), "Test for positions failed for structure {}".format(row.tag)

    """ Test that each element is equal """
    assert len(atoms_row.get_chemical_symbols()) == len(structure.get_elements()), "Test for len of elements failed for for structure {}".format(atoms_row.tag)
    
    for ase_symbol, struct_symbol in zip(atoms_row.get_atomic_numbers(),
                                         structure.get_elements()):
        assert ase_symbol == struct_symbol,"Test for elements failed for structure {} when {} don't match {}".format(row.tag, ase_symbol, struct_symbol)

    """ Test pbc equality """
    assert (structure.pbc == atoms_row.pbc).all(), "Test for periodic boundary conditions failed for structure {}".format(row.tag)

    """ Test unit cell equality """
    assert (structure.cell == atoms_row.cell).all(), "Test for cell failed for structure {}".format(row.tag)

    """ Assert that structure return as an equal ASE atoms object """
    atoms_icet = structure.to_atoms()
    assert atoms_icet == atoms_row, "Test for conversion back to atoms failed for structure {}".format(row.tag)

