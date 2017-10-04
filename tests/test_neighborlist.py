from icetdev import *
from icetdev.structure import *
import numpy.random as random
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.db import connect
import spglib as spg

"""
BUG: AssertionError: Testing len of neighborlist indices failed for structure PdH-slab-surface 
BUG: spglib symmetry dataset gives None for non periodic structures
"""


db = connect('structures_for_testing.db')

neighbor_cutoff = 6

for row in db.select('natoms>1'):

    atoms_row = row.toatoms()
    # ASE neighborlist
    ase_nl = NeighborList(len(atoms_row)*[neighbor_cutoff/2], skin=1e-8,
                              bothways=True, self_interaction=False)
    ase_nl.update(atoms_row)
    ase_indices, ase_offsets = ase_nl.get_neighbors(1)

    # icet neighborlist
    structure = structure_from_atoms(atoms_row)
    nl = Neighborlist(neighbor_cutoff)
    nl.build(structure)
    neighbors = nl.get_neighbors(1)
    indices = []
    offsets = []
    for nbr in neighbors:
        indices.append(nbr.index)
        offsets.append(nbr.unitcellOffset)
    """
    For debugging
    print(len(indices), len(ase_indices))
    for ind, offset in zip(indices, offsets):
        print(ind, offset, structure.get_distance2(0,[0,0,0],ind,offset))
        print("====================================")
    for ind, offset in zip(ase_indices, ase_offsets):
        print(ind, offset, structure.get_distance2(0,[0,0,0],ind,offset))
    """
    assert len(indices) == len(ase_indices), "Testing len of neighborlist indices failed for structure {}".format(row.tag)
    assert len(ase_offsets) == len(offsets), "Testing len of neighborlist offsets failed for structure {}".format(row.tag)

    for i, offset in zip(indices, offsets):
        assert offset in ase_offsets, "Testing each offset in neigborlist failed for structure {}".format(row.tag)
        eq_indices = [x for x, ase_offset in enumerate(ase_offsets) if ase_indices[x] == i and (ase_offset == offset).all()]
        if len(eq_indices) > 1:
            print(i, offset, eq_indices)
        assert len(eq_indices) == 1, "Testing duplicates offset failed for structure {}".format(row.tag)
        assert i == ase_indices[eq_indices[0]], "Testing indices for offsets failed for structure {}".format(row.tag)


    count_neighbors = {}
    dataset = spg.get_symmetry_dataset(atoms_row, symprec=1e-5, angle_tolerance=-1.0, hall_number=0)
    if not dataset == None:
        for atom, wyckoff in enumerate(dataset['wyckoffs']):
            atom_neighbors = nl.get_neighbors(atom)
            if wyckoff in count_neighbors:
                print(atom, wyckoff, count_neighbors[wyckoff],len(atom_neighbors))
                assert count_neighbors[wyckoff] == len(atom_neighbors), "Testing number of neighbors failed for structure {}".format(row.tag)
            else:
                count_neighbors[wyckoff] = len(atom_neighbors)
                print(atom, wyckoff, count_neighbors[wyckoff])