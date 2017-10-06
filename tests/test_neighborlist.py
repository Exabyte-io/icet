from icetdev import *
from icetdev.structure import *
import numpy.random as random
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.db import connect
import spglib as spg

"""
Include some docstring here explaining the current test

Raises: 
    AssertionError: if 
"""


db = connect('structures_for_testing.db')

neighbor_cutoff = 1.6

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
    for ngb in neighbors:
        indices.append(ngb.index)
        offsets.append(ngb.unitcellOffset)

    assert len(indices) == len(ase_indices), "Testing size of neighborlist indices "\
        "failed for {} when gives {} != {} ".format(row.tag, len(indices), len(ase_indices))
    assert len(offsets) == len(ase_offsets), "Testing size of neighborlist offsets "\
        "failed for  {} when gives {} != {} ".format(row.tag, len(offsets), len(ase_offsets))

    for i, offset in zip(indices, offsets):
        assert offset in ase_offsets, "Testing each offset in neigborlist failed "\
            "for {}".format(row.tag)
        equiv_indices = [x for x, ase_offset in enumerate(ase_offsets) if ase_indices[x] == i and (ase_offset == offset).all()]
        if len(equiv_indices) > 1:
            print(i, offset, equiv_indices)
        assert len(equiv_indices) == 1, "Testing duplicates offset failed for {}".format(row.tag)
        assert i == ase_indices[equiv_indices[0]], "Testing indices for offsets failed"\
            "for {}".format(row.tag)


    count_neighbors = {}
    inequiv_index = {}
    dataset = spg.get_symmetry_dataset(atoms_row, symprec=1e-5, angle_tolerance=-1.0, hall_number=0)
    for index, equiv_index in enumerate(dataset['equivalent_atoms']):
        neighbors = nl.get_neighbors(index)
        if equiv_index in count_neighbors:
            #print(index, equiv_index, count_neighbors[equiv_index], len(neighbors))
            assert count_neighbors[equiv_index] == len(neighbors), "Testing number of neighbors "\
                "failed for {} when counts {}({}) at atom {}({}) ".format(
                    row.tag, len(neighbors), count_neighbors[equiv_index], index, inequiv_index[equiv_index])
        else:
            count_neighbors[equiv_index] = len(neighbors)
            inequiv_index[equiv_index] = index
            #print(index, equiv_index, count_neighbors[equiv_index], len(neighbors))
