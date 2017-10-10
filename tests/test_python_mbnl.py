from tests import manybodyNeighborlistTester
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.db import connect
import spglib as spg

"""
Testing manybodyneighborlist implemented in python (mblnl.tester) againts ASE neighborlist

Parameters:
    neighbor_cutoff : cutoff radii for neighbor search
    order(int) : highest order for manybody neighbor indices 

Raises: 
    AssertionError: if lists of neighbors obatined from mbnl.tester and ASE are not the same
"""

mbnl_tester = manybodyNeighborlistTester.manybodyNeighborlistTester()

neighbor_cutoff = 1.4

db = connect("structures_for_testing.db")

for row in db.select('natoms>1'):

    atoms_row = row.toatoms()

    ase_nl = NeighborList(len(atoms_row) * [neighbor_cutoff / 2.0], skin=1e-8,
                          bothways=True, self_interaction=False)
    ase_nl.update(atoms_row)

    order = 3

    mbnl_tester = manybodyNeighborlistTester.manybodyNeighborlistTester()
    count_neighbors = {}

    dataset = spg.get_symmetry_dataset(atoms_row, symprec=1e-5, angle_tolerance=-1.0, hall_number=0)

    for index, equiv_index in enumerate(dataset['equivalent_atoms']):
        neighbors = mbnl_tester.build(order * [ase_nl], index, bothways=True)
        if equiv_index in count_neighbors:
            #print(index, equiv_index, count_neighbors[equiv_index], len(neighbors))
            assert count_neighbors[equiv_index] == len(neighbors), "Testing number "\
                "of neighbors from mbnl_tester with bothways=True failed for {} "\
                "when counts {}!={}".format(row.tag, len(neighbors), count_neighbors[equiv_index])
        else:
            count_neighbors[equiv_index] = len(neighbors)
            #print(index, equiv_index, count_neighbors[equiv_index])

    mbnl_tester = manybodyNeighborlistTester.manybodyNeighborlistTester()
    count_neighbors = {}

    for index, equiv_index in enumerate(dataset['equivalent_atoms']):
        neighbors = mbnl_tester.build(order * [ase_nl], index, bothways=False)
        if equiv_index in count_neighbors:
            #print(index, equiv_index, count_neighbors[equiv_index], len(neighbors))
            assert count_neighbors[equiv_index] >= len(neighbors), "Testing number "\
                "of neighbors from mbnl_tester with bothways=False failed for {} "\
                "when counts {}<{}".format(row.tag, len(neighbors), count_neighbors[equiv_index])
        else:
            count_neighbors[equiv_index] = len(neighbors)
            #print(index, equiv_index, count_neighbors[equiv_index])
