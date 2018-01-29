import spglib as spg
from ase.neighbor_list import NeighborList
from ase.db import connect
from tests import many_body_neighbor_listTester

'''
Testing manybodyneighbor_list implemented in python (mblnl.tester) againts ASE
neighbor_list

Parameters
----------
neighbor_cutoff : list
    cutoff radii for neighbor search
order : int
    highest order for manybody neighbor indices

Raises
------
AssertionError
    if lists of neighbors obatined from mbnl.tester and ASE are not the same
'''

mbnl_tester = many_body_neighbor_listTester.many_body_neighbor_listTester()

neighbor_cutoff = 1.4

db = connect('structures_for_testing.db')

for row in db.select('natoms>1'):

    atoms_row = row.toatoms()

    ase_nl = NeighborList(len(atoms_row) * [neighbor_cutoff / 2.0], skin=1e-8,
                          bothways=True, self_interaction=False)
    ase_nl.update(atoms_row)

    order = 3

    mbnl_tester = many_body_neighbor_listTester.many_body_neighbor_listTester()
    count_neighbors = {}

    dataset = spg.get_symmetry_dataset(atoms_row, symprec=1e-5,
                                       angle_tolerance=-1.0, hall_number=0)

    for index, equiv_index in enumerate(dataset['equivalent_atoms']):
        neighbors = mbnl_tester.build(order * [ase_nl], index, bothways=True)
        if equiv_index in count_neighbors:
            assert count_neighbors[equiv_index] == len(neighbors), \
                '''Testing number of neighbors from mbnl_tester with
                bothways=True failed for {} when counts {}!={}'''.format(
                    row.tag, len(neighbors), count_neighbors[equiv_index])
        else:
            count_neighbors[equiv_index] = len(neighbors)

    mbnl_tester = many_body_neighbor_listTester.many_body_neighbor_listTester()
    count_neighbors = {}

    for index, equiv_index in enumerate(dataset['equivalent_atoms']):
        neighbors = mbnl_tester.build(order * [ase_nl], index, bothways=False)
        if equiv_index in count_neighbors:
            assert count_neighbors[equiv_index] >= len(neighbors), \
                '''Testing number of neighbors from mbnl_tester with
                bothways=False failed for {} when counts {}<{}'''.format(
                    row.tag, len(neighbors), count_neighbors[equiv_index])
        else:
            count_neighbors[equiv_index] = len(neighbors)
