from ase import Atoms
from ase.db import connect
from ase.neighborlist import NeighborList
from icetdev import *
from icetdev.manybodyNeighborlist import *
from icetdev.structure import *
from tests import manybodyNeighborlistTester
import spglib as spg

"""
This test compares manybodyneighborlist obtained from c++ implementation (mbnl.icetdev)
and the one obtained from python implmentation (mbnl.tester)

Parameters:
    neighbor_cutoff : cutoff radii for neighbor search
    order(int) : highest order for manybody neighbor indices 

Raises:
    AssertionError: if neighbors in manybodyneighborlist are not the same in both implementations 
"""

neighbor_cutoff = 1.4

db = connect("structures_for_testing.db")

for row in db.select('natoms>1'):
    atoms_row = row.toatoms()
    structure = structure_from_atoms(atoms_row)

    """ Set up icet neighborlist for input to manybody neighborlist """
    nl = Neighborlist(neighbor_cutoff)
    nl.build(structure)
    ngb_1 = nl.get_neighbors(0)
    ngb_2 = nl.get_neighbors(1)

    """ Set up manybody neighborlist """
    mbnl = ManybodyNeighborlist()

    """" This is intersect between neighbors of atom 0 and atom 1 """
    intersect = mbnl.calc_intersection(ngb_1, ngb_2)

    """ Test intersect by doing a naive intersect """
    naive_intersect = []
    for n1 in ngb_1:
        for n2 in ngb_2:
            if n1.index == n2.index and (n1.unitcellOffset == n2.unitcellOffset).all():
                naive_intersect.append(n1)

    """ Assert that all the intersects are equal """
    for n1, n2 in zip(intersect, naive_intersect):
        assert n1.index == n2.index and (
            n1.unitcellOffset == n2.unitcellOffset).all(), "Testing for instersects from mbnl "\
            "failed for {}".format(row.tag)


    # test icetdev mbnl
    order = 5
    bothways = True
    count_neighbors = {}
    inequiv_index = {}

    dataset = spg.get_symmetry_dataset(atoms_row, symprec=1e-5, angle_tolerance=-1.0, hall_number=0)

    for index, equiv_index in enumerate(dataset['equivalent_atoms']):
        neighbors = mbnl.build(order * [nl], index, bothways)
        if equiv_index in count_neighbors:
            #print(index, equiv_index, count_neighbors[equiv_index], len(neighbors))
            assert count_neighbors[equiv_index] == len(neighbors), "Testing number "\
                "of neighbors from mbnl with bothways=True failed for "\
                "structure {}".format(row.tag)
        else:
            count_neighbors[equiv_index] = len(neighbors)
            inequiv_index[equiv_index] = index
            #print(index, equiv_index, count_neighbors[wyckoff])



    """ Compare neighborlists from mbnl.icetdev and mbnl.tester """
    mbnl_tester = manybodyNeighborlistTester.manybodyNeighborlistTester()

    ase_nl = NeighborList(len(atoms_row) * [neighbor_cutoff / 2.0], skin=1e-8,
                          bothways=True, self_interaction=False)
    ase_nl.update(atoms_row)

    bothways = True
    max_order = 4

    for i in inequiv_index:
        for j in range(2, max_order):
            index = inequiv_index[i]
            order = j
            ngb_tester = mbnl_tester.build(
                (order - 1) * [ase_nl], index, bothways)
            ngb_icetdev = mbnl.build((order - 1) * [nl], index, bothways)
            assert len(ngb_tester) == len(ngb_icetdev), "Testing number of neighbors "\
                "from mbnl and mbnl.tester failed with bothways=True at index {0} "\
                "with order {1} for {3}".format(index, order, row.tag)


    """ Test that bothways = false also works """
    bothways = False
    for i in inequiv_index:
        for j in range(1, max_order):
            index = inequiv_index[i]
            order = j
            ngb_tester = mbnl_tester.build(order * [ase_nl], index, bothways)
            ngb_icetdev = mbnl.build(order * [nl], index, bothways)
            assert len(ngb_tester) == len(ngb_icetdev), "Testing number of neighbors from "\
                "from mbnl and mbnl.tester failed with bothways=False at index {0} "\
                "with order {1} for {3}".format(index, order, row.tag)

