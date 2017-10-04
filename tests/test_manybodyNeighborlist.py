import numpy as np
import numpy.random as random
from ase import Atoms
from ase.db import connect
from ase.neighborlist import NeighborList
from icetdev import *
from icetdev.manybodyNeighborlist import *
from icetdev.structure import *
from tests import manybodyNeighborlistTester

"""
OBS: structure from icet, neighborlist from icet/ase, manybodyNL from icet 
BUG: (*) note that currently test at row 44 fails if cutoff is 6.0
"""

neighbor_cutoff = 6.1

db = connect("structures_for_testing.db")

for row in db.select():
    atoms_row = row.toatoms()
    structure = structure_from_atoms(atoms_row)

    # set up icet neighborlist for input to manybody neighborlist
    nl = Neighborlist(neighbor_cutoff)
    nl.build(structure)
    ngbs_1 = nl.get_neighbors(0)
    ngbs_2 = nl.get_neighbors(1)

    # set up manybody neighborlist
    mbnl = ManybodyNeighborlist()

    # this is intersect between neighbors of atom 0 and atom 1
    intersect = mbnl.calc_intersection(ngbs_1, ngbs_2)

    # test intersect by doing a naive intersect
    naive_intersect = []
    for n1 in ngbs_1:
        for n2 in ngbs_2:
            if n1.index == n2.index and (n1.unitcellOffset == n2.unitcellOffset).all():
                naive_intersect.append(n1)

    # assert that all the intersects are equal
    for n1, n2 in zip(intersect, naive_intersect):
        assert n1.index == n2.index and (
            n1.unitcellOffset == n2.unitcellOffset).all(), "Testing for instersects from mbnl.build "/
            "failed for structure {}".format(row.tag)


    # test actual mbnl
    order = 5
    bothways = True
    index1 = 0
    index2 = len(atoms_row) - 1
    ngbs_1 = mbnl.build(order * [nl], index1, bothways)
    ngbs_2 = mbnl.build(order * [nl], index2, bothways)
    
    # This assertion may fail for inequivalent sites e.g. perovskites
    assert len(ngbs_1) == len(ngbs_2), "Testing for numbers of neighbors from mbnl.built"\
        "under bothway=True failed for structure {}".format(row.tag)


    # get manybodyNeighbors to third order
    mbnl_tester = manybodyNeighborlistTester.manybodyNeighborlistTester()
    
    ase_nl = NeighborList(len(atoms_row) * [neighbor_cutoff / 2.0], skin=1e-8,
                      bothways=True, self_interaction=False)
    ase_nl.update(atoms_row)

    maxorder = 4
    bothways = True
    index = 0
    for i in range(len(atoms_row)):
        for j in range(2, maxorder):
            index = i
            order = j
            nbgs_tester = mbnl_tester.build(
                (order - 1) * [ase_nl], index, bothways)
            nbgs_build = mbnl.build((order - 1) * [nl], index, bothways)
            assert len(nbrs_tester) == len(nbrs_build), "Testing for number of neighbors with "\
                "mbnl.build and mbnl.tester failed at index {0} with order {1} for "\
                "structure {3}".format(i, j, row.tag)


    # test that bothways = false also works
    bothways = False
    for i in range(len(a)):
        for j in range(1, maxorder):
            index = i
            order = j
            nbrs_tester = mbnl_T.build(order * [ase_nl], index, bothways)
            nbrs_cpp = mbnl.build(order * [nl], index, bothways)
            assert len(nbrs_tester) == len(nbrs_cpp), "python mbnl and cpp mbnl do not give same amount of "\
                "neighbors at index {0} with order {1} was not equal. {2} != {3}".format(
                i, j, len(nbrs_tester), len(nbrs_cpp))


    # debug
    def printNeighbor(nbr, onlyIndice=False):
        if onlyIndice:
            print(nbr.index, end=" ")
        else:
            print(nbr.index, nbr.unitcellOffset, end=" ")
