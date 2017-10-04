from tests import manybodyNeighborlistTester
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.db import connect
import spglib as spg

"""
BUG: AssertionError: Testing number of neighbors from mbnl_tester with 
     bothways=True for different indexes failed for structure BaZrO3-perovskite

"""


mbnl_tester = manybodyNeighborlistTester.manybodyNeighborlistTester()

neighbor_cutoff = 6.

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

    if not dataset == None:
        for index, wyckoff in enumerate(dataset['wyckoffs']):
            neighbors = mbnl_tester.build(order * [ase_nl], index, bothways=True)
            if wyckoff in count_neighbors:
                print(index, wyckoff, count_neighbors[wyckoff],len(neighbors))
                assert count_neighbors[wyckoff] == len(neighbors), "Testing number "\
                "of neighbors from mbnl_tester with bothways=True failed for "\
                "structure {}".format(row.tag)
            else:
                count_neighbors[wyckoff] = len(neighbors)
                print(index, wyckoff, count_neighbors[wyckoff])

    print("second part ...")
    mbnl_tester = manybodyNeighborlistTester.manybodyNeighborlistTester()
    count_neighbors = {}

    if not dataset == None:
        for index, wyckoff in enumerate(dataset['wyckoffs']):
            neighbors = mbnl_tester.build(order * [ase_nl], index, bothways=False)
            if wyckoff in count_neighbors:
                print(index, wyckoff, count_neighbors[wyckoff],len(neighbors))
                assert count_neighbors[wyckoff] == len(neighbors), "Testing number "\
                "of neighbors from mbnl_tester with bothways=False failed for "\
                "structure {}".format(row.tag)
            else:
                count_neighbors[wyckoff] = len(neighbors)
                print(index, wyckoff, count_neighbors[wyckoff])

