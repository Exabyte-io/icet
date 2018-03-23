import unittest

from ase.build import bulk
from icet import Structure
from icet.core.lattice_site import LatticeSite
from icet.core.neighbor_list import get_neighbor_lists
from icet.core.many_body_neighbor_list import (
    ManyBodyNeighborList)


class TestManyBodyNeighborList(unittest.TestCase):
    """
    Container for test of the module functionality.

    """

    def __init__(self, *args, **kwargs):
        super(TestManyBodyNeighborList, self).__init__(*args, **kwargs)

        self.atoms = bulk('Ni', 'hcp', a=3.0).repeat([2, 2, 1])
        self.cutoffs = [5.0, 5.0]

    def setUp(self):
        """
        SetUp
        """
        self.mbnl = ManyBodyNeighborList()
        structure = Structure.from_atoms(self.atoms)
        self.neighbor_lists = get_neighbor_lists(structure, self.cutoffs)

    def test_build(self):
        """
        Test build.
        """
        for index in range(len(self.atoms)):
            self.mbnl.build(self.neighbor_lists, index, True)

    def test_bothways_true(self):
        """
        Build the mbnl with bothways = True and assert that
        each index in the atoms object have the same number
        of neighbors.
        """
        mbnl_size = len(self.mbnl.build(self.neighbor_lists, 0, True))
        for index in range(1, len(self.atoms)):
            self.assertEqual(mbnl_size, len(self.mbnl.build(
                self.neighbor_lists, index, True)))

    def test_bothways_false(self):
        """
        Build the mbnl with bothways = False and assert that
        mbnl built on the first index in the atoms object do not
        have the same number of neighbors as the other atoms.
        """
        mbnl_size = len(self.mbnl.build(self.neighbor_lists, 0, False))
        for index in range(1, len(self.atoms)):
            self.assertNotEqual(mbnl_size, len(self.mbnl.build(
                self.neighbor_lists, index, False)))

    def test_singlets(self):
        """
        Test that every singlet lattice site is listed
        in the many-body neighbor list.
        """
        for index in range(len(self.atoms)):
            target = tuple(([LatticeSite(index, [0., 0., 0.])], []))
            singlet = self.mbnl.build(self.neighbor_lists, index, False)[0]
            self.assertEqual(singlet, target)

    def test_pairs(self):
        """
        Test that many-body_neighbor list includes
        all the pairs returned by  neighbor_list for
        a specific lattice site.
        """
        index = 0
        nl_neighbors = self.neighbor_lists[0].get_neighbors(0)
        target = tuple(([LatticeSite(index, [0., 0., 0.])], nl_neighbors))
        pairs = self.mbnl.build(self.neighbor_lists, index, True)[1]
        self.assertEqual(pairs, target)

    def test_higher_order_neighbors(self):
        """
        Test higher order neighbors in many-body
        neighbor list for a specific lattice site.
        """
        index = 0
        high_order_neighbors = \
            self.mbnl.build(self.neighbor_lists, index, False)[2]

        target = ([LatticeSite(0, [0, 0, 0]), LatticeSite(0, [0, 0, 1])],
                  [LatticeSite(1, [0, 0, 0]),
                   LatticeSite(3, [0, -1, 0]),
                   LatticeSite(5, [-1, -1, 0]),
                   LatticeSite(5, [-1, 0, 0]),
                   LatticeSite(5, [0, 0, 0]),
                   LatticeSite(7, [-1, -1, 0])])

        self.assertEqual(target, high_order_neighbors)

    def test_calculate_intersections(self):
        """
        Test intersection between two list of neighbors.
        """
        lattice_sites = []
        lattice_sites.append(LatticeSite(0, [0, 0, 0]))
        lattice_sites.append(LatticeSite(0, [1, 0, 0]))
        lattice_sites.append(LatticeSite(1, [0, 0, 0]))
        lattice_sites.append(LatticeSite(3, [0, 0, 0]))

        lattice_sites2 = []
        lattice_sites2.append(LatticeSite(0, [0, 0, 0]))
        lattice_sites2.append(LatticeSite(0, [1, 0, 0]))

        intersection = self.mbnl.calculate_intersection(
            lattice_sites, lattice_sites2)

        self.assertEqual(sorted(intersection), [LatticeSite(
            0, [0, 0, 0]), LatticeSite(0, [1, 0, 0])])

    def test_mbnl_non_pbc(self):
        """
        Test many-body neighbor list for non-pbc structure.
        """
        atoms = self.atoms.copy()
        atoms.set_pbc([False])
        structure = Structure.from_atoms(atoms)
        neighbor_lists = get_neighbor_lists(structure, self.cutoffs)

        mbnl = ManyBodyNeighborList()

        target = [([LatticeSite(0, [0, 0, 0])], []),
                  ([LatticeSite(0, [0, 0, 0])],
                   [LatticeSite(1, [0, 0, 0]),
                    LatticeSite(2, [0, 0, 0]),
                    LatticeSite(4, [0, 0, 0]),
                    LatticeSite(5, [0, 0, 0]),
                    LatticeSite(6, [0, 0, 0])]),
                  ([LatticeSite(0, [0, 0, 0]), LatticeSite(1, [0, 0, 0])],
                   [LatticeSite(2, [0, 0, 0]), LatticeSite(4, [0, 0, 0]),
                    LatticeSite(5, [0, 0, 0]), LatticeSite(6, [0, 0, 0])]),
                  ([LatticeSite(0, [0, 0, 0]), LatticeSite(2, [0, 0, 0])],
                   [LatticeSite(6, [0, 0, 0])]),
                  ([LatticeSite(0, [0, 0, 0]), LatticeSite(4, [0, 0, 0])],
                   [LatticeSite(5, [0, 0, 0]), LatticeSite(6, [0, 0, 0])]),
                  ([LatticeSite(0, [0, 0, 0]), LatticeSite(5, [0, 0, 0])],
                   [LatticeSite(6, [0, 0, 0])])]

        neighbors_non_pbc = mbnl.build(neighbor_lists, 0, False)

        for k, latt_neighbors in enumerate(neighbors_non_pbc):
            self.assertEqual(target[k], latt_neighbors)

    def test_mbnl_cubic_non_pbc(self):
        """
        Test that corners sites in a large cubic cell have
        only three neighbors in many-body neighbor list.
        """
        atoms = bulk('Al', 'sc', a=4.0).repeat(4)
        atoms.set_pbc(False)

        structure = Structure.from_atoms(atoms)
        neighbor_lists = get_neighbor_lists(structure, self.cutoffs)

        mbnl = ManyBodyNeighborList()
        # atomic indices located at the corner of structure
        corner_sites = [0, 3, 12, 15, 48, 51, 60, 63]
        for index in corner_sites:
            lattice_neighbor = mbnl.build(neighbor_lists,
                                          index, True)
            # check pairs
            self.assertEqual(len(lattice_neighbor[1][1]), 3)
            # not neighbors besides above pairs
            with self.assertRaises(IndexError):
                lattice_neighbor[2]


if __name__ == '__main__':
    unittest.main()
