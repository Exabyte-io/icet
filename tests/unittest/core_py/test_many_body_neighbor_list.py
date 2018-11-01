import unittest

from ase.build import bulk
from ase.neighborlist import NeighborList
from icet.core.neighbor_list import NeighborList as NeighborList_cpp
from icet.core_py.many_body_neighbor_list import ManyBodyNeighborList
from icet.core_py.lattice_site import LatticeSite
from icet import Structure
from icet.core.many_body_neighbor_list import (
    ManyBodyNeighborList as ManyBodyNeighborList_cpp)


class TestManyBodyNeighborList(unittest.TestCase):
    """
        TODO
        ----
        * compare each latticesite or list of lattice site in each test
          since right now you might not test true equivalence between versions.
    """

    def __init__(self, *args, **kwargs):
        super(TestManyBodyNeighborList, self).__init__(*args, **kwargs)

        self.atoms_prim = bulk("Al")
        self.atoms = bulk("Al").repeat(2)
        self.cutoffs = [5, 5]

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up before each test."""
        self.mbnl = ManyBodyNeighborList(self.atoms, self.cutoffs)
        self.mbnl_cpp = ManyBodyNeighborList_cpp()
        self.neighbor_lists = []
        self.neighbor_lists_cpp = []
        for co in self.cutoffs:
            structure = Structure.from_atoms(self.atoms)
            nl = NeighborList_cpp(co)
            nl.build(structure)
            ase_nl = NeighborList(len(self.atoms) * [co / 2], skin=1e-8,
                                  bothways=True, self_interaction=False)
            ase_nl.update(self.atoms)

            self.neighbor_lists.append(ase_nl)
            self.neighbor_lists_cpp.append(nl)

    def test_build(self):
        """Tests that a simple build works."""
        for index in range(len(self.atoms)):
            mbnl_py = self.mbnl.build(index, bothways=False)
            mbnl_cpp = self.mbnl_cpp.build(
                self.neighbor_lists_cpp, index, False)
            self.assertEqual(len(mbnl_py), len(mbnl_cpp))

    def test_bothways_true(self):
        """
        Build the mbnl with bothways = True and
        assert that each index in the atoms object
        have the same number of neighbors
        """

        mbnl_size = len(self.mbnl.build(0, bothways=True))
        mbnl_size_cpp = len(self.mbnl_cpp.build(
            self.neighbor_lists_cpp, 0, True))

        for index in range(len(self.atoms)):
            self.assertEqual(mbnl_size, len(self.mbnl.build(
                index, bothways=True)))
            self.assertEqual(mbnl_size_cpp, len(self.mbnl_cpp.build(
                self.neighbor_lists_cpp, index,
                True)))

    def test_bothways_false(self):
        """
        Build the mbnl with bothways = False and
        assert that mbnl built on the first
        index in the atoms object do not
        have the same number of neighbors as
        the other atoms.
        """
        mbnl_size = len(self.mbnl.build(
            0, bothways=False))
        mbnl_size_cpp = len(self.mbnl_cpp.build(
            self.neighbor_lists_cpp, 0, False))
        for index in range(1, len(self.atoms)):
            self.assertNotEqual(mbnl_size, len(self.mbnl.build(
                                index,
                                bothways=False)))
            self.assertNotEqual(mbnl_size_cpp, len(self.mbnl_cpp.build(
                                self.neighbor_lists_cpp, index,
                                False)))

        # compare to cpp
        for index in range(len(self.atoms)):
            mbnl_py = self.mbnl.build(
                index, bothways=False)
            mbnl_cpp = self.mbnl_cpp.build(
                self.neighbor_lists_cpp, index, False)
            for lat_site_cpp, lat_site in zip(mbnl_py, mbnl_cpp):
                self.assertEqual(lat_site_cpp[0], lat_site[0])
                self.assertEqual(lat_site_cpp[1], lat_site[1])

    def test_get_intersection(self):
        """Tests intersection functionality."""
        lattice_sites = []
        lattice_sites.append(LatticeSite(0, [0, 0, 0]))
        lattice_sites.append(LatticeSite(0, [1, 0, 0]))
        lattice_sites.append(LatticeSite(1, [0, 0, 0]))
        lattice_sites.append(LatticeSite(3, [0, 0, 0]))

        lattice_sites2 = []
        lattice_sites2.append(LatticeSite(0, [0, 0, 0]))
        lattice_sites2.append(LatticeSite(0, [1, 0, 0]))

        intersection = self.mbnl.get_intersection(
            lattice_sites, lattice_sites2)

        self.assertEqual(sorted(intersection), [LatticeSite(
            0, [0, 0, 0]), LatticeSite(0, [1, 0, 0])])

    def test_translate_all_neighbor(self):
        """Tests translating a list of lattice sites with an offset."""
        lattice_sites = []
        lattice_sites.append(LatticeSite(0, [0, 0, 0]))
        lattice_sites.append(LatticeSite(0, [1, 0, 0]))
        lattice_sites.append(LatticeSite(1, [0, 0, 0]))
        lattice_sites.append(LatticeSite(3, [0, 0, 0]))

        lattice_sites_offset = []
        lattice_sites.append(LatticeSite(0, [1, 2, 3]))
        lattice_sites.append(LatticeSite(0, [2, 2, 3]))
        lattice_sites.append(LatticeSite(1, [1, 2, 3]))
        lattice_sites.append(LatticeSite(3, [1, 2, 3]))

        mbnl_offsets = self.mbnl.translate_all_neighbor(
            lattice_sites_offset, [1, 2, 3])
        self.assertEqual(mbnl_offsets, lattice_sites_offset)

    def test_neighbor_from_smaller(self):
        """Tests filter neighbor from smaller in mbnl."""
        lattice_sites = []
        lattice_sites.append(LatticeSite(0, [0, 0, 0]))
        lattice_sites.append(LatticeSite(0, [1, 0, 0]))
        lattice_sites.append(LatticeSite(1, [0, 0, 0]))
        lattice_sites.append(LatticeSite(3, [-1, 0, 0]))
        # Filter only the first
        filtered_sites = self.mbnl.filter_neighbor_from_smaller(
            lattice_sites, LatticeSite(0, [0, 0, 0]))
        self.assertEqual(len(filtered_sites), 3)

        # Filter everything
        filtered_sites = self.mbnl.filter_neighbor_from_smaller(
            lattice_sites, LatticeSite(5, [0, 0, 0]))
        self.assertEqual(len(filtered_sites), 0)
        # Filter nothing
        filtered_sites = self.mbnl.filter_neighbor_from_smaller(
            lattice_sites, LatticeSite(-1, [10, 10, 10]))
        self.assertEqual(len(filtered_sites), 4)

    def test_get_neighbor_from_nl(self):
        """Tests getting lattice sites from ASE neighbor-list."""
        ase_nl = self.neighbor_lists[0]
        index = 0
        sites_from_nl = self.mbnl.get_neighbor_from_nl(ase_nl, index)
        self.assertEqual(len(sites_from_nl),
                         len(ase_nl.get_neighbors(index)[0]))

    def test_add_singlet(self):
        """Tests adding a singlet."""
        mbn_indices = []
        index = 13
        target = [tuple(([LatticeSite(index, [0., 0., 0.])], []))]
        self.mbnl.add_singlet(index, mbn_indices)
        self.assertEqual(mbn_indices, target)

    def test_add_pairs(self):
        """Tests adding pairs."""
        mbn_indices = []
        index = 0
        target = []
        neighbors = self.mbnl.get_neighbor_from_nl(
            self.neighbor_lists[0], index)
        target = [
            tuple(([LatticeSite(index, [0., 0., 0.])], sorted(neighbors)))]
        self.mbnl.add_pairs(
            index, self.neighbor_lists[0], mbn_indices, bothways=False)
        self.assertEqual(mbn_indices, target)

    def test_unzip(self):
        """
        Tests the unzip functionality.

        sites are in the format:
        list( list(lattice_sites), list_of_lattice_sites)
        call it list(sites1, sites2)
        the unzip will do the following:
        unzipped_sits = []
        for site in sites2
            unzipped_sites.append(sites1+site)
        return unzipped sites

        i.e. if sites1=[ls1, ls2]
        and sites2= [ls3,ls4,ls5]
        unzipped will return
        [[ls1,ls2,ls3],
        [ls1,ls2,ls4],
        [ls1,sl2,ls5]]

        """

        ls = LatticeSite(0, [1, 2, 3])
        # Should get one triplet
        sites = [[ls, ls], [ls]]
        unzipped_sites = self.mbnl.unzip(sites)
        target = [[ls, ls, ls]]
        self.assertEqual(len(unzipped_sites), 1)
        self.assertListEqual(target, unzipped_sites)

        # Should get two triplets
        sites = [[ls, ls], [ls, ls]]
        unzipped_sites = self.mbnl.unzip(sites)
        target = [[ls, ls, ls], [ls, ls, ls]]
        self.assertEqual(len(unzipped_sites), 2)
        self.assertListEqual(target, unzipped_sites)

        ls2 = LatticeSite(1, [3, 2, 1])
        # should get 4 triplets
        sites = [[ls, ls2], [ls, ls, ls2, ls]]
        unzipped_sites = self.mbnl.unzip(sites)
        target = [
            [ls, ls2, ls],
            [ls, ls2, ls],
            [ls, ls2, ls2],
            [ls, ls2, ls]]
        self.assertListEqual(target, unzipped_sites)

        sites = [[ls], []]
        unzipped_sites = self.mbnl.unzip(sites)
        target = [[ls]]
        self.assertEqual(len(unzipped_sites), 1)
        self.assertListEqual(target, unzipped_sites)
        # Tests unzipping entire mbnl
        for index in range(len(self.atoms)):
            mbnl_py = self.mbnl.build(index, bothways=False)
            for compressed_sites in mbnl_py:
                for unzipped in self.mbnl.unzip(compressed_sites):
                    # If singlet len of unzip is the len of sites1
                    if len(compressed_sites[1]) == 0:
                        self.assertEqual(
                            len(unzipped), len(compressed_sites[0]))
                    else:  # else len of unzip is the len of sites1 + 1
                        self.assertEqual(len(unzipped), len(
                            compressed_sites[0]) + 1)


if __name__ == '__main__':
    unittest.main()
