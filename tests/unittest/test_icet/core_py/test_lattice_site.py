from icet.core.lattice_site import LatticeSite as LatticeSite_cpp
from icet.core_py.lattice_site import LatticeSite as LatticeSite
from icet.core_py.lattice_site import (cmp_mbnl_lattice_site_list,
                                       cmp_list_of_lattice_sites)

import numpy as np

import itertools
import unittest


class TestLatticeSite(unittest.TestCase):
    """
    Container for tests of the class functionality.
    This also tests that python LatticeSite behaves the same
    as the C++ LatticeSite.
    """

    def __init__(self, *args, **kwargs):
        super(TestLatticeSite, self).__init__(*args, **kwargs)

        self.indices = [i for i in range(8)]
        self.unitcell_offsets = []
        cartesian_product_lists = [[0., 1.], [0., 1.], [0., 1.]]
        for element in itertools.product(*cartesian_product_lists):
            self.unitcell_offsets.append(list(element))

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup."""
        self.lattice_sites = []
        self.lattice_sites_cpp = []
        for index, unitcell_offset in zip(self.indices, self.unitcell_offsets):
            lattice_site = LatticeSite(index, unitcell_offset)
            lattice_site_cpp = LatticeSite_cpp(index, unitcell_offset)
            self.lattice_sites.append(lattice_site)
            self.lattice_sites_cpp.append(lattice_site_cpp)

    def test_index_property(self):
        """Tests index property."""
        self.assertIsInstance(self.lattice_sites[0].index, int)
        self.assertEqual(self.lattice_sites[0].index, 0)

    def test_offset_property(self):
        """Tests unitcell_offset property."""
        self.assertIsInstance(
            self.lattice_sites[0].unitcell_offset,
            type(self.lattice_sites_cpp[0].unitcell_offset))
        self.assertIsInstance(
            self.lattice_sites[0].unitcell_offset, type(np.array([0])))

        self.assertEqual(
            list(self.lattice_sites[0].unitcell_offset), [0., 0., 0.])

    def test_str(self):
        """Tests printing a LatticeSite."""
        self.assertEqual(self.lattice_sites[0].__str__(
        ), self.lattice_sites_cpp[0].__str__())
        self.assertEqual(self.lattice_sites[-1].__str__(
        ), self.lattice_sites_cpp[-1].__str__())

    def test_sorting(self):
        """Tests sorting the lattice sites."""
        self.lattice_sites.sort()
        self.assertEqual(self.lattice_sites[0].index, 0.)
        self.assertEqual(
            list(self.lattice_sites[0].unitcell_offset), [0., 0., 0.])

        self.lattice_sites.sort(reverse=True)
        self.assertEqual(self.lattice_sites[0].index, 7)
        self.assertEqual(
            list(self.lattice_sites[0].unitcell_offset), [1., 1., 1.])

    def test_lt(self):
        """Tests less than operator."""
        self.assertLess(self.lattice_sites[0], self.lattice_sites[1])
        self.assertLess(self.lattice_sites[0], self.lattice_sites_cpp[1])
        self.assertLess(self.lattice_sites_cpp[0], self.lattice_sites[1])

    def test_eq(self):
        """Tests eq operator."""
        index = 152453453
        unitcell_offset = [-234234., 32423423., 235567567.]

        lattice_site = LatticeSite(index, unitcell_offset)
        lattice_site_cpp = LatticeSite_cpp(index, unitcell_offset)
        lattice_site_other = LatticeSite(index, unitcell_offset)

        self.assertEqual(lattice_site, lattice_site_other)
        self.assertEqual(lattice_site, lattice_site_cpp)

        self.assertNotEqual(lattice_site, self.lattice_sites[0])
        self.assertNotEqual(lattice_site_cpp, self.lattice_sites[0])

        self.assertNotEqual(lattice_site, self.lattice_sites_cpp[0])
        self.assertNotEqual(lattice_site_cpp, self.lattice_sites_cpp[0])

        self.assertNotEqual(
            self.lattice_sites_cpp[1], self.lattice_sites_cpp[0])
        self.assertNotEqual(self.lattice_sites[1], self.lattice_sites[0])
        self.assertNotEqual(self.lattice_sites_cpp[1], self.lattice_sites[0])

    def test_hash(self):
        """Tests hash function."""
        index = 152453453
        unitcell_offset = [-234234., 32423423., 235567567.]

        lattice_site = LatticeSite(index, unitcell_offset)
        lattice_site_other = LatticeSite(index, unitcell_offset)

        self.assertEqual(lattice_site.__hash__(),
                         lattice_site_other.__hash__())

    def test_add_property(self):
        """Tests changing the property."""
        lattice_site = LatticeSite(0, [0, 0, 0])
        lattice_site2 = LatticeSite(0, [-1, -1, 3])
        lattice_site2.unitcell_offset += [1, 1, -3]
        self.assertEqual(lattice_site, lattice_site2)

    def test_sub_property(self):
        """Tests changing the property."""
        lattice_site = LatticeSite_cpp(0, [0, 0, 0])
        lattice_site2 = LatticeSite_cpp(0, [1, 1, -3])
        lattice_site2.unitcell_offset -= [1, 1, -3]
        self.assertEqual(lattice_site, lattice_site2)

        lattice_site = LatticeSite(0, [0, 0, 0])
        lattice_site2 = LatticeSite(0, [1, 1, -3])
        lattice_site2.unitcell_offset -= [1, 1, -3]
        self.assertEqual(lattice_site, lattice_site2)

    def test_cmp_mbnl_lattice_site_list(self):
        """Tests the comparer of list of lattice site."""
        indices = range(10)
        offsets = [[x, y, z]
                   for x, y, z in zip(range(10), range(10), range(10))]
        sites = []
        for index, offset in zip(indices, offsets):
            sites.append(LatticeSite(index, offset))
        sites = [sites, []]
        indices = range(5)
        offsets = [[x, y, z] for x, y, z in zip(range(5), range(5), range(5))]
        sites2 = []
        for index, offset in zip(indices, offsets):
            sites2.append(LatticeSite(index, offset))
        sites2 = [sites2, []]
        # Test that a smaller list is considered smaller
        self.assertTrue(cmp_mbnl_lattice_site_list(sites2, sites))

        # Test that equality is not less than
        self.assertFalse(cmp_mbnl_lattice_site_list(sites, sites))

    def test_cmp_lattice_site_list(self):
        """Tests the comparer of list of lattice site."""
        indices = range(10)
        offsets = [[x, y, z]
                   for x, y, z in zip(range(10), range(10), range(10))]
        sites = []
        for index, offset in zip(indices, offsets):
            sites.append(LatticeSite(index + 1, offset))
        sites = [sites]
        indices = range(10)
        offsets = [[x, y, z] for x, y, z in zip(range(5), range(5), range(5))]
        sites2 = []
        for index, offset in zip(indices, offsets):
            sites2.append(LatticeSite(index, offset))
        sites2 = [sites2]
        # Test that a smaller list is considered smaller
        self.assertTrue(cmp_list_of_lattice_sites(sites2, sites))

        # Test that equality is not less than
        self.assertFalse(cmp_list_of_lattice_sites(sites, sites))


if __name__ == '__main__':
    unittest.main()
