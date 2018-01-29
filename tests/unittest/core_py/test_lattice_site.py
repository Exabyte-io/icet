from icet.core.lattice_site import LatticeSite as LatticeSite_cpp
from icet.core_py.lattice_site import LatticeSite as LatticeSite


import numpy as np

import itertools
import unittest


class TestLatticeSite(unittest.TestCase):
    '''
    Container for tests of the class functionality.
    This also tests that python LatticeSite behaves the same
    as the C++ LatticeSite.
    '''

    def __init__(self, *args, **kwargs):
        super(TestLatticeSite, self).__init__(*args, **kwargs)

        self.indices = [i for i in range(8)]
        self.unitcell_offsets = []
        cartesian_product_lists = [[0., 1.], [0., 1.], [0., 1.]]
        for element in itertools.product(*cartesian_product_lists):
            self.unitcell_offsets.append(list(element))

    def setUp(self):
        """
        Setup.
        """
        self.lattice_sites = []
        self.lattice_sites_cpp = []
        for index, unitcell_offset in zip(self.indices, self.unitcell_offsets):
            lattice_site = LatticeSite(index, unitcell_offset)
            lattice_site_cpp = LatticeSite_cpp(index, unitcell_offset)
            self.lattice_sites.append(lattice_site)
            self.lattice_sites_cpp.append(lattice_site_cpp)

    def test_index_property(self):
        """
        Test index property.
        """

        self.assertIsInstance(self.lattice_sites[0].index, int)
        self.assertIsInstance(self.lattice_sites_cpp[0].index, int)

        self.assertEqual(self.lattice_sites[0].index, 0)
        self.assertEqual(self.lattice_sites_cpp[0].index, 0)

    def test_offset_property(self):
        """
        Test unitcell_offset property.
        """
        self.assertIsInstance(
            self.lattice_sites[0].unitcell_offset,
            type(self.lattice_sites_cpp[0].unitcell_offset))
        self.assertIsInstance(
            self.lattice_sites[0].unitcell_offset, type(np.array([0])))

        self.assertEqual(
            list(self.lattice_sites[0].unitcell_offset), [0., 0., 0.])
        self.assertEqual(
            list(self.lattice_sites_cpp[0].unitcell_offset), [0., 0., 0.])

    def test_str(self):
        """
        Test printing a LatticeSite.
        """
        print(self.lattice_sites[0])
        print(self.lattice_sites_cpp[0])

        self.assertEqual(self.lattice_sites[0].__str__(
        ), self.lattice_sites_cpp[0].__str__())
        self.assertEqual(self.lattice_sites[-1].__str__(
        ), self.lattice_sites_cpp[-1].__str__())

    def test_sorting(self):
        """
        Test sorting the lattice sites.
        """
        self.lattice_sites.sort()
        self.assertEqual(self.lattice_sites[0].index, 0.)
        self.assertEqual(
            list(self.lattice_sites[0].unitcell_offset), [0., 0., 0.])

        self.lattice_sites.sort(reverse=True)
        self.assertEqual(self.lattice_sites[0].index, 7)
        self.assertEqual(
            list(self.lattice_sites[0].unitcell_offset), [1., 1., 1.])

    def test_lt(self):
        """
        Test less than operator.
        """
        self.assertLess(self.lattice_sites[0], self.lattice_sites[1])
        self.assertLess(self.lattice_sites_cpp[0], self.lattice_sites_cpp[1])
        self.assertLess(self.lattice_sites[0], self.lattice_sites_cpp[1])
        self.assertLess(self.lattice_sites_cpp[0], self.lattice_sites[1])

    def test_eq(self):
        """
        Test eq operator.
        """
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
        """
        Test hash function.
        """

        index = 152453453
        unitcell_offset = [-234234., 32423423., 235567567.]

        lattice_site = LatticeSite(index, unitcell_offset)
        lattice_site_cpp = LatticeSite_cpp(index, unitcell_offset)

        lattice_site_other = LatticeSite(index, unitcell_offset)
        lattice_site_other_cpp = LatticeSite_cpp(index, unitcell_offset)

        self.assertEqual(lattice_site.__hash__(),
                         lattice_site_other.__hash__())
        self.assertEqual(lattice_site_cpp.__hash__(),
                         lattice_site_other_cpp.__hash__())

    def test_add_property(self):
        """
        Tests changing the property.
        """
        lattice_site = LatticeSite_cpp(0, [0, 0, 0])
        lattice_site2 = LatticeSite_cpp(0, [-1, -1, 3])
        lattice_site2.unitcell_offset += [1, 1, -3]
        self.assertEqual(lattice_site, lattice_site2)

        lattice_site = LatticeSite(0, [0, 0, 0])
        lattice_site2 = LatticeSite(0, [-1, -1, 3])
        lattice_site2.unitcell_offset += [1, 1, -3]
        self.assertEqual(lattice_site, lattice_site2)


if __name__ == '__main__':
    unittest.main()
