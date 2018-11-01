from icet.core_py.permutation_matrix import PermutationMatrix
from icet.core.permutation_matrix import permutation_matrix_from_atoms
from icet.core.permutation_matrix import \
    PermutationMatrix as PermutationMatrix_cpp
from ase.build import bulk
from icet.core.permutation_matrix import _get_lattice_site_permutation_matrix\
    as get_lattice_site_permutation_matrix

import unittest

import numpy as np


class TestPermutationMatrix(unittest.TestCase):
    """Tests the python implementation of permutation matrix."""

    def __init__(self, *args, **kwargs):
        super(TestPermutationMatrix, self).__init__(*args, **kwargs)

        self.atoms = bulk("Al").repeat(2)
        self.cutoff = 5.0

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up class before each test case"""
        self.pm = PermutationMatrix(self.atoms, self.cutoff)
        self.pm_cpp, self.prim_structure_cpp, _ = \
            permutation_matrix_from_atoms(
                self.atoms, self.cutoff)
        self.pm_lattice_sites_cpp = \
            get_lattice_site_permutation_matrix(self.prim_structure_cpp,
                                                self.pm_cpp,
                                                prune=True)

    def test_init(self):
        """Tests the initializer."""
        pm = PermutationMatrix(self.atoms, self.cutoff)
        # Tests pm with find_prim = False
        pm_prim_false = PermutationMatrix(self.atoms, self.cutoff,
                                          find_prim=False)
        pm_cpp, _, _ = permutation_matrix_from_atoms(
            self.atoms, self.cutoff)
        self.assertIsInstance(pm, PermutationMatrix)
        self.assertIsInstance(pm_prim_false, PermutationMatrix)
        self.assertIsInstance(pm_cpp, PermutationMatrix_cpp)

    def test_equal_first_column(self):
        """Tests that the first column are the same for both matrices."""
        matrix_py = self.pm.pm_lattice_sites
        matrix_cpp = sorted(self.pm_lattice_sites_cpp)
        col1_py = [row[0] for row in matrix_py]
        col1_cpp = [row[0] for row in matrix_cpp]

        # Tests length of column 1
        self.assertEqual(len(col1_py), 43)

        # check that there are no duplicates
        self.assertEqual(len(col1_py), len(set(col1_py)))
        self.assertEqual(len(col1_cpp), len(set(col1_cpp)))

        for lattice_site_py, lattice_site_cpp in zip((col1_py),
                                                     (col1_cpp)):
            self.assertEqual(lattice_site_py, lattice_site_cpp)
        self.assertListEqual((col1_py), (col1_cpp))

    def test_equal_rows(self):
        """Tests row lengths"""
        matrix_py = self.pm.pm_lattice_sites
        matrix_cpp = sorted(self.pm_lattice_sites_cpp)

        self.assertEqual(len(matrix_py[0]), 48)
        self.assertEqual(len(matrix_cpp[0]), 48)

        for i in range(len(matrix_py)):
            self.assertEqual(len(matrix_py[i]), 48)
            self.assertEqual(len(matrix_cpp[i]), 48)

    def test_all_matrix_elements_equal(self):
        """
        Tests that all elements of both Lattice site permutation
        matrices are identical.
        """
        pm_cpp = sorted(self.pm_lattice_sites_cpp)
        for row_py, row_cpp in zip(self.pm.pm_lattice_sites, pm_cpp):
            self.assertEqual(len(row_py), len(row_cpp))
            for element_py, element_cpp in zip(row_py, row_cpp):
                self.assertEqual(element_py, element_cpp)

    def test_equal_dist_for_same_row(self):
        """
        Tests that distance between site_ij and site_ik
        are the same for all j,k.

        This is only done for the python version
        since the C++ version is tested to be equivalent
        """

        for i in range(len(self.pm.pm_lattice_sites)):
            for j in range(i + 1, len(self.pm.pm_lattice_sites)):
                dist_last = -1
                for k in range(len(self.pm.pm_lattice_sites[i])):
                    site_1 = self.pm.pm_lattice_sites[i][k]
                    site_2 = self.pm.pm_lattice_sites[j][k]
                    pos1 = self.atoms[site_1.index].position +\
                        np.dot(site_1.unitcell_offset, self.atoms.cell)
                    pos2 = self.atoms[site_2.index].position +\
                        np.dot(site_2.unitcell_offset, self.atoms.cell)
                    dist_first = np.linalg.norm(pos1 - pos2)
                    if dist_last != -1:
                        self.assertAlmostEqual(dist_first, dist_last, places=8)
                    dist_last = dist_first

    def test_primitive_structure(self):
        """
        Tests primitive structure property by comparing to
        the other versions primitive structure.
        """
        for position_py, position_cpp in zip(
                self.pm.primitive_structure.positions,
                self.prim_structure_cpp.positions):
            self.assertAlmostEqual(np.linalg.norm(
                position_cpp - position_py), 0)

        self.assertEqual(self.pm.primitive_structure.cell.all(),
                         self.prim_structure_cpp.cell.all())

    def test_property_column1(self):
        """Tests the column1 property of permutation matrix."""

        matrix = self.pm.pm_lattice_sites
        for col1, row in zip(self.pm.column1, matrix):
            self.assertEqual(col1, row[0])


if __name__ == '__main__':
    unittest.main()
