import unittest
from ase.build import bulk
import spglib
import numpy as np
from icet import Structure
from icet.core.neighbor_list import NeighborList
from icet.core.permutation_matrix import (
    PermutationMatrix, permutation_matrix_from_atoms)
from icet.core.permutation_matrix import (
    _get_lattice_site_permutation_matrix as
    get_lattice_site_permutation_matrix)
from icet.core.permutation_matrix import (
    _fractional_to_cartesian as fractional_to_cartesian)
from icet.core.permutation_matrix import (
    _prune_permutation_matrix as prune_permutation_matrix)
from icet.tools.geometry import (
    get_primitive_structure, get_fractional_positions_from_neighbor_list)


class TestPermutationMatrix(unittest.TestCase):
    """Container for test of the module functionality."""

    def __init__(self, *args, **kwargs):
        super(TestPermutationMatrix, self).__init__(*args, **kwargs)

        self.atoms = bulk('Ni', 'hcp', a=3.0).repeat([2, 2, 1])
        self.cutoff = 5.0

        self.atoms_prim = get_primitive_structure(self.atoms)
        prim_structure = Structure.from_atoms(self.atoms_prim)
        neighbor_list = NeighborList(self.cutoff)
        neighbor_list.build(prim_structure)
        self.frac_positions = get_fractional_positions_from_neighbor_list(
            prim_structure, neighbor_list)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        symmetry = spglib.get_symmetry(self.atoms_prim)
        self.translations = symmetry['translations']
        self.rotations = symmetry['rotations']

        self.pm = PermutationMatrix(self.translations,
                                    self.rotations)
        self.pm.build(self.frac_positions)

    def test_init(self):
        """Test initializer."""
        self.assertIsInstance(self.pm, PermutationMatrix)

    def test_dimension_permutation_matrix(self):
        """
        Tests dimensions of permutation matrix. Number of rows should
        be equal to the number of symmetry operations while number of columns
        must correpond to the total number of fractional positions.
        """
        pm_frac = self.pm.get_permuted_positions()
        for row in pm_frac:
            self.assertEqual(len(row), len(self.rotations))
        self.assertEqual(len(pm_frac), len(self.frac_positions))

    def test_get_permuted_positions(self):
        """
        Tests that first row and first column of permutation matrix match
        the target lists.
        """
        pm_frac = self.pm.get_permuted_positions()

        target_row = [[0.0, 0.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [0.0, 0.0, 0.0]]

        retval_row = [pos.tolist() for pos in pm_frac[0]]
        self.assertListEqual(target_row, retval_row)

        target_col = [[0.0, 0.0, 0.0],
                      [-1.0, -1.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0],
                      [-0.6666667, -1.3333333, -0.5],
                      [-0.6666667, -1.3333333, 0.5],
                      [-0.6666667, -0.3333333, -0.5],
                      [-0.6666667, -0.3333333, 0.5],
                      [-0.6666667, 0.6666667, -0.5],
                      [-0.6666667, 0.6666667, 0.5],
                      [0.3333333, -0.3333333, -0.5],
                      [0.3333333, -0.3333333, 0.5],
                      [0.3333333, 0.6666667, -0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [1.3333333, 0.6666667, -0.5],
                      [1.3333333, 0.6666667, 0.5],
                      [0.3333333, 0.6666667, 0.5],
                      [-1.0, 0.0, 0.0],
                      [-1.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 1.0, 1.0],
                      [1.0, 0.0, 0.0],
                      [1.0, 0.0, 1.0],
                      [1.0, 1.0, 0.0],
                      [1.0, 1.0, 1.0],
                      [1.0, 2.0, 0.0],
                      [1.0, 2.0, 1.0],
                      [-0.6666667, -0.3333333, 0.5],
                      [-0.6666667, 0.6666667, 0.5],
                      [0.3333333, -0.3333333, 0.5],
                      [0.3333333, 0.6666667, -0.5],
                      [0.3333333, 0.6666667, 1.5],
                      [0.3333333, 1.6666667, 0.5],
                      [1.3333333, 0.6666667, 0.5],
                      [1.3333333, 1.6666667, 0.5]]

        retval_col = [row[0].tolist() for row in pm_frac]
        self.assertListEqual(target_col, retval_col)

    def test_get_indexed_positions(self):
        """
        Tests that first set of indices along with unique positions in indexed
        positions reproduce correctly the positions retuned in the first row
        of permutation matrix.
        """
        pm_frac = self.pm.get_permuted_positions()
        pm_ind = self.pm.get_indexed_positions()

        indices = pm_ind[0][0]
        repr_positions = pm_ind[-1]
        ind_positions = [repr_positions[ind] for ind in indices]

        for pos, ind_pos in zip(pm_frac[0], ind_positions):
            self.assertEqual(pos.tolist(), ind_pos.tolist())

    def test_permutation_matrix_from_atoms(self):
        """Tests permutation matrix from atoms functionality."""
        pm, _, _ = \
            permutation_matrix_from_atoms(self.atoms, self.cutoff)

        matrix = pm.get_permuted_positions()
        matrix2 = self.pm.get_permuted_positions()

        for row, row2 in zip(matrix, matrix2):
            self.assertEqual(len(row), len(row2))
            for element, element2 in zip(row, row2):
                self.assertEqual(element.tolist(), element2.tolist())

        pm_prim, _, _ = \
            permutation_matrix_from_atoms(
                self.atoms_prim, self.cutoff, find_prim=False)

        matrix_prim = pm_prim.get_permuted_positions()

        for row, row2 in zip(matrix, matrix_prim):
            self.assertEqual(len(row), len(row2))
            for element, element2 in zip(row, row2):
                self.assertEqual(element.tolist(), element2.tolist())

    def test_fractional_to_cartesian(self):
        """
        Tests fractional coordinates are converted into cartesians coordinates.
        """
        target = [[0.0, 0.0, 0.0],
                  [-1.5, -0.87, -2.45],
                  [-1.5, -0.87, -2.45],
                  [0.0, 0.0, 0.0],
                  [-3.0, 0.0, 0.0],
                  [-1.5, -0.87, -2.45],
                  [-1.5, -0.87, -2.45],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [-1.5, -0.87, -2.45],
                  [-1.5, -0.87, -2.45],
                  [-3.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [-1.5, -0.87, -2.45],
                  [-1.5, -0.87, -2.45],
                  [-3.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [-1.5, -0.87, -2.45],
                  [-1.5, -0.87, -2.45],
                  [0.0, 0.0, 0.0],
                  [-3.0, 0.0, 0.0],
                  [-1.5, -0.87, -2.45],
                  [-1.5, -0.87, -2.45],
                  [0.0, 0.0, 0.0]]

        fractional_pos = self.pm.get_permuted_positions()[0]
        cartesian_pos = fractional_to_cartesian(
            fractional_pos, self.atoms_prim.cell)
        retval = np.around(cartesian_pos, decimals=2).tolist()
        self.assertListEqual(retval, target)

    def test_lattice_site_permutation_matrix(self):
        """
        Tests lattice sites in permutation matrix by asserting the distances
        between r_ik and r_jk sites in the same column.
        """
        # TODO: Some part of the implementation cannot be covered as test fails
        # for non-pbc structures.
        atoms = bulk('Al').repeat(2)
        cutoff = 4.2
        pm, prim_structure, _ = \
            permutation_matrix_from_atoms(atoms, cutoff)
        pm_lattice_site = \
            get_lattice_site_permutation_matrix(prim_structure, pm)
        for i in range(len(pm_lattice_site)):
            for j in range(i + 1, len(pm_lattice_site)):
                dist_last = -1
                for k in range(len(pm_lattice_site[i])):
                    site_1 = pm_lattice_site[i][k]
                    site_2 = pm_lattice_site[j][k]
                    pos1 = self.atoms[site_1.index].position +\
                        np.dot(site_1.unitcell_offset, atoms.cell)
                    pos2 = self.atoms[site_2.index].position +\
                        np.dot(site_2.unitcell_offset, atoms.cell)
                    dist_first = np.linalg.norm(pos1 - pos2)
                    if dist_last != -1:
                        self.assertAlmostEqual(dist_first, dist_last, places=8)
                    dist_last = dist_first

    def test_prune_permutation_matrix(self):
        """
        Tests that first column of pruned permutation matrix
        containes unique elements.
        """
        pm, prim_structure, _ = \
            permutation_matrix_from_atoms(self.atoms, self.cutoff)

        pm_lattice_site = \
            get_lattice_site_permutation_matrix(prim_structure, pm)

        pruned_matrix = prune_permutation_matrix(pm_lattice_site)
        first_col = []
        for row in pruned_matrix:
            first_col.append(row[0])
        for i, site_i in enumerate(first_col):
            for j, site_j in enumerate(first_col):
                if i <= j:
                    continue
                else:
                    self.assertNotEqual(site_i, site_j)


if __name__ == '__main__':
    unittest.main()
