

import random
import unittest
import numpy as np

from ase.build import bulk
from ase.neighborlist import NeighborList

from icet.core.lattice_site import LatticeSite
from icet.tools.geometry import find_lattice_site_by_position
from icet.tools.geometry import get_position_from_lattice_site
from icet.tools.geometry import fractional_to_cartesian
from icet.tools.geometry import get_permutation
from icet.tools.geometry import ase_atoms_to_spglib_cell
from icet.tools.geometry import atomic_number_to_chemical_symbol
from icet.tools.geometry import chemical_symbols_to_numbers


class TestGeometry(unittest.TestCase):
    """Container for tests to the geometry module."""

    def __init__(self, *args, **kwargs):
        super(TestGeometry, self).__init__(*args, **kwargs)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Sets up some basic stuff which can be useful in the tests."""
        cutoff = 3.0
        self.structure = bulk('Al')
        self.neighborlist = NeighborList(
            len(self.structure) * [cutoff / 2], skin=1e-8,
            bothways=True, self_interaction=False)
        self.neighborlist.update(self.structure)

    def test_find_lattice_site_by_position_simple(self):
        """
        Tests finding lattice site by position, simple version using
        only one atom cell.

        1. Create a bunch of lattice sites all with index 0 and
        integer unitcell offsets
        2. convert these to x,y,z positions. Nothing strange so far
        3. Find lattice site from the position and assert that it should
           be equivalent to the original lattice site.
        """
        lattice_sites = []
        unit_cell_range = 100
        for j in range(500):
            offset = [random.randint(-unit_cell_range, unit_cell_range)
                      for i in range(3)]
            lattice_sites.append(LatticeSite(0, offset))

        positions = []
        for site in lattice_sites:
            pos = get_position_from_lattice_site(self.structure, site)
            positions.append(pos)
        for site, pos in zip(lattice_sites, positions):
            found_site = find_lattice_site_by_position(self.structure, pos)
            self.assertEqual(site, found_site)

    def test_find_lattice_site_by_position_medium(self):
        """
        Tests finding lattice site by position, medium version
        tests against hcp and user more than one atom in the basis
        1. Create a bunch of lattice sites all with index 0 and
        integer unitcell offsets
        2. convert these to x,y,z positions. Nothing strange so far
        3. Find lattice site from the position and assert that it should
           be equivalent to the original lattice site.
        """
        structure = bulk('Au', 'hcp', a=2.0).repeat([3, 2, 5])
        lattice_sites = []
        unit_cell_range = 100
        for j in range(500):
            offset = [random.randint(-unit_cell_range, unit_cell_range)
                      for i in range(3)]
            index = random.randint(0, len(structure) - 1)
            lattice_sites.append(LatticeSite(index, offset))

        positions = []
        for site in lattice_sites:
            pos = get_position_from_lattice_site(structure, site)
            positions.append(pos)
        for site, pos in zip(lattice_sites, positions):
            found_site = find_lattice_site_by_position(structure, pos)

            self.assertEqual(site, found_site)

    def test_find_lattice_site_by_position_hard(self):
        """
        Tests finding lattice site by position, hard version tests against hcp,
        many structure in the basis AND pbc = [True, True, False] !
        1. Create a bunch of lattice sites all with index 0 and
        integer unitcell offsets
        2. convert these to x,y,z positions. Nothing strange so far
        3. Find lattice site from the position and assert that it should
           be equivalent to the original lattice site.
        """
        structure = bulk('Au', 'hcp', a=2.0).repeat([3, 5, 5])
        # Set pbc false in Z-direction and add vacuum
        structure.pbc = [True, True, False]
        structure.center(30, axis=[2])

        lattice_sites = []
        unit_cell_range = 100
        for j in range(500):
            offset = [random.randint(-unit_cell_range, unit_cell_range)
                      for i in range(3)]
            offset[2] = 0
            index = random.randint(0, len(structure) - 1)
            lattice_sites.append(LatticeSite(index, offset))

        positions = []
        for site in lattice_sites:
            pos = get_position_from_lattice_site(structure, site)
            positions.append(pos)
        for site, pos in zip(lattice_sites, positions):
            found_site = find_lattice_site_by_position(structure, pos)
            self.assertEqual(site, found_site)

    def test_fractional_to_cartesian(self):
        """Tests the geometry function fractional_to_cartesian."""

        # reference data
        atoms = bulk('Al')
        frac_pos = np.array([[ 0.0,  0.0, -0.0],
                             [ 0.0,  0.0,  1.0],
                             [ 0.0,  1.0, -1.0],
                             [ 0.0,  1.0, -0.0],
                             [ 1.0, -1.0, -0.0],
                             [ 1.0,  0.0, -1.0],
                             [ 1.0,  0.0, -0.0],
                             [ 0.0,  0.0, -1.0],
                             [ 0.0, -1.0,  1.0],
                             [ 0.0, -1.0, -0.0],
                             [-1.0,  1.0, -0.0],
                             [-1.0,  0.0,  1.0],
                             [-1.0,  0.0, -0.0]])

        cart_pos_target = [[0., 0., 0.],
                           [2.025, 2.025, 0.],
                           [0., - 2.025, 2.025],
                           [2.025, 0., 2.025],
                           [-2.025, 2.025, 0.],
                           [-2.025, 0., 2.025],
                           [0., 2.025, 2.025],
                           [-2.025, - 2.025, 0.],
                           [0., 2.025, - 2.025],
                           [-2.025, 0., - 2.025],
                           [2.025, - 2.025, 0.],
                           [2.025, 0., - 2.025],
                           [0., - 2.025, - 2.025]]

        # Transform to cartesian
        cart_pos_predicted = []
        for fractional in frac_pos:
            cart_pos_predicted.append(fractional_to_cartesian(atoms, fractional))

        for target, predicted in zip(cart_pos_target, cart_pos_predicted):
            np.testing.assert_almost_equal(target, predicted)

    def test_get_permutation(self):
        """Tests the get_permutation function."""
        value = ['a', 'b', 'c']
        target = ['a', 'b', 'c']
        permutation = [0, 1, 2]
        self.assertEqual(target, get_permutation(value, permutation))

        value = ['a', 'b', 'c']
        target = ['a', 'c', 'b']
        permutation = [0, 2, 1]
        self.assertEqual(target, get_permutation(value, permutation))

        value = [0, 3, 'c']
        target = [3, 'c', 0]
        permutation = [1, 2, 0]
        self.assertEqual(target, get_permutation(value, permutation))

        # Error on permutation list too short
        with self.assertRaises(Exception):
            get_permutation(value, [0, 1])

        # Error on permutation list not unique values
        with self.assertRaises(Exception):
            get_permutation(value, [0, 1, 1])

        # Error on permutation list going out of range
        with self.assertRaises(IndexError):
            get_permutation(value, [0, 1, 3])

    def test_ase_atoms_to_spglib_cell(self):
        """
        Tests that function returns the right tuple from the provided ASE
        Atoms object.
        """
        structure = bulk('Al').repeat(3)
        structure[1].symbol = 'Ag'

        cell, positions, species \
            = ase_atoms_to_spglib_cell(self.structure)

        self.assertTrue((cell == self.structure.get_cell()).all())
        self.assertTrue(
            (positions == self.structure.get_scaled_positions()).all())
        self.assertTrue(
            (species == self.structure.get_atomic_numbers()).all())

    def test_chemical_symbols_to_numbers(self):
        """Tests chemical_symbols_to_numbers method."""

        symbols = ['Al', 'H', 'He']
        expected_numbers = [13, 1, 2]
        retval = chemical_symbols_to_numbers(symbols)
        self.assertEqual(expected_numbers, retval)

    def test_atomic_number_to_chemical_symbol(self):
        """Tests chemical_symbols_to_numbers method."""

        numbers = [13, 1, 2]
        expected_symbols = ['Al', 'H', 'He']
        retval = atomic_number_to_chemical_symbol(numbers)
        self.assertEqual(expected_symbols, retval)


if __name__ == '__main__':
    unittest.main()
