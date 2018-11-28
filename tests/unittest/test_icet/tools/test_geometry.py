

import random
import unittest

from ase.build import bulk
from ase.neighborlist import NeighborList

from icet.core_py.lattice_site import LatticeSite as LatticeSite_py
from icet.tools.geometry import get_fractional_positions_from_ase_neighbor_list
from icet.tools.geometry import find_lattice_site_by_position
from icet.tools.geometry import get_position_from_lattice_site
from icet.tools.geometry import fractional_to_cartesian
from icet.tools.geometry import get_permutation
from icet.tools.geometry import ase_atoms_to_spglib_cell


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
        self.atoms = bulk('Al')
        self.neighborlist = NeighborList(
            len(self.atoms) * [cutoff / 2], skin=1e-8,
            bothways=True, self_interaction=False)
        self.neighborlist.update(self.atoms)

    def test_get_frac_pos_from_ase_neighborlist(self):
        """Tests the get fractional position from ase neighborlist."""

        # This system is simple so all neighbors are integer
        #  fractional positions
        frac_pos = get_fractional_positions_from_ase_neighbor_list(
            self.atoms, self.neighborlist)

        target = [[0., 0., - 0.],
                  [0., 0., 1.],
                  [0., 1., - 1.],
                  [0., 1., - 0.],
                  [1., - 1., - 0.],
                  [1., 0., - 1.],
                  [1., 0., - 0.],
                  [0., 0., - 1.],
                  [0., - 1., 1.],
                  [0., - 1., - 0.],
                  [-1., 1., - 0.],
                  [-1., 0., 1.],
                  [-1., 0., - 0.]]

        for pos1, pos2 in zip(frac_pos, target):
            self.assertListEqual(list(pos1), pos2)

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
            lattice_sites.append(LatticeSite_py(0, offset))

        positions = []
        for site in lattice_sites:
            pos = get_position_from_lattice_site(self.atoms, site)
            positions.append(pos)
        for site, pos in zip(lattice_sites, positions):
            found_site = find_lattice_site_by_position(self.atoms, pos)
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
        atoms = bulk('Au', 'hcp', a=2.0).repeat([3, 2, 5])
        lattice_sites = []
        unit_cell_range = 100
        for j in range(500):
            offset = [random.randint(-unit_cell_range, unit_cell_range)
                      for i in range(3)]
            index = random.randint(0, len(atoms) - 1)
            lattice_sites.append(LatticeSite_py(index, offset))

        positions = []
        for site in lattice_sites:
            pos = get_position_from_lattice_site(atoms, site)
            positions.append(pos)
        for site, pos in zip(lattice_sites, positions):
            found_site = find_lattice_site_by_position(atoms, pos)

            self.assertEqual(site, found_site)

    def test_find_lattice_site_by_position_hard(self):
        """
        Tests finding lattice site by position, hard version tests against hcp,
        many atoms in the basis AND pbc = [True, True, False] !
        1. Create a bunch of lattice sites all with index 0 and
        integer unitcell offsets
        2. convert these to x,y,z positions. Nothing strange so far
        3. Find lattice site from the position and assert that it should
           be equivalent to the original lattice site.
        """
        atoms = bulk('Au', 'hcp', a=2.0).repeat([3, 5, 5])
        # Set pbc false in Z-direction and add vacuum
        atoms.pbc = [True, True, False]
        atoms.center(30, axis=[2])

        lattice_sites = []
        unit_cell_range = 100
        for j in range(500):
            offset = [random.randint(-unit_cell_range, unit_cell_range)
                      for i in range(3)]
            offset[2] = 0
            index = random.randint(0, len(atoms) - 1)
            lattice_sites.append(LatticeSite_py(index, offset))

        positions = []
        for site in lattice_sites:
            pos = get_position_from_lattice_site(atoms, site)
            positions.append(pos)
        for site, pos in zip(lattice_sites, positions):
            found_site = find_lattice_site_by_position(atoms, pos)
            self.assertEqual(site, found_site)

    def test_fractional_to_cartesian(self):
        """Tests the geometry function fractional_to_cartesian."""
        # Use the get frac positions from
        #  neighborlist to have something to work with

        frac_pos = get_fractional_positions_from_ase_neighbor_list(
            self.atoms, self.neighborlist)

        # Transform to cartesian
        positions = []
        for fractional in frac_pos:
            positions.append(fractional_to_cartesian(self.atoms, fractional))

        target_pos = [[0., 0., 0.],
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

        for target, pos in zip(target_pos, positions):
            self.assertEqual(target, list(pos))

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
        atoms = bulk('Al').repeat(3)
        atoms[1].symbol = 'Ag'

        cell, positions, species \
            = ase_atoms_to_spglib_cell(self.atoms)

        self.assertTrue((cell == self.atoms.get_cell()).all())
        self.assertTrue(
            (positions == self.atoms.get_scaled_positions()).all())
        self.assertTrue(
            (species == self.atoms.get_atomic_numbers()).all())


if __name__ == '__main__':
    unittest.main()
