import random
import unittest
import numpy as np

from ase import Atoms
from ase.build import bulk
from ase.neighborlist import NeighborList

from icet.core.lattice_site import LatticeSite
from icet.tools.geometry import (get_wyckoff_sites,
                                 ase_atoms_to_spglib_cell,
                                 atomic_number_to_chemical_symbol,
                                 chemical_symbols_to_numbers,
                                 find_lattice_site_by_position,
                                 fractional_to_cartesian,
                                 get_position_from_lattice_site,
                                 get_permutation,
                                 get_primitive_structure,
                                 get_scaled_positions)


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

    def test_get_scaled_positions(self):
        """ Tests the test_get_scaled_positions method. """
        positions = np.array([[6.5, 5.1, 3.0],
                              [-0.1, 1.3, 4.5],
                              [0, 0, 0],
                              [15, 7, 4.5]])
        cell = np.array([[4.0, 1.0, 0.1],
                         [-0.4, 6.7, 0],
                         [-4, 2, 16]])

        retval = get_scaled_positions(positions, cell, wrap=False)
        targetval = np.array([[1.84431247, 0.43339424, 0.17597305],
                              [0.26176336, 0.07149383, 0.27961398],
                              [0, 0, 0],
                              [4.04248515, 0.36500685, 0.25598447]])
        np.testing.assert_almost_equal(retval, targetval)

        retval = get_scaled_positions(positions, cell, wrap=True)
        targetval = np.array([[0.84431247, 0.43339424, 0.17597305],
                              [0.26176336, 0.07149383, 0.27961398],
                              [0, 0, 0],
                              [0.04248515, 0.36500685, 0.25598447]])
        np.testing.assert_almost_equal(retval, targetval)

        retval = get_scaled_positions(positions, cell, wrap=True, pbc=3*[False])
        targetval = np.array([[1.84431247, 0.43339424, 0.17597305],
                              [0.26176336, 0.07149383, 0.27961398],
                              [0, 0, 0],
                              [4.04248515, 0.36500685, 0.25598447]])
        np.testing.assert_almost_equal(retval, targetval)

        retval = get_scaled_positions(positions, cell, wrap=True, pbc=[True, True, False])
        targetval = np.array([[0.84431247, 0.43339424, 0.17597305],
                              [0.26176336, 0.07149383, 0.27961398],
                              [0, 0, 0],
                              [0.04248515, 0.36500685, 0.25598447]])
        np.testing.assert_almost_equal(retval, targetval)

    def test_get_primitive_structure(self) -> None:
        """ Tests the get_primitive_structure method. """
        def compare_structures(s1: Atoms, s2: Atoms) -> bool:
            if len(s1) != len(s2):
                return False
            if not np.all(np.isclose(s1.cell, s2.cell)):
                return False
            if not np.all(np.isclose(s1.get_scaled_positions(), s2.get_scaled_positions())):
                return False
            if not np.all(s1.get_chemical_symbols() == s2.get_chemical_symbols()):
                return False
            return True

        structure = bulk('Al', crystalstructure='fcc', a=4).repeat(2)
        retval = get_primitive_structure(structure, no_idealize=True,
                                         to_primitive=True, symprec=1e-5)
        targetval = Atoms('Al',
                          cell=[[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]],
                          pbc=True,
                          positions=[[0, 0, 0]])
        self.assertTrue(compare_structures(retval, targetval))

        structure = bulk('Al', crystalstructure='fcc', a=4).repeat(2)
        noise_level = 1e-3
        structure[0].position += noise_level * np.array([1, -5, 3])
        structure[3].position += noise_level * np.array([-2, 3, -1])
        retval = get_primitive_structure(structure, no_idealize=True,
                                         to_primitive=True, symprec=1e-5)
        targetval = Atoms(8*'Al',
                          cell=[[-4.0, 0.0, -4.0], [-4.0, 4.0, 0.0], [0.0, 4.0, -4.0]],
                          pbc=True,
                          positions=[[-3.999e+00, 7.995e+00, -3.997e+00],
                                     [-2.000e+00, 2.000e+00, -4.000e+00],
                                     [-2.000e+00, 0.000e+00, -2.000e+00],
                                     [-2.000e-03, 2.003e+00, -2.001e+00],
                                     [-4.000e+00, 2.000e+00, -2.000e+00],
                                     [-2.000e+00, 4.000e+00, -2.000e+00],
                                     [-2.000e+00, 2.000e+00,  0.000e+00],
                                     [-4.000e+00, 4.000e+00, -4.000e+00]])
        self.assertTrue(compare_structures(retval, targetval))

        structure = bulk('Al', crystalstructure='fcc', a=4).repeat(2)
        noise_level = 1e-7
        structure[0].position += noise_level * np.array([1, -5, 3])
        structure[3].position += noise_level * np.array([-2, 3, -1])
        retval = get_primitive_structure(structure, no_idealize=True,
                                         to_primitive=True, symprec=1e-5)
        targetval = Atoms('Al',
                          cell=[[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]],
                          pbc=True,
                          positions=[[-1.24999998e-08, -2.49999997e-08, 2.50000001e-08]])
        self.assertTrue(compare_structures(retval, targetval))

        structure = bulk('SiC', crystalstructure='zincblende', a=4).repeat((2, 2, 1))
        structure.pbc = [True, True, False]
        retval = get_primitive_structure(structure, no_idealize=True,
                                         to_primitive=True, symprec=1e-5)
        targetval = Atoms('SiC',
                          cell=[[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]],
                          pbc=[True, True, False],
                          positions=[[0, 0, 0], [1, 1, 1]])
        self.assertTrue(compare_structures(retval, targetval))

        structure = bulk('SiC', crystalstructure='zincblende', a=4).repeat((2, 2, 1))
        structure.pbc = [True, True, False]
        retval = get_primitive_structure(structure, no_idealize=True,
                                         to_primitive=False, symprec=1e-5)
        targetval = Atoms(4 * 'SiC',
                          cell=[4, 4, 4],
                          pbc=[True, True, False],
                          positions=[[0, 0, 0],
                                     [1, 1, 1],
                                     [0, 2, 2],
                                     [1, 3, 3],
                                     [2, 0, 2],
                                     [3, 1, 3],
                                     [2, 2, 0],
                                     [3, 3, 1]])
        self.assertTrue(compare_structures(retval, targetval))

        structure = bulk('SiC', crystalstructure='zincblende', a=4)
        structure.cell = [[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.00001]]
        retval = get_primitive_structure(structure, no_idealize=False,
                                         to_primitive=True, symprec=1e-3)
        targetval = Atoms('SiC',
                          cell=[[0.0, 1.9999983333374998, 1.9999983333374998],
                                [1.9999983333374998, 0.0, 1.9999983333374998],
                                [1.9999983333374998, 1.9999983333374998, 0.0]],
                          pbc=[True, True, False],
                          positions=[[0, 0, 0], [0.99999917, 0.99999917, 0.99999917]])
        self.assertTrue(compare_structures(retval, targetval))

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

        def compare_lattice_sites(s1: LatticeSite, s2: LatticeSite) -> bool:
            if s1.index != s2.index:
                return False
            if np.any(s1.unitcell_offset != s2.unitcell_offset):
                return False
            return True

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

        # check site at origin
        structure = bulk('Al', crystalstructure='fcc', a=4.0)
        retval = find_lattice_site_by_position(structure, [0, 0, 0])
        targetval = LatticeSite(0, [0, 0, 0])
        self.assertTrue(compare_lattice_sites(retval, targetval))

        # let the method fail
        structure = bulk('Al', crystalstructure='fcc', a=4.0)
        with self.assertRaises(RuntimeError) as e:
            retval = find_lattice_site_by_position(structure, [0, 0, 0.001])
        self.assertTrue('not find' in str(e.exception))

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
        structure = bulk('Al')
        frac_pos = np.array([[0.0,  0.0, -0.0],
                             [0.0,  0.0,  1.0],
                             [0.0,  1.0, -1.0],
                             [0.0,  1.0, -0.0],
                             [1.0, -1.0, -0.0],
                             [1.0,  0.0, -1.0],
                             [1.0,  0.0, -0.0],
                             [0.0,  0.0, -1.0],
                             [0.0, -1.0,  1.0],
                             [0.0, -1.0, -0.0],
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
            cart_pos_predicted.append(fractional_to_cartesian(structure, fractional))

        # Test if predicted cartesian positions are equal to target
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

    def test_get_wyckoff_sites(self):
        """Tests get_wyckoff_sites method."""

        # structures and reference data to test
        structures, targetvals = [], []
        structures.append(bulk('Po', crystalstructure='sc', a=4))
        targetvals.append(['1a'])
        structures.append(bulk('W', crystalstructure='bcc', a=4))
        targetvals.append(['2a'])
        structures.append(bulk('Al', crystalstructure='fcc', a=4))
        targetvals.append(['4a'])
        structures.append(bulk('Ti', crystalstructure='hcp', a=4, c=6))
        targetvals.append(2 * ['2d'])
        structures.append(bulk('SiC', crystalstructure='zincblende', a=4))
        targetvals.append(['4a', '4d'])
        structures.append(bulk('NaCl', crystalstructure='rocksalt', a=4))
        targetvals.append(['4a', '4b'])
        structures.append(bulk('ZnO', crystalstructure='wurtzite', a=4, c=5))
        targetvals.append(4 * ['2b'])

        structures.append(bulk('Al', crystalstructure='fcc', a=4, cubic=True))
        targetvals.append(4 * ['4a'])

        structures.append(bulk('Al').repeat(2))
        targetvals.append(8 * ['4a'])
        structures.append(bulk('Ti').repeat((3, 2, 1)))
        targetvals.append(12 * ['2d'])

        structure = bulk('Al').repeat(2)
        structure[0].position += [0, 0, 0.1]
        structures.append(structure)
        targetvals.append(['2a', '4b', '8c', '8c', '8c', '8c', '4b', '2a'])

        for structure, targetval in zip(structures, targetvals):
            retval = get_wyckoff_sites(structure)
            self.assertEqual(targetval, retval)

        structure = bulk('GaAs', crystalstructure='zincblende', a=3.0).repeat(2)
        structure.set_chemical_symbols(
               ['Ga', 'As', 'Al', 'As', 'Ga', 'As', 'Al', 'As',
                'Ga', 'As', 'Ga', 'As', 'Al', 'As', 'Ga', 'As'])

        retval = get_wyckoff_sites(structure)
        targetval = ['8g', '8i', '4e', '8i', '8g', '8i', '2c', '8i',
                     '2d', '8i', '8g', '8i', '4e', '8i', '8g', '8i']
        self.assertEqual(targetval, retval)

        retval = get_wyckoff_sites(structure, map_occupations=[['Ga', 'Al'], ['As']])
        targetval = 8 * ['4a', '4c']
        self.assertEqual(targetval, retval)

        retval = get_wyckoff_sites(structure, map_occupations=[])
        targetval = len(structure) * ['8a']
        self.assertEqual(targetval, retval)


if __name__ == '__main__':
    unittest.main()
