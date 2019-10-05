import unittest

import random
import numpy as np
from ase.build import bulk
from icet import Structure
from icet.core.lattice_site import LatticeSite


def strip_surrounding_spaces(input_string):
    """
    Helper function that removes both leading and trailing spaces from a
    multi-line string.

    Returns
    -------
    str
        original string minus surrounding spaces and empty lines
    """
    from io import StringIO
    s = []
    for line in StringIO(input_string):
        if len(line.strip()) == 0:
            continue
        s += [line.strip()]
    return '\n'.join(s)


class TestStructure(unittest.TestCase):
    """Container for test of the module functionality."""

    def __init__(self, *args, **kwargs):
        super(TestStructure, self).__init__(*args, **kwargs)
        self.ase_atoms = bulk('Ag', 'hcp', a=2.0)
        self.noise = 1e-6
        self.positions = [[0., 0., 0.],
                          [0., 1.15470054, 1.63299316]]
        self.chemical_symbols = ['Ag', 'Ag']
        self.cell = [[2., 0., 0.],
                     [-1., 1.73205081, 0.],
                     [0., 0., 3.26598632]]

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.icet_structure = Structure(
            positions=self.positions,
            chemical_symbols=self.chemical_symbols,
            cell=self.cell,
            pbc=[True, True, True])
        random.seed(113)

    def test_positions(self):
        """Tests positions of atoms in structure."""
        for i, vec in enumerate(self.icet_structure.positions):
            self.assertListEqual(vec.tolist(), self.positions[i])

    def test_chemical_symbols(self):
        """Tests chemical symbols of atoms in structure."""
        self.assertListEqual(self.icet_structure.chemical_symbols,
                             self.chemical_symbols)

    def test_atomic_numbers(self):
        """Tests atomic numbers."""
        self.assertListEqual(self.icet_structure.atomic_numbers,
                             [47, 47])

    def test_cell(self):
        """Tests cell."""
        for i, vec in enumerate(self.icet_structure.cell):
            self.assertListEqual(vec.tolist(), self.cell[i])

    def test_pbc(self):
        """Tests periodic boundary conditions."""
        self.assertListEqual(self.icet_structure.pbc, [True, True, True])

    def test_unique_sites(self):
        """Tests unique sites."""
        self.assertListEqual(self.icet_structure.unique_sites,
                             [0, 0])

    def test_set_and_get_positions(self):
        """Tests set and get positions."""
        new_positions = [[0., 0., 0.001],
                         [0., 1.15470054, 1.63299316]]
        self.icet_structure.set_positions(new_positions)
        retval = self.icet_structure.get_positions()
        for i, vec in enumerate(retval):
            self.assertListEqual(vec.tolist(), new_positions[i])

    def test_set_and_get_chemical_symbols(self):
        """Tests set and get chemical symbols."""
        new_chemical_symbols = ['Au', 'Au']
        self.icet_structure.set_chemical_symbols(new_chemical_symbols)
        retval = self.icet_structure.get_chemical_symbols()
        self.assertListEqual(retval, new_chemical_symbols)

    def test_set_and_get_atomic_numbers(self):
        """Tests set and get atomic numbers."""
        self.icet_structure.set_atomic_numbers([48, 47])
        retval = self.icet_structure.get_atomic_numbers()
        self.assertListEqual(retval, [48, 47])

    def test_set_and_get_cell(self):
        """Tests set and get cell."""
        new_cell = [[2., 0., 0.],
                    [-1., 2., 0.],
                    [0., 0., 4.]]
        self.icet_structure.set_cell(new_cell)
        retval = self.icet_structure.get_cell()
        for i, vec in enumerate(retval):
            self.assertListEqual(vec.tolist(), new_cell[i])

    def test_set_and_get_pbc(self):
        """Tests set and get pbc."""
        self.icet_structure.set_pbc([True, True, False])
        retval = self.icet_structure.get_pbc()
        self.assertListEqual(retval, [True, True, False])

    def test_set_and_get_unique_sites(self):
        """Tests set and get unique sites."""
        self.icet_structure.set_unique_sites([0, 1])
        retval = self.icet_structure.get_unique_sites()
        self.assertListEqual(retval, [0, 1])

    def test_get_position(self):
        """Tests get_position functionality."""
        retval = self.icet_structure.get_position(LatticeSite(1, [0, 0, 0]))
        self.assertListEqual(retval.tolist(), self.positions[1])

    def test_get_distance(self):
        """Tests get_distance functionality."""
        retval = self.icet_structure.get_distance(0, 1,
                                                  [0., 0., 0.],
                                                  [0., 0., 0.])

        target = self.icet_structure.get_distance(0, 1)
        self.assertAlmostEqual(retval, target)

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
        noise_position = []
        unit_cell_range = 1000
        for j in range(5000):
            offset = [random.randint(-unit_cell_range, unit_cell_range)
                      for i in range(3)]
            noise_position.append(
                [self.noise * random.uniform(-1, 1) for i in range(3)])
            lattice_sites.append(LatticeSite(0, offset))

        positions = []
        for i, site in enumerate(lattice_sites):
            # Get position with a little noise
            pos = self.icet_structure.get_position(site)
            pos = pos + np.array(noise_position[i])
            positions.append(pos)
        for site, pos in zip(lattice_sites, positions):
            found_site = self.icet_structure.find_lattice_site_by_position(pos)
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
        ase_atoms = self.ase_atoms.repeat([3, 2, 5])

        icet_structure = Structure.from_atoms(ase_atoms)
        lattice_sites = []
        unit_cell_range = 1000
        noise_position = []

        for j in range(5000):
            offset = [random.randint(-unit_cell_range, unit_cell_range)
                      for i in range(3)]
            index = random.randint(0, len(ase_atoms) - 1)
            noise_position.append(
                [self.noise * random.uniform(-1, 1) for i in range(3)])
            lattice_sites.append(LatticeSite(index, offset))

        positions = []
        for i, site in enumerate(lattice_sites):
            pos = icet_structure.get_position(site)
            pos = pos + np.array(noise_position[i])
            positions.append(pos)
        for site, pos in zip(lattice_sites, positions):
            found_site = icet_structure.find_lattice_site_by_position(pos)

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
        ase_atoms = self.ase_atoms.repeat([3, 5, 5])

        # Set pbc false in Z-direction and add vacuum
        ase_atoms.pbc = [True, True, False]
        ase_atoms.center(30, axis=[2])
        icet_structure = Structure.from_atoms(ase_atoms)
        noise_position = []

        lattice_sites = []
        unit_cell_range = 100
        for j in range(500):
            offset = [random.randint(-unit_cell_range, unit_cell_range)
                      for i in range(3)]
            offset[2] = 0
            index = random.randint(0, len(ase_atoms) - 1)
            noise_position.append(
                [self.noise * random.uniform(-1, 1) for i in range(3)])

            lattice_sites.append(LatticeSite(index, offset))

        positions = []
        for i, site in enumerate(lattice_sites):
            pos = icet_structure.get_position(site)
            pos += np.array(noise_position[i])
            positions.append(pos)
        for site, pos in zip(lattice_sites, positions):
            found_site = icet_structure.find_lattice_site_by_position(pos)
            self.assertEqual(site, found_site)

    def test_structure_from_atoms(self):
        """Tests ASE Atoms-to-icet Structure conversion."""
        icet_structure = Structure.from_atoms(self.ase_atoms)

        icet_positions = icet_structure.get_positions()
        ase_positions = self.ase_atoms.get_positions()
        for icet_pos, ase_pos in zip(icet_positions, ase_positions):
            self.assertListEqual(icet_pos.tolist(), ase_pos.tolist())

        chem_symbols = icet_structure.get_chemical_symbols()
        self.assertListEqual(chem_symbols, ['Ag', 'Ag'])

    def test_structure_to_atoms(self):
        """Tests icet Structure-to-ASE Atoms conversion."""
        ase_structure = Structure.to_atoms(self.icet_structure)
        ase_positions = ase_structure.get_positions()
        icet_positions = self.icet_structure.get_positions()
        for ase_pos, icet_pos in zip(ase_positions, icet_positions):
            self.assertListEqual(ase_pos.tolist(), icet_pos.tolist())

        chem_symbols = ase_structure.get_chemical_symbols()
        self.assertListEqual(chem_symbols, ['Ag', 'Ag'])

    def test_repr_function(self):
        """Tests string representation."""
        retval = self.icet_structure.__repr__()
        target = """
Cell:
[[ 2.          0.          0.        ]
 [-1.          1.73205081  0.        ]
 [ 0.          0.          3.26598632]]

Element and positions:
Ag  [0.0  0.0  0.0]
Ag  [0.0  1.15470054  1.63299316]
"""
        self.assertEqual(strip_surrounding_spaces(retval),
                         strip_surrounding_spaces(target))


if __name__ == '__main__':
    unittest.main()
