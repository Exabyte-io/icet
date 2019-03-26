import unittest

from ase.build import bulk
from icet.core.sublattices import Sublattices, Sublattice


class TestSublattice(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestSublattice,
              self).__init__(*args, **kwargs)

        self.chemical_symbols = ['Al', 'Ge', 'Si']
        self.indices = [1, 2, 3, 4, 5, 6, 7]

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up sublattice before each test."""
        self.sublattice = Sublattice(
            chemical_symbols=self.chemical_symbols, indices=self.indices)

    def test_indices(self):
        """Tests indices property."""

        self.assertEqual(self.sublattice.indices, [1, 2, 3, 4, 5, 6, 7])

        # Test that indices are copied
        indices = self.sublattice.indices
        indices[0] = -99
        indices.append(37)
        self.assertEqual(self.sublattice.indices, [1, 2, 3, 4, 5, 6, 7])

    def test_chemical_symbols(self):
        """Tests chemical symbols property."""
        self.assertEqual(self.sublattice.chemical_symbols, ['Al', 'Ge', 'Si'])

        # Test that symbols are copied
        symbols = self.sublattice.chemical_symbols
        symbols[0] = 'H'
        symbols.append('Pt')
        self.assertEqual(self.sublattice.chemical_symbols, ['Al', 'Ge', 'Si'])


class TestSublattices(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestSublattices,
              self).__init__(*args, **kwargs)

        self.prim = bulk('Au').repeat([2, 1, 1])
        self.prim[1].symbol = 'H'
        self.allowed_species = [('Pd', 'Au'), ('H', 'V')]

        self.supercell = self.prim.repeat(3)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up sublattices before each test."""
        self.sublattices = Sublattices(allowed_species=self.allowed_species,
                                       primitive_structure=self.prim, structure=self.supercell)

    def test_allowed_species(self):
        """Tests the allowed species property."""
        # Note that the Au, Pd order has changed due to lexicographically sorted symbols
        self.assertEqual(self.sublattices.allowed_species,
                         [('Au', 'Pd'), ('H', 'V')])

    def test_get_sublattice_sites(self):
        """Tests the get sublattice sites method."""

        self.assertEqual(self.sublattices.get_sublattice_sites(
            index=0), self.sublattices[0].indices)

        self.assertEqual(self.sublattices[0].indices, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
                                                       24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44,
                                                       46, 48, 50, 52])

        self.assertEqual(self.sublattices[1].indices, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21,
                                                       23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
                                                       43, 45, 47, 49, 51, 53])

    def test_get_sublattice_index(self):
        """Tests the get sublattice index method."""

        for i in range(len(self.supercell)):
            sublattice_index = self.sublattices.get_sublattice_index(index=i)
            if i % 2 == 0:
                self.assertEqual(sublattice_index, 0)
            else:
                self.assertEqual(sublattice_index, 1)


if __name__ == '__main__':
    unittest.main()
