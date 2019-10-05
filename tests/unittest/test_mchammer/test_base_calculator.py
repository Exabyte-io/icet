import unittest
from mchammer.calculators.base_calculator import BaseCalculator
from ase.build import bulk
from ase.atoms import Atoms


class ConcreteCalculator(BaseCalculator):
    def __init__(self, structure, name='ConcreteCalc'):
        super().__init__(structure, name=name)

    def calculate_total(self):
        super().calculate_total()

    def calculate_local_contribution(self):
        super().calculate_local_contribution()


class TestBaseCalculator(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestBaseCalculator, self).__init__(*args, **kwargs)
        self.structure = bulk("Al", a=4.0).repeat(3)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ConcreteCalculator(self.structure)

    def test_property_structure(self):
        """Tests property structure."""
        self.assertIsInstance(self.calculator.structure, Atoms)
        self.assertEqual(self.structure, self.calculator.structure)
        print(self.structure.positions, self.calculator.structure.positions)

    def test_calculate_total(self):
        """Tests calculate total."""
        self.calculator.calculate_total()

    def test_calculate_local_contribution(self):
        """Tests calculate local contribution."""
        self.calculator.calculate_local_contribution()

    def test_update_occupations(self):
        """Tests set elements method."""
        structure_cpy = self.structure.copy()

        indices = [0, 1, 3]
        elements = [6, 1, 2]

        # Test first that everything is Al
        self.assertEqual(self.calculator.structure[0].symbol, 'Al')
        self.assertEqual(self.calculator.structure[1].symbol, 'Al')
        self.assertEqual(self.calculator.structure[3].symbol, 'Al')

        self.calculator.update_occupations(indices, elements)
        self.assertEqual(self.calculator.structure[0].symbol, 'C')
        self.assertEqual(self.calculator.structure[1].symbol, 'H')
        self.assertEqual(self.calculator.structure[3].symbol, 'He')

        # test input structure remains unchanged
        self.assertEqual(self.structure, structure_cpy)

        # test that correct exceptions are raised
        with self.assertRaises(TypeError) as context:
            self.calculator.update_occupations(0, 0)
        self.assertTrue('must be of type list'
                        in str(context.exception))
        with self.assertRaises(ValueError) as context:
            self.calculator.update_occupations([0, 1], [0, 1, 2, 3])
        self.assertTrue('must have the same length'
                        in str(context.exception))


if __name__ == '__main__':
    unittest.main()
