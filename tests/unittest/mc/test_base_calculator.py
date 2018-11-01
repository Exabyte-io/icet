import unittest
from mchammer.calculators.base_calculator import BaseCalculator
from ase.build import bulk
from ase.atoms import Atoms


class TestBaseCalculator(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestBaseCalculator, self).__init__(*args, **kwargs)
        self.atoms = bulk("Al").repeat(3)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        class ConcreteCalculator(BaseCalculator):
            def __init__(self, atoms, name='ConcreteCalc'):
                super().__init__(atoms, name=name)

            def calculate_total(self):
                super().calculate_total()

            def calculate_local_contribution(self):
                super().calculate_local_contribution()

        self.calculator = ConcreteCalculator(self.atoms)

    def test_property_atoms(self):
        """Tests property atoms."""
        self.assertIsInstance(self.calculator.atoms, Atoms)
        self.assertEqual(self.atoms, self.calculator.atoms)

    def test_calculate_total(self):
        """Tests calculate total."""
        self.calculator.calculate_total()

    def test_calculate_local_contribution(self):
        """Tests calculate local contribution."""
        self.calculator.calculate_local_contribution()

    def test_update_occupations(self):
        """Tests set elements method."""
        indices = [0, 1, 3]
        elements = [6, 1, 2]

        # Test first that everything is Al
        self.assertEqual(self.atoms[0].symbol, 'Al')
        self.assertEqual(self.atoms[1].symbol, 'Al')
        self.assertEqual(self.atoms[3].symbol, 'Al')

        self.calculator.update_occupations(indices, elements)
        self.assertEqual(self.atoms[0].symbol, 'C')
        self.assertEqual(self.atoms[1].symbol, 'H')
        self.assertEqual(self.atoms[3].symbol, 'He')

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
