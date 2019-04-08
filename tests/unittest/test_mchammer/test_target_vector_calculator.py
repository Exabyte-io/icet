import unittest

import numpy as np
from ase.build import bulk
from icet import ClusterSpace
from icet import Structure
from mchammer.calculators import TargetVectorCalculator


class TestTVCalculatorBinary(unittest.TestCase):
    """
    Container for tests of the class functionality.
    """

    def __init__(self, *args, **kwargs):
        super(TestTVCalculatorBinary, self).__init__(*args, **kwargs)

        self.prim = bulk('Al', a=4.0)

        atoms = self.prim.repeat((2, 1, 1))
        self.atoms = atoms.repeat(3)
        self.cutoffs = [5, 5]
        self.elements = ['Al', 'Ge']
        self.cs = ClusterSpace(self.prim, self.cutoffs, self.elements)
        self.target_vector = np.linspace(-1, 1, len(self.cs))

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.atoms = self.prim.repeat(4)

        self.calculator = TargetVectorCalculator(
            atoms=self.atoms, cluster_space=self.cs,
            target_vector=self.target_vector,
            name='Tests target vector calc')

    def test_property_cluster_space(self):
        """Tests the cluster expansion property."""
        self.assertIsInstance(
            self.calculator.cluster_space, ClusterSpace)

    def test_calculate_total(self):
        """Test calculate_total function."""
        occupations = []
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                occupations.append(13)
            else:
                occupations.append(32)
        self.assertAlmostEqual(self.calculator.calculate_total(occupations), 7.6363636)


class TestTVCalculatorBinaryHCP(unittest.TestCase):
    """
    Container for tests of the class functionality.
    """

    def __init__(self, *args, **kwargs):
        super(TestTVCalculatorBinaryHCP, self).__init__(*args, **kwargs)

        self.prim = bulk('Al', a=4.0, crystalstructure='hcp')

        atoms = self.prim.repeat((2, 1, 1))
        self.atoms = atoms.repeat(3)
        self.cutoffs = [5, 5]
        self.elements = [['Al', 'Ge'], ['Al', 'Ge']]
        self.cs = ClusterSpace(self.prim, self.cutoffs, self.elements)
        self.target_vector = np.linspace(-1, 1, len(self.cs))

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.atoms = self.prim.repeat(3)

        self.calculator = TargetVectorCalculator(
            atoms=self.atoms, cluster_space=self.cs,
            target_vector=self.target_vector,
            optimality_weight=0,
            name='Tests target vector calc')

    def test_calculate_total(self):
        """Test calculate_total function."""
        occupations = []
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                occupations.append(32)
            else:
                occupations.append(13)
        self.assertAlmostEqual(self.calculator.calculate_total(occupations), 7.0)




class TestTVCalculatorTernary(unittest.TestCase):
    """
    Container for tests of the class functionality.
    """

    def __init__(self, *args, **kwargs):
        super(TestTVCalculatorTernary, self).__init__(*args, **kwargs)

        self.prim = bulk('Al', a=4.0)

        atoms = self.prim.repeat((2, 1, 1))
        self.atoms = atoms.repeat(3)
        self.cutoffs = [5, 5]
        self.elements = ['Al', 'Ge', 'Ga']
        self.cs = ClusterSpace(self.prim, self.cutoffs, self.elements)
        self.target_vector = np.linspace(-1, 1, len(self.cs))

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.atoms = self.prim.repeat(4)

        self.calculator = TargetVectorCalculator(
            atoms=self.atoms, cluster_space=self.cs,
            target_vector=self.target_vector,
            weights=np.linspace(3.1, 1.2, len(self.cs)),
            name='Tests target vector calc')

    def test_property_cluster_space(self):
        """Tests the cluster space property."""
        self.assertIsInstance(
            self.calculator.cluster_space, ClusterSpace)

    def test_calculate_total(self):
        """Test calculate_total function."""
        occupations = []
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                occupations.append(13)
            else:
                if i % 3 == 0:
                    occupations.append(32)
                else:
                    occupations.append(31)
        self.assertAlmostEqual(self.calculator.calculate_total(occupations), 26.9446875)



if __name__ == '__main__':
    unittest.main()
