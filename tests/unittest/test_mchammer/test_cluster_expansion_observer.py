import unittest

from ase.build import bulk
from icet import ClusterSpace, ClusterExpansion
from mchammer.observers.cluster_expansion_observer import \
    ClusterExpansionObserver
from mchammer.calculators.cluster_expansion_calculator import \
    ClusterExpansionCalculator


class TestCEObserver(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestCEObserver, self).__init__(*args, **kwargs)

        self.atoms = bulk('Al').repeat(3)

        cutoffs = [6, 6, 5]
        subelements = ['Al', 'Ge']
        cs = ClusterSpace(self.atoms, cutoffs, subelements)
        params_len = cs.get_cluster_space_size()
        params = list(range(params_len))

        self.ce = ClusterExpansion(cs, params)
        self.calculator = ClusterExpansionCalculator(self.atoms, self.ce)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up observer before each test."""
        self.observer = ClusterExpansionObserver(
            self.ce, tag='ce_band_gap', interval=10)

    def test_property_tag(self):
        """Tests property tag."""
        self.assertEqual(self.observer.tag, 'ce_band_gap')

    def test_property_interval(self):
        """Tests property interval."""
        self.assertEqual(self.observer.interval, 10)

    def test_get_observable(self):
        """Tests observable is returned accordingly."""
        self.assertEqual(self.observer.get_observable(
            atoms=self.atoms), 283.0)

        # updated occupation using calculator
        indices = [10, 2, 4, 2]
        elements = [32] * 4
        self.calculator.update_occupations(indices, elements)
        self.assertAlmostEqual(self.observer.get_observable(
            atoms=self.calculator.atoms), 1808.0 / 27)


if __name__ == '__main__':
    unittest.main()
