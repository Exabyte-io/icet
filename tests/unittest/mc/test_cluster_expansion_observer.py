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

    def setUp(self):
        """Set up observer before each test."""
        self.observer = ClusterExpansionObserver(
            self.ce, tag='ce_band_gap', interval=10)

        # without interval
        with self.assertRaises(Exception) as context:
            ClusterExpansionObserver(self.ce)

        self.assertTrue("The value of interval must be specified" in
                        str(context.exception))

    def test_property_tag(self):
        """Test property tag."""
        self.assertEqual(self.observer.tag, 'ce_band_gap')

    def test_property_interval(self):
        """Test property interval."""
        self.assertEqual(self.observer.interval, 10)

    def test_get_observable(self):
        """Test observable is returned accordingly."""
        self.assertEqual(self.observer.get_observable(
            atoms=self.atoms), 7641.0)

        # updated occupation using calculator
        indices = [10, 2, 4, 2]
        elements = [32] * 4
        self.calculator.update_occupations(indices, elements)
        self.assertAlmostEqual(self.observer.get_observable(
            atoms=self.calculator.atoms), 1808.0)

    if __name__ == '__main__':
        unittest.main()
