import unittest

from ase.build import bulk
from icet import ClusterSpace
from mchammer.observers import ClusterCountObserver


class TestClusterCountObserver(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterCountObserver, self).__init__(*args, **kwargs)

        self.atoms = bulk('Al').repeat([2, 1, 1])
        self.atoms[1].symbol = 'Ge'

        cutoffs = [3]
        subelements = ['Al', 'Ge', 'Si']
        self.cs = ClusterSpace(self.atoms, cutoffs, subelements)
        self.interval = 10

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Set up observer before each test."""
        self.observer = ClusterCountObserver(
            cluster_space=self.cs, atoms=self.atoms, interval=self.interval)

    def test_property_tag(self):
        """Tests property tag."""
        self.assertEqual(self.observer.tag, "ClusterCountObserver")

    def test_property_interval(self):
        """Tests property interval."""
        self.assertEqual(self.observer.interval, self.interval)

    def test_get_observable(self):
        """Tests observable is returned accordingly."""

        prim = bulk('Au')
        structure = prim.repeat(3)
        cutoffs = [3]
        subelements = ['Au', 'Pd']
        cs = ClusterSpace(prim, cutoffs, subelements)
        observer = ClusterCountObserver(
            cluster_space=cs, atoms=structure, interval=self.interval)

        structure.set_chemical_symbols(['Au'] * len(structure))
        # 1 Pd in pure Au sro
        structure[0].symbol = 'Pd'
        counts = observer.get_observable(structure)

        # In total there will be 12 Pd neighboring an Au atom
        expected_Au_Pd_count = 12
        actual_counts = 0
        for count in counts.keys():
            if 'Au' in count and '1' in count and 'Pd' in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Pd_count, actual_counts)

        # Number of Au-Au neighbors should be
        expected_Au_Au_count = 6 * len(structure) - 12
        actual_counts = 0
        for count in counts.keys():
            if 'Au' in count and '1' in count and 'Pd' not in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Au_count, actual_counts)

        # Number of Pd-Pd neighbors should be
        expected_Au_Au_count = 0
        actual_counts = 0
        for count in counts.keys():
            if 'Pd' in count and '1' in count and 'Au' not in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Au_count, actual_counts)

        # 1 Au in Pure Pd sro
        structure.set_chemical_symbols(['Pd'] * len(structure))
        structure[0].symbol = 'Au'
        counts = observer.get_observable(structure)

        # In total there will be 12 Pd neighboring an Au atom
        expected_Au_Pd_count = 12
        actual_counts = 0
        for count in counts.keys():
            if 'Au' in count and '1' in count and 'Pd' in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Pd_count, actual_counts)

        # Number of Au-Au neighbors should be
        expected_Au_Au_count = 0
        actual_counts = 0
        for count in counts.keys():
            if 'Au' in count and '1' in count and 'Pd' not in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Au_count, actual_counts)

        # Number of Pd-Pd neighbors should be
        expected_Au_Au_count = 6 * len(structure) - 12
        actual_counts = 0
        for count in counts.keys():
            if 'Pd' in count and '1' in count and 'Au' not in count:
                actual_counts += counts[count]
        self.assertEqual(expected_Au_Au_count, actual_counts)


if __name__ == '__main__':
    unittest.main()
