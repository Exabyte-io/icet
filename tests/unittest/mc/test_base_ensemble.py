import unittest

from icet import ClusterSpace
from icet import ClusterExpansion
from mchammer.ensembles.base_ensemble import BaseEnsemble
from mchammer.observers.base_observer import BaseObserver
from mchammer.calculators.cluster_expansion_calculator import \
    ClusterExpansionCalculator
from ase.build import bulk
import numpy as np


class ParakeetObserver(BaseObserver):
    """Parakeet says 141516."""

    def __init__(self, interval, tag='Parakeet'):
        super().__init__(interval=interval, tag=tag)

    def get_observable(self, atoms):  # noqa
        """Say 141516."""
        return 3.141516

    def return_type(self):
        """ Return float."""
        return float

# Create a concrete child of Ensemble for testing


class ConcreteEnsemble(BaseEnsemble):

    def __init__(self, calculator, name=None, random_seed=None):
        super().__init__(calculator, name=name,
                         random_seed=random_seed)

    def do_trial_step(self):
        pass


class TestEnsemble(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        self.atoms = bulk("Al")
        cutoffs = [5, 5, 4]
        elements = ["Al", "Ga"]
        self.cs = ClusterSpace(self.atoms, cutoffs, elements)
        parameters = np.array([1.2 for _ in range(len(self.cs))])
        self.ce = ClusterExpansion(self.cs, parameters)

    def setUp(self):
        """Setup before each test."""
        calculator = ClusterExpansionCalculator(self.atoms, self.ce)
        self.ensemble = ConcreteEnsemble(
            calculator=calculator, name='test-ensemble', random_seed=42)

        # Create an observer for testing.
        observer = ParakeetObserver(interval=7)
        self.ensemble.attach_observer(observer)
        observer = ParakeetObserver(interval=14, tag='Parakeet2')
        self.ensemble.attach_observer(observer)

    def test_property_name(self):
        """Test name property."""
        self.assertEqual('test-ensemble', self.ensemble.name)

    def test_property_random_seed(self):
        """Test random seed property."""
        self.assertEqual(self.ensemble.random_seed, 42)

    def test_property_accepted_trials(self):
        """Test property accepted trials."""
        self.assertEqual(self.ensemble.accepted_trials, 0)
        self.ensemble.accepted_trials += 1
        self.assertEqual(self.ensemble.accepted_trials, 1)

    def test_property_totals_trials(self):
        """Test property accepted trials."""
        self.assertEqual(self.ensemble.total_trials, 0)
        self.ensemble.total_trials += 1
        self.assertEqual(self.ensemble.total_trials, 1)

    def test_property_calculator(self):
        """Test the calculator property."""
        pass

    def test_get_next_random_number(self):
        """Test the get_next_random_number method."""
        self.assertAlmostEqual(
            self.ensemble.next_random_number(), 0.6394267984578837)

    def test_run(self):
        """Test the run method."""

        n_iters = 50
        self.ensemble.run(n_iters)
        dc_data = self.ensemble.data_container.get_data(tags=['Parakeet2'])
        print(dc_data)
        # plus one since we also count step 0
        number_of_observations = len([x for x in dc_data[0] if x is not None])
        self.assertEqual(
            number_of_observations,
            n_iters // self.ensemble.observers['Parakeet2'].interval + 1)

    def test_internal_run(self):
        """Test the _run method."""
        pass

    def test_property_structure(self):
        """Test the get current structure method."""
        # need calculator for structure
        pass
        # self.assertEqual(self.ensemble.structure, self.atoms)

    def test_attach_observer(self):
        """Test the attach method."""
        self.assertEqual(len(self.ensemble.observers), 2)

        self.ensemble.attach_observer(
            ParakeetObserver(interval=10, tag='test_Parakeet'))
        self.assertEqual(self.ensemble.observers['test_Parakeet'].interval, 10)
        self.assertEqual(
            self.ensemble.observers['test_Parakeet'].tag, 'test_Parakeet')
        self.assertEqual(len(self.ensemble.observers), 3)

        # test no duplicates, this should overwrite the last Parakeet
        self.ensemble.attach_observer(
            ParakeetObserver(interval=15, tag='test_Parakeet'))
        self.assertEqual(len(self.ensemble.observers), 3)
        self.assertEqual(self.ensemble.observers['test_Parakeet'].interval, 15)
        self.assertEqual(
            self.ensemble.observers['test_Parakeet'].tag, 'test_Parakeet')

    def test_property_data_container(self):
        """Test the data container property."""
        pass

    def test_find_minimum_observation_interval(self):
        """Test the method to find the minimum observation interval."""
        pass

    def test_property_minimum_observation_interval(self):
        """Test property minimum observation interval."""
        pass

    def test_get_gcd(self):
        """Test the get gcd method."""
        input = [2, 4, 6, 8]
        target = 2
        self.assertEqual(self.ensemble._get_gcd(input), target)

        input = [20, 15, 10]
        target = 5
        self.assertEqual(self.ensemble._get_gcd(input), target)


if __name__ == '__main__':
    unittest.main()
