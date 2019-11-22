import unittest
import numpy as np
from ase.build import bulk
from collections import OrderedDict
from mchammer.data_containers.data_container import DataContainer
from mchammer.observers.base_observer import BaseObserver


class ConcreteObserver(BaseObserver):
    """Child class of BaseObserver created for testing."""
    def __init__(self, interval, tag='ConcreteObserver'):
        super().__init__(interval=interval, return_type=int, tag=tag)

    def get_observable(self, structure):
        """Returns number of Al atoms."""
        return structure.get_chemical_symbols().count('Al')


class TestDataContainer(unittest.TestCase):
    """Container for the tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestDataContainer, self).__init__(*args, **kwargs)
        self.structure = bulk('Al').repeat(2)
        self.ensemble_parameters = {'number_of_atoms': len(self.structure),
                                    'temperature': 375.15}

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test case."""
        self.dc = \
            DataContainer(structure=self.structure,
                          ensemble_parameters=self.ensemble_parameters,
                          metadata=OrderedDict(ensemble_name='test-ensemble',
                                               seed=144))

    def test_init(self):
        """Tests initializing DataContainer."""
        self.assertIsInstance(self.dc, DataContainer)

        # test fails with a non ASE Atoms type
        with self.assertRaises(TypeError) as context:
            DataContainer(structure='structure',
                          ensemble_parameters=self.ensemble_parameters,
                          metadata=OrderedDict(ensemble_name='test-ensemble',
                                               seed=144))

        self.assertTrue('structure is not an ASE Atoms object'
                        in str(context.exception))

    def test_analyze_data(self):
        """Tests analyze_data functionality."""

        # set up a random list of values with a normal distribution
        n_iter, mu, sigma = 100, 1.0, 0.1
        np.random.seed(12)
        for mctrial in range(n_iter):
            row = {'obs1': np.random.normal(mu, sigma), 'obs2': 4.0}
            self.dc.append(mctrial, record=row)

        # check obs1
        summary1 = self.dc.analyze_data('obs1')
        mean1 = self.dc.get_data('obs1').mean()
        std1 = self.dc.get_data('obs1').std()
        self.assertEqual(summary1['mean'], mean1)
        self.assertEqual(summary1['std'], std1)
        self.assertEqual(summary1['correlation_length'], 1)

        # check obs2
        summary2 = self.dc.analyze_data('obs2')
        self.assertTrue(np.isnan(summary2['correlation_length']))

    def test_get_average_and_standard_deviation(self):
        """Tests get average functionality."""
        # set up a random list of values with a normal distribution
        n_iter, mu, sigma = 100, 1.0, 0.1
        np.random.seed(12)
        obs_val = np.random.normal(mu, sigma, n_iter).tolist()

        # append above random data to data container
        for mctrial in range(n_iter):
            self.dc.append(mctrial, record={'obs1': obs_val[mctrial]})

        # get average over all mctrials
        mean = self.dc.get_average('obs1')
        self.assertAlmostEqual(mean, 0.9855693, places=7)

        # get average over slice of data
        mean = self.dc.get_average('obs1', start=60)
        self.assertAlmostEqual(mean, 0.9851106, places=7)

        mean = self.dc.get_average('obs1', stop=60)
        self.assertAlmostEqual(mean, 0.9876534, places=7)

        mean = self.dc.get_average('obs1', start=40, stop=60)
        self.assertAlmostEqual(mean, 1.0137074, places=7)

        # test fails for non-existing data
        with self.assertRaises(ValueError) as context:
            self.dc.get_average('temperature')
        self.assertTrue('No observable named temperature'
                        in str(context.exception))

        # test fails for non-scalar data
        with self.assertRaises(ValueError) as context:
            self.dc.get_average('trajectory')
        self.assertTrue('trajectory is not scalar'
                        in str(context.exception))


if __name__ == '__main__':
    unittest.main()
