import unittest
import tempfile
import random
import numpy as np
import pandas as pd
from ase.build import bulk
from collections import OrderedDict
from mchammer import DataContainer
from mchammer.observers.base_observer import BaseObserver


class ConcreteObserver(BaseObserver):
    """Child class of BaseObserver created for testing."""
    def __init__(self, interval, tag='ConcreteObserver'):
        super().__init__(interval, return_type=int, tag=tag)

    def get_observable(self, atoms):
        """Returns number of Al atoms."""
        return atoms.get_chemical_symbols().count('Al')


class TestDataContainer(unittest.TestCase):
    """Container for the tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestDataContainer, self).__init__(*args, **kwargs)
        self.atoms = bulk('Al').repeat(2)
        self.ensemble_parameters = {'number_of_atoms': len(self.atoms),
                                    'temperature': 375.15}

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test case."""
        self.dc = \
            DataContainer(atoms=self.atoms,
                          ensemble_parameters=self.ensemble_parameters,
                          metadata=OrderedDict(ensemble_name='test-ensemble',
                                               seed=144))

    def test_init(self):
        """Tests initializing DataContainer."""
        self.assertIsInstance(self.dc, DataContainer)

        # test fails with a non ASE Atoms type
        with self.assertRaises(TypeError) as context:
            DataContainer(atoms='atoms',
                          ensemble_parameters=self.ensemble_parameters,
                          metadata=OrderedDict(ensemble_name='test-ensemble',
                                               seed=144))

        self.assertTrue('atoms is not an ASE Atoms object'
                        in str(context.exception))

    def test_atoms(self):
        """Tests reference atoms property."""
        self.assertEqual(self.dc.atoms, self.atoms)

    def test_append_data(self):
        """Tests append data functionality."""
        observers = [ConcreteObserver(interval=10, tag='obs1'),
                     ConcreteObserver(interval=20, tag='obs2')]

        mctrials = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        trajectory_write_interval = 50

        # append data
        observal_interval = min([obs.interval for obs in observers])
        for mctrial in mctrials:
            row_data = {}
            if mctrial % observal_interval == 0:
                for obs in observers:
                    if mctrial % obs.interval == 0:
                        observable = obs.get_observable(self.atoms)
                        row_data[obs.tag] = observable
            if mctrial % trajectory_write_interval == 0:
                row_data['occupations'] = [13, 13, 13]
            self.dc.append(mctrial, row_data)

        # check number of entries
        self.assertEqual(self.dc.get_number_of_entries(), 10)
        self.assertEqual(self.dc.get_number_of_entries('obs2'), 5)
        self.assertEqual(
            self.dc.get_number_of_entries('occupations'), 2)

        # test whether method raises correct Exceptions
        with self.assertRaises(TypeError) as context:
            self.dc.append(5.0, 1.0)
        self.assertTrue('mctrial has the wrong type'
                        in str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.dc.append(10, 1.0)
        self.assertTrue('mctrial values should be given in ascending order'
                        in str(context.exception))

        with self.assertRaises(TypeError) as context:
            self.dc.append(110, 'tst')
        self.assertTrue('record has the wrong type'
                        in str(context.exception))

    def test_update_last_state(self):
        """Tests update_last_state functionality."""
        self.dc._update_last_state(last_step=10001,
                                   occupations=[13] * len(self.atoms),
                                   accepted_trials=12,
                                   random_state=random.getstate())

        for key, value in self.dc._last_state.items():
            if key == 'last_step':
                self.assertIsInstance(value, int)
            if key == 'occupations':
                self.assertIsInstance(value, list)
            if key == 'accepted_trials':
                self.assertIsInstance(value, int)
            if key == 'random_state':
                self.assertIsInstance(value, tuple)

    def test_property_data(self):
        """Tests data property."""
        self.assertIsInstance(self.dc.data, pd.DataFrame)

    def test_property_parameters(self):
        """Tests parameters property."""
        self.assertEqual(self.dc.ensemble_parameters,
                         self.ensemble_parameters)

    def test_property_observables(self):
        """Tests observables property."""
        self.dc.append(mctrial=100,
                       record=dict(obs1=13, potential=-0.123))
        self.assertListEqual(self.dc.observables, ['obs1', 'potential'])

    def test_property_metadata(self):
        """Tests get metadata method."""
        metadata = self.dc.metadata
        self.assertIn('seed', metadata.keys())
        self.assertIn('ensemble_name', metadata.keys())
        self.assertIn('username', metadata.keys())
        self.assertIn('hostname', metadata.keys())
        self.assertIn('icet_version', metadata.keys())

    def test_property_last_state(self):
        """Tests last_state property."""
        self.dc._update_last_state(last_step=10001,
                                   occupations=[13] * len(self.atoms),
                                   accepted_trials=12,
                                   random_state=random.getstate())
        self.assertEqual(self.dc.last_state,
                         dict([('last_step', 10001),
                               ('occupations', [13] * len(self.atoms)),
                               ('accepted_trials', 12),
                               ('random_state', random.getstate())]))

    def test_get_data(self):
        """
        Tests the returned data is a list of list and the options provided by
        the method works as expected.
        """
        # append data to data container
        data_rows = \
            {'mctrial': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
             'acceptance_ratio': [0.0, 0.9, 0.7, 0.7, 0.75, 0.7, 0.6,
                                  0.65, 0.66, 0.666, 0.7],
             'obs1': [16, None, 16, None, 14, None, 16, None, 14, None, 16],
             'obs2': [11, None, None, 13, None, None, 10, None, None, 10, None]
             }

        self.dc._data = pd.DataFrame(data_rows)

        # assert ValueError if no tags are given.
        with self.assertRaises(TypeError) as context:
            self.dc.get_data()
        self.assertTrue('Missing tags argument'
                        in str(context.exception))

        # assert numpy array is returned.
        mctrial, accept_ratio = self.dc.get_data('mctrial', 'acceptance_ratio')
        self.assertIsInstance(mctrial, np.ndarray)
        self.assertIsInstance(accept_ratio, np.ndarray)

        # default skip_none
        mctrial, obs1 = \
            self.dc.get_data('mctrial', 'obs1')
        self.assertEqual(mctrial.tolist(), [0, 20, 40, 60, 80, 100])
        self.assertEqual(obs1.tolist(), [16, 16, 14, 16, 14, 16])

        # using fill_backward
        obs1 = self.dc.get_data('obs1', fill_method='fill_backward')
        self.assertEqual(obs1.tolist(),
                         [16, 16, 16, 14, 14, 16, 16, 14, 14, 16, 16])

        # using fill_forward
        obs1 = self.dc.get_data('obs1', fill_method='fill_forward')
        self.assertEqual(obs1.tolist(),
                         [16, 16, 16, 16, 14, 14, 16, 16, 14, 14, 16])

        # using linear_interpolate
        obs1 = \
            self.dc.get_data('obs1', fill_method='linear_interpolate')
        self.assertEqual(obs1.tolist(),
                         [16, 16, 16, 15, 14, 15, 16, 15, 14, 15, 16])

        # skip_none only for obs1
        obs1, obs2 = self.dc.get_data(
            'obs1', 'obs2', fill_method='skip_none', apply_to=['obs1'])
        self.assertEqual(obs1.tolist(), [16, 16, 14, 16, 14, 16])
        self.assertEqual(obs2.tolist(), [11, None, None, 10, None, None])

        # with a given start, stop and interval
        mctrial, obs1 = \
            self.dc.get_data('mctrial', 'obs1', start=20, stop=80, interval=4)
        self.assertEqual(mctrial.tolist(), [20, 60])
        self.assertEqual(obs1.tolist(), [16, 16])

        # test fails for non-stock data
        with self.assertRaises(ValueError) as context:
            self.dc.get_data('temperature')
        self.assertTrue('No observable named temperature'
                        in str(context.exception))

        # test fails with unknown method
        with self.assertRaises(ValueError) as context:
            self.dc.get_data('mctrial', fill_method='xyz')
        self.assertTrue('Unknown fill method'
                        in str(context.exception))

    def test_reset(self):
        """Tests appended data is cleared."""
        # add some data first
        for mctrial in range(10):
            self.dc.append(mctrial, dict(energy=2.123))
        # clears data
        self.dc.reset()
        self.assertEqual(self.dc.get_number_of_entries(), 0)

    def test_get_number_of_entries(self):
        """Tests number of entries is returned from function."""
        for mctrial in range(10):
            if mctrial % 2 == 0:
                self.dc.append(
                    mctrial, dict(energy=2.123, temperature=4.0))
            else:
                self.dc.append(mctrial, dict(energy=2.123))

        # test total number of entries
        self.assertEqual(self.dc.get_number_of_entries(), 10)
        # test number of entries in the temperature column
        self.assertEqual(self.dc.get_number_of_entries('temperature'), 5)

        # test that the correct Exceptions are raised
        with self.assertRaises(ValueError) as context:
            self.dc.get_number_of_entries('xyz')
        self.assertTrue('No observable named xyz'
                        in str(context.exception))

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
        std = self.dc.get_standard_deviation('obs1')
        self.assertAlmostEqual(mean, 0.9855693, places=7)
        self.assertAlmostEqual(std, 0.1045950, places=7)

        # get average over slice of data
        mean = self.dc.get_average('obs1', start=60)
        std = self.dc.get_standard_deviation('obs1', start=60)
        self.assertAlmostEqual(mean, 0.9851106, places=7)
        self.assertAlmostEqual(std, 0.0981344, places=7)

        mean = self.dc.get_average('obs1', stop=60)
        std = self.dc.get_standard_deviation('obs1', stop=60)
        self.assertAlmostEqual(mean, 0.9876534, places=7)
        self.assertAlmostEqual(std, 0.1086700, places=7)

        mean = self.dc.get_average('obs1', start=40, stop=60)
        std = self.dc.get_standard_deviation('obs1', start=40, stop=60)
        self.assertAlmostEqual(mean, 1.0137074, places=7)
        self.assertAlmostEqual(std, 0.1124826, places=7)

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

    def test_get_trajectory(self):
        """Tests get_trajectory functionality."""
        data_rows = \
            {'mctrial': [0, 10, 20, 30, 40, 50, 60],
             'potential': [-1.32, -1.35, -1.33, -1.07, -1.02, -1.4, -1.3],
             'occupations': [[14, 14, 14, 14, 14, 14, 14, 14],
                             np.nan,
                             [14, 13, 14, 14, 14, 14, 14, 14],
                             np.nan,
                             [14, 13, 13, 14, 14, 13, 14, 14],
                             np.nan,
                             [13, 13, 13, 13, 13, 13, 13, 14]]}

        self.dc._data = pd.DataFrame(data_rows)

        # only trajectory
        occupations = \
            [occ for occ in data_rows['occupations'] if occ is not np.nan]
        atoms_list = self.dc.get_data('trajectory')
        for atoms, occupation in zip(atoms_list, occupations):
            self.assertEqual(atoms.numbers.tolist(), occupation)

        # trajectory and properties
        mctrial, atoms_list, energies = \
            self.dc.get_data('mctrial', 'trajectory', 'potential')

        self.assertEqual(mctrial.tolist(), [0, 20, 40, 60])
        self.assertEqual(energies.tolist(), [-1.32, -1.33, -1.02, -1.3])
        self.assertIsInstance(atoms_list, list)

        # test fails for non skip_none fill method
        with self.assertRaises(ValueError) as context:
            self.dc.get_data('trajectory', fill_method='fill_backward')
        self.assertTrue('Only skip_none fill method is avaliable'
                        ' when trajectory is requested'
                        in str(context.exception))

    def test_write_trajectory(self):
        """Tests write trajectory functionality."""
        # append data
        data_rows = \
            {'mctrial': [0, 10, 20, 30, 40, 50, 60],
             'potential': [-1.32, -1.35, -1.33, -1.07, -1.02, -1.4, -1.3],
             'occupations': [[14, 14, 14, 14, 14, 14, 14, 14],
                             np.nan,
                             [14, 13, 14, 14, 14, 14, 14, 14],
                             np.nan,
                             [14, 13, 13, 14, 14, 13, 14, 14],
                             np.nan,
                             [13, 13, 13, 13, 13, 13, 13, 14]]}

        self.dc._data = pd.DataFrame(data_rows)

        temp_file = tempfile.NamedTemporaryFile()
        self.dc.write_trajectory(temp_file.name)

    def test_read_and_write(self):
        """Tests write and read functionalities of data container."""

        # append data for testing
        row_data = {}
        row_data['obs1'] = 64
        row_data['occupations'] = [13, 13, 13]
        for mctrial in range(1, 101):
            self.dc.append(mctrial, row_data)

        temp_file = tempfile.NamedTemporaryFile()

        # check before with a non-tar file
        with self.assertRaises(TypeError) as context:
            self.dc.read(temp_file.name)
        self.assertTrue('{} is not a tar file'.format(str(temp_file.name))
                        in str(context.exception))

        # save to file
        self.dc.write(temp_file.name)

        # read from file object
        dc_read = self.dc.read(temp_file)

        # check properties and metadata
        self.assertEqual(self.atoms, dc_read.atoms)
        self.assertEqual(self.dc.metadata, dc_read.metadata)
        self.assertEqual(self.dc.ensemble_parameters,
                         dc_read.ensemble_parameters)

        # check data
        pd.testing.assert_frame_equal(
            self.dc.data, dc_read.data, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
