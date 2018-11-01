import unittest
import tempfile
import random
from collections import OrderedDict
import numpy as np
import pandas as pd
from ase.build import bulk
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
        self.atoms = bulk('Al').repeat(4)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test case."""
        self.dc = DataContainer(self.atoms,
                                ensemble_name='test-ensemble',
                                random_seed=44)
        test_observer = ConcreteObserver(interval=10, tag='obs1')
        self.dc.add_observable(test_observer.tag)
        self.dc.add_parameter('temperature', 375.15)

    def test_init(self):
        """Tests initializing DataContainer."""
        self.assertIsInstance(self.dc, DataContainer)

        # test fails with a non ASE Atoms type
        with self.assertRaises(TypeError) as context:
            DataContainer('atoms', 'test-ensemble', 44)
        self.assertTrue('atoms is not an ASE Atoms object'
                        in str(context.exception))

    def test_structure(self):
        """Tests reference structure property."""
        self.assertEqual(self.dc.atoms, self.atoms)

    def test_add_observable(self):
        """Tests add observable functionality."""
        test_observer = ConcreteObserver(interval=20, tag='obs2')
        self.dc.add_observable(test_observer.tag)
        self.assertEqual(len(self.dc.observables), 2)

        # test no duplicates
        self.dc.add_observable(test_observer.tag)
        self.assertEqual(len(self.dc.observables), 2)

        # test whether method raises Exception
        with self.assertRaises(TypeError) as context:
            self.dc.add_observable(1)
        self.assertTrue('tag has the wrong type'
                        in str(context.exception))

    def test_add_parameter(self):
        """Tests add parameter functionality."""
        self.dc.add_parameter('sro', -0.1)

        # add a list as parameters
        index_atoms = [i for i in range(len(self.atoms))]
        self.dc.add_parameter('index_atoms', index_atoms)
        self.assertEqual(len(self.dc.parameters), 4)

        # test whether method raises Exceptions
        with self.assertRaises(TypeError) as context:
            self.dc.add_parameter(1, 'tst')
        self.assertTrue('tag has the wrong type'
                        in str(context.exception))
        with self.assertRaises(TypeError) as context:
            self.dc.add_parameter('tst', 'tst')
        self.assertTrue('value has the wrong type'
                        in str(context.exception))

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
                                   occupations=[13]*len(self.atoms),
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
        self.assertEqual(self.dc.parameters,
                         OrderedDict([('seed', 44),
                                      ('temperature', 375.15)]))

    def test_property_observables(self):
        """Tests observables property."""
        self.assertListEqual(self.dc.observables, ['obs1'])

    def test_property_metadata(self):
        """
        Tests metadata property.
        """
        for key in self.dc.metadata:
            self.assertIsInstance(self.dc.metadata[key], str)

    def test_property_last_state(self):
        """Tests last_state property."""
        self.dc._update_last_state(last_step=10001,
                                   occupations=[13]*len(self.atoms),
                                   accepted_trials=12,
                                   random_state=random.getstate())
        self.assertEqual(self.dc.last_state,
                         dict([('last_step', 10001),
                               ('occupations', [13]*len(self.atoms)),
                               ('accepted_trials', 12),
                               ('random_state', random.getstate())]))

    def test_get_data(self):
        """
        Tests the returned data is a list of list and the options provided by
        the method works as expected.
        """
        # lets suppose the following data was appended to the data container
        # mctrials
        mctrials = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # energies
        energy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # observable 1
        obs1 = [1, None, 3, None, 5, None, 7, None, 9, None]
        # observable 2
        obs2 = [None, None, 3, None, None, None, 7, None, None, None]

        rows_data = \
            {'mctrial': mctrials, 'potential': energy,
             'obs1': obs1, 'obs2': obs2}

        self.dc._data = \
            pd.DataFrame(rows_data,
                         columns=['mctrial', 'potential', 'obs1', 'obs2'])

        retval = self.dc.get_data()
        self.assertEqual(retval, (mctrials, energy, obs1, obs2))

        # using skip_none
        retval1, retval2 = \
            self.dc.get_data(tags=['mctrial', 'obs1'], fill_method='skip_none')
        self.assertEqual(retval1, [10, 30, 50, 70, 90])
        self.assertEqual(retval2, [1, 3, 5, 7, 9])

        # using fill_backward
        retval = self.dc.get_data(tags=['obs1'], fill_method='fill_backward')
        self.assertEqual(retval, [1, 3, 3, 5, 5, 7, 7, 9, 9])

        # using fill_forward
        retval = self.dc.get_data(tags=['obs1'], fill_method='fill_forward')
        self.assertEqual(retval, [1, 1, 3, 3, 5, 5, 7, 7, 9, 9])

        # using linear_interpolate
        retval = \
            self.dc.get_data(tags=['obs1'], fill_method='linear_interpolate')
        self.assertEqual(retval, [1, 2, 3, 4, 5, 6, 7, 8, 9])

        # skip_none only for obs1
        retval1, retval2 = \
            self.dc.get_data(tags=['obs1', 'obs2'],
                             fill_method='skip_none', apply_to=['obs1'])
        self.assertEqual(retval1, [1, 3, 5, 7, 9])
        self.assertEqual(retval2, [None, 3, None, 7, None])

        # with a given start, stop and interval
        retval1, retval2 = \
            self.dc.get_data(tags=['mctrial', 'obs1'],
                             start=20, stop=90, interval=3)
        self.assertEqual(retval1, [20, 50, 80])
        self.assertEqual(retval2, [None, 5, None])

        # check occupations
        occupations = [np.nan, np.nan, np.nan, np.nan, [13, 13, 11],
                       np.nan, np.nan, np.nan, np.nan, [13, 11, 11]]
        rows_data = {'mctrial': mctrials, 'occupations': occupations}
        self.dc._data = \
            pd.DataFrame(rows_data, columns=['mctrial', 'occupations'])
        retval = \
            self.dc.get_data(tags=['occupations'], start=50, interval=5)
        self.assertEqual(retval, [[13, 13, 11], [13, 11, 11]])

        # test fails for non-stock data
        with self.assertRaises(ValueError) as context:
            self.dc.get_data(['temperature'])
        self.assertTrue('No observable named temperature'
                        in str(context.exception))

        # test fails with unkown method
        with self.assertRaises(ValueError) as context:
            self.dc.get_data(fill_method='xyz')
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

    def test_get_average(self):
        """Tests get average functionality."""
        # set up a random list of values with a normal distribution
        n_iter, mu, sigma = 100, 1.0, 0.1
        np.random.seed(12)
        obs_val = np.random.normal(mu, sigma, n_iter).tolist()

        # append above random data to data container
        for mctrial in range(n_iter):
            self.dc.append(mctrial, record={'obs1': obs_val[mctrial]})

        # get average over all mctrials
        mean, std = self.dc.get_average('obs1')
        self.assertAlmostEqual(mean, 0.9855693, places=7)
        self.assertAlmostEqual(std, 0.1051220, places=7)

        # get average over slice of data
        mean, std = self.dc.get_average('obs1', start=60)
        self.assertAlmostEqual(mean, 0.9851106, places=7)
        self.assertAlmostEqual(std, 0.0981344, places=7)

        mean, std = self.dc.get_average('obs1', stop=60)
        self.assertAlmostEqual(mean, 0.9876534, places=7)
        self.assertAlmostEqual(std, 0.1086700, places=7)

        mean, std = self.dc.get_average('obs1', start=40, stop=60)
        self.assertAlmostEqual(mean, 1.0137074, places=7)
        self.assertAlmostEqual(std, 0.1124826, places=7)

        # test fails for non-existing data
        with self.assertRaises(ValueError) as context:
            self.dc.get_average('temperature')
        self.assertTrue('No observable named temperature'
                        in str(context.exception))

        # test fails for non-scalar data like occupation vector
        self.dc.reset()
        for mctrial in range(10):
            self.dc.append(
                mctrial, record={'occupations': [14, 14, 14]})

        with self.assertRaises(TypeError) as context:
            self.dc.get_average('occupations')
        self.assertTrue('occupations is not scalar'
                        in str(context.exception))

    def test_get_trajectory(self):
        """Tests get_trajectory functionality."""
        occupation_vector = [14] * len(self.atoms)
        row_data = dict(occupations=occupation_vector,
                        potential=-0.120000001)

        for mctrial in range(len(self.atoms)):
            self.dc.append(mctrial, row_data)

        # only trajectory
        for atoms in self.dc.get_trajectory():
            self.assertEqual(atoms.numbers.tolist(), occupation_vector)

        # trajectory and energies
        atoms_list, energies \
            = self.dc.get_trajectory(scalar_property='potential')
        for atoms, energy in zip(atoms_list, energies):
            self.assertEqual(atoms.numbers.tolist(), occupation_vector)
            self.assertEqual(energy, -0.120000001)

    def test_write_trajectory(self):
        """Tests write trajectory functionality."""
        # append data
        occupation_vector = [14] * len(self.atoms)
        row_data = dict(occupations=occupation_vector,
                        potential=-0.120000001)
        for mctrial in range(len(self.atoms)):
            self.dc.append(mctrial, row_data)
        # save to file
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
        self.assertEqual(self.dc.parameters, dc_read.parameters)
        self.assertEqual(self.dc.observables, dc_read.observables)

        # check data
        pd.testing.assert_frame_equal(
            self.dc.data, dc_read.data, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
