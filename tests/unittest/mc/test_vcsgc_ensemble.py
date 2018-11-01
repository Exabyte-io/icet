import unittest

import numpy as np
from ase.build import bulk

from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator

from mchammer.ensembles import VCSGCEnsemble


class TestEnsemble(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        self.atoms = bulk('Al').repeat(3)
        for i, atom in enumerate(self.atoms):
            if i % 2 == 0:
                atom.symbol = 'Ga'
        cutoffs = [5, 5, 4]
        elements = ['Al', 'Ga']
        self.phis = {'Al': -1.3, 'Ga': -0.7}
        self.kappa = 10.0
        self.cs = ClusterSpace(self.atoms, cutoffs, elements)
        parameters = parameters = np.array([1.2] * len(self.cs))
        self.ce = ClusterExpansion(self.cs, parameters)
        self.temperature = 100.0

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.atoms, self.ce)

        self.ensemble = VCSGCEnsemble(
            calculator=self.calculator, atoms=self.atoms,
            name='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40,
            temperature=self.temperature,
            phis=self.phis,
            kappa=self.kappa,
            boltzmann_constant=1e-5)

    def test_property_phis(self):
        """Test property phis."""
        retval = self.ensemble.phis
        target = {13: -1.3, 31: -0.7}
        self.assertEqual(retval, target)

        self.ensemble.phis = {'Al': -1.2, 'Ga': -0.8}
        retval = self.ensemble.phis
        target = {13: -1.2, 31: -0.8}
        self.assertEqual(retval, target)

        self.ensemble.phis = {13: -2.2, 31: 0.2}
        retval = self.ensemble.phis
        target = {13: -2.2, 31: 0.2}
        self.assertEqual(retval, target)

        # test exceptions
        with self.assertRaises(ValueError) as context:
            self.ensemble.phis = {13: -2.0}
        self.assertTrue('phis were not set' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            self.ensemble.phis = 'xyz'
        self.assertTrue('phis must be dict' in str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.ensemble.phis = {13: -1.2, 31: -0.7}
        self.assertTrue('The sum of all phis must' in str(context.exception))

    def test_temperature_attribute(self):
        """Test temperature attribute."""
        self.assertEqual(self.ensemble.temperature, self.temperature)
        self.ensemble.temperature = 300
        self.assertEqual(self.ensemble.temperature, 300)

    def test_do_trial_step(self):
        """Test the do trial step."""
        # Do it many times and hopefully get both a reject and an accept
        for _ in range(10):
            self.ensemble._do_trial_step()

        self.assertEqual(self.ensemble.total_trials, 10)

    def test_acceptance_condition(self):
        """ Test the acceptance condition method."""

        self.assertTrue(self.ensemble._acceptance_condition(-10.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(10.0)

    def test_init_with_integer_phis(self):
        """ Test init with integer chemical potentials."""

        phis = {13: -1, 31: -1}
        ensemble = VCSGCEnsemble(
            calculator=self.calculator, atoms=self.atoms, name='test-ensemble',
            random_seed=42, temperature=self.temperature,
            phis=phis,
            kappa=self.kappa)
        ensemble._do_trial_step()

        # Test both int and str
        phis = {'Al': -1, 31: -1}
        ensemble = VCSGCEnsemble(
            calculator=self.calculator, atoms=self.atoms, name='test-ensemble',
            random_seed=42, temperature=self.temperature,
            phis=phis,
            kappa=self.kappa)
        ensemble._do_trial_step()

    def test_get_ensemble_data(self):
        """Test the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()

        self.assertIn('potential', data.keys())
        self.assertIn('Al_count', data.keys())
        self.assertIn('Ga_count', data.keys())
        self.assertIn('phi_Al', data.keys())
        self.assertIn('phi_Ga', data.keys())
        self.assertIn('kappa', data.keys())
        self.assertIn('temperature', data.keys())

        self.assertEqual(data['Al_count'], 13)
        self.assertEqual(data['Ga_count'], 14)
        self.assertEqual(data['temperature'], 100.0)
        self.assertAlmostEqual(data['phi_Al'], -1.3)
        self.assertAlmostEqual(data['phi_Ga'], -0.7)
        self.assertEqual(data['kappa'], 10)

    def test_write_interval_and_period(self):
        """
        Test interval and period for writing data from ensemble.
        """
        self.assertEqual(self.ensemble.data_container_write_period, 499.0)
        self.assertEqual(self.ensemble._ensemble_data_write_interval, 25)
        self.assertEqual(self.ensemble._trajectory_write_interval, 40)


if __name__ == '__main__':
    unittest.main()
