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

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.atoms, self.ce)

        self.ensemble = VCSGCEnsemble(
            atoms=self.atoms,
            calculator=self.calculator,
            temperature=self.temperature,
            phis=self.phis,
            kappa=self.kappa,
            boltzmann_constant=1e-5,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(TypeError) as context:
            VCSGCEnsemble(atoms=self.atoms, calculator=self.calculator)
        self.assertTrue("required positional arguments: 'temperature'" in
                        str(context.exception))

        with self.assertRaises(TypeError) as context:
            VCSGCEnsemble(atoms=self.atoms,
                          calculator=self.calculator,
                          temperature=self.temperature)
        self.assertTrue("required positional arguments: 'phis'"
                        in str(context.exception))

        with self.assertRaises(TypeError) as context:
            VCSGCEnsemble(atoms=self.atoms,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis=self.phis)
        self.assertTrue("required positional argument: 'kappa'"
                        in str(context.exception))

        with self.assertRaises(ValueError) as context:
            VCSGCEnsemble(atoms=self.atoms,
                          calculator=self.calculator,
                          temperature=self.temperature,
                          phis={13: -2.0},
                          kappa=self.kappa)
        self.assertTrue('phis were not set' in str(context.exception))

    def test_property_phis(self):
        """Tests phis property."""
        retval = self.ensemble.phis
        target = {13: -1.3, 31: -0.7}
        self.assertEqual(retval, target)

        self.ensemble._set_phis({'Al': -1.2, 'Ga': -0.8})
        retval = self.ensemble.phis
        target = {13: -1.2, 31: -0.8}
        self.assertEqual(retval, target)

        self.ensemble._set_phis({13: -2.2, 31: 0.2})
        retval = self.ensemble.phis
        target = {13: -2.2, 31: 0.2}
        self.assertEqual(retval, target)

        with self.assertRaises(TypeError) as context:
            self.ensemble._set_phis('xyz')
        self.assertTrue('phis has the wrong type' in str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.ensemble._set_phis({13: -1.2, 31: -0.7})
        self.assertTrue('The sum of all phis must' in str(context.exception))

    def test_property_boltzmann(self):
        """Tests explicit Boltzmann constant."""
        self.assertAlmostEqual(1e-5, self.ensemble.boltzmann_constant)

    def test_property_temperature(self):
        """Tests temperature property."""
        self.assertEqual(self.ensemble.temperature, self.temperature)

    def test_property_kappa(self):
        """Tests kappa property."""
        self.assertEqual(self.ensemble.kappa, self.kappa)

    def test_do_trial_step(self):
        """Tests the do trial step."""
        # Do it many times and hopefully get both a reject and an accept
        for _ in range(10):
            self.ensemble._do_trial_step()

        self.assertEqual(self.ensemble._total_trials, 10)

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""

        self.assertTrue(self.ensemble._acceptance_condition(-10.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(10.0)

    def test_init_with_integer_phis(self):
        """Tests init with integer chemical potentials."""

        phis = {13: -1, 31: -1}
        ensemble = VCSGCEnsemble(
            atoms=self.atoms, calculator=self.calculator,
            user_tag='test-ensemble', random_seed=42,
            temperature=self.temperature,
            phis=phis,
            kappa=self.kappa)
        ensemble._do_trial_step()

        # Test both int and str
        phis = {'Al': -1, 31: -1}
        ensemble = VCSGCEnsemble(
            atoms=self.atoms, calculator=self.calculator,
            user_tag='test-ensemble', random_seed=42,
            temperature=self.temperature,
            phis=phis,
            kappa=self.kappa)
        ensemble._do_trial_step()

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()

        self.assertIn('potential', data.keys())
        self.assertIn('Al_count', data.keys())
        self.assertIn('Ga_count', data.keys())

        self.assertEqual(data['Al_count'], 13)
        self.assertEqual(data['Ga_count'], 14)

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms'],
                         len(self.atoms))
        self.assertEqual(self.ensemble.ensemble_parameters['temperature'],
                         self.temperature)
        self.assertEqual(self.ensemble.ensemble_parameters['phi_Al'], -1.3)
        self.assertEqual(self.ensemble.ensemble_parameters['phi_Ga'], -0.7)
        self.assertEqual(self.ensemble.ensemble_parameters['kappa'], 10)

        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['n_atoms'],
            len(self.atoms))
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['temperature'],
            self.temperature)
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['phi_Al'], -1.3)
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['phi_Ga'], -0.7)
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['kappa'], 10)

    def test_write_interval_and_period(self):
        """
        Tests interval and period for writing data from ensemble.
        """
        self.assertEqual(self.ensemble.data_container_write_period, 499.0)
        self.assertEqual(self.ensemble._ensemble_data_write_interval, 25)
        self.assertEqual(self.ensemble._trajectory_write_interval, 40)


if __name__ == '__main__':
    unittest.main()
