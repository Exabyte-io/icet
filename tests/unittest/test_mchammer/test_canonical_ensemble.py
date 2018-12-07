import unittest

import numpy as np
from ase.build import bulk

from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator

from mchammer.ensembles.canonical_ensemble import CanonicalEnsemble


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
        self.ensemble = CanonicalEnsemble(
            atoms=self.atoms,
            calculator=self.calculator,
            name='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40,
            temperature=self.temperature)

    def test_init(self):
        """ Tests exceptions are raised during initialization. """
        with self.assertRaises(TypeError) as context:
            CanonicalEnsemble(atoms=self.atoms, calculator=self.calculator)
        self.assertTrue("required positional argument: 'temperature'" in
                        str(context.exception))

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

    def test_property_boltzmann(self):
        """Tests init with explicit Boltzmann constant."""
        from ase.units import kB
        ens = CanonicalEnsemble(
            atoms=self.atoms,
            calculator=self.calculator,
            name='test-ensemble',
            random_seed=42, temperature=100.0)
        self.assertAlmostEqual(kB, ens.boltzmann_constant)

        ens = CanonicalEnsemble(
            atoms=self.atoms,
            calculator=self.calculator,
            name='test-ensemble',
            random_seed=42, temperature=100.0, boltzmann_constant=1.0)
        self.assertAlmostEqual(1.0, ens.boltzmann_constant)

    def test_property_temperature(self):
        """Tests property temperature."""
        self.assertEqual(self.ensemble.temperature, self.temperature)

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()
        self.assertIn('potential', data.keys())

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""
        self.assertEqual(self.ensemble.ensemble_parameters['n_atoms'],
                         len(self.atoms))
        self.assertEqual(self.ensemble.ensemble_parameters['temperature'],
                         self.temperature)
        self.assertEqual(self.ensemble.ensemble_parameters['ensemble_name'],
                         'CanonicalEnsemble')

        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['n_atoms'],
            len(self.atoms))
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['temperature'],
            self.temperature)
        self.assertEqual(
            self.ensemble.data_container.ensemble_parameters['ensemble_name'],
            'CanonicalEnsemble')

    def test_write_interval_and_period(self):
        """Tests interval and period for writing data from ensemble."""
        self.assertEqual(self.ensemble.data_container_write_period, 499.0)
        self.assertEqual(self.ensemble._ensemble_data_write_interval, 25)
        self.assertEqual(self.ensemble._trajectory_write_interval, 40)


if __name__ == '__main__':
    unittest.main()
