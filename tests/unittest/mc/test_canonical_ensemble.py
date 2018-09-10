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
        parameters = parameters = np.array([1.2]*len(self.cs))
        self.ce = ClusterExpansion(self.cs, parameters)
        self.temperature = 100.0

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.atoms, self.ce)
        self.ensemble = CanonicalEnsemble(
            calculator=self.calculator, atoms=self.atoms, name='test-ensemble',
            random_seed=42, temperature=self.temperature)

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

    def test_init_without_temperature(self):
        """ Test init without temperature."""
        with self.assertRaises(TypeError) as context:
            CanonicalEnsemble(
                calculator=self.calculator, atoms=self.atoms,
                name='test-ensemble', random_seed=42)
        self.assertTrue('Missing required argument: temperature'
                        in str(context.exception))

    def test_property_boltzmann(self):
        """ Test init with explicit Boltzmann constant."""
        from ase.units import kB
        ens = CanonicalEnsemble(
            calculator=self.calculator, atoms=self.atoms, name='test-ensemble',
            random_seed=42, temperature=100.0)
        self.assertAlmostEqual(kB, ens.boltzmann_constant)

        ens = CanonicalEnsemble(
            calculator=self.calculator, atoms=self.atoms, name='test-ensemble',
            random_seed=42, temperature=100.0, boltzmann_constant=1.0)
        self.assertAlmostEqual(1.0, ens.boltzmann_constant)

    def test_get_ensemble_data(self):
        """Test the get ensemble data method."""
        data = self.ensemble.get_ensemble_data()

        self.assertIn('potential', data.keys())
        self.assertIn('temperature', data.keys())

        self.assertEqual(data['temperature'], 100.0)


if __name__ == '__main__':
    unittest.main()
