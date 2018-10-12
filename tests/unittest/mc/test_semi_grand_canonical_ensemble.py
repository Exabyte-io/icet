import unittest
import numpy as np
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble


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
        self.chemical_potentials = {'Al': 5, 'Ga': 0}
        self.cs = ClusterSpace(self.atoms, cutoffs, elements)
        parameters = parameters = np.array([1.2]*len(self.cs))
        self.ce = ClusterExpansion(self.cs, parameters)
        self.temperature = 100.0

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.atoms, self.ce)

        self.ensemble = SemiGrandCanonicalEnsemble(
            calculator=self.calculator, atoms=self.atoms,
            name='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40,
            temperature=self.temperature,
            chemical_potentials=self.chemical_potentials,
            boltzmann_constant=1e-5)

    def test_property_chemical_potentials(self):
        """Tests property chemical_potentials."""
        retval = self.ensemble.chemical_potentials
        target = {13: 5, 31: 0}
        self.assertEqual(retval, target)

        self.ensemble.chemical_potentials = {'Al': 4, 'Ga': 1}
        retval = self.ensemble.chemical_potentials
        target = {13: 4, 31: 1}
        self.assertEqual(retval, target)

        self.ensemble.chemical_potentials = {13: 10, 31: -1}
        retval = self.ensemble.chemical_potentials
        target = {13: 10, 31: -1}
        self.assertEqual(retval, target)

        self.ensemble.chemical_potentials = {13: 16}
        retval = self.ensemble.chemical_potentials
        target = {13: 16, 31: -1}
        self.assertEqual(retval, target)

        # test exceptions
        with self.assertRaises(TypeError) as context:
            self.ensemble.chemical_potentials = 'xyz'
        self.assertTrue('chemical_potentials has the wrong type'
                        in str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.ensemble.chemical_potentials = {'Ni': 3}
        self.assertTrue('Unknown species'
                        in str(context.exception))

    def test_temperature_attribute(self):
        """Tests temperature attribute."""
        self.assertEqual(self.ensemble.temperature, self.temperature)
        self.ensemble.temperature = 300
        self.assertEqual(self.ensemble.temperature, 300)

    def test_do_trial_step(self):
        """Tests the do trial step."""
        # Do it many times and hopefully get both a reject and an accept
        for _ in range(10):
            self.ensemble._do_trial_step()

        self.assertEqual(self.ensemble.total_trials, 10)

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""
        self.assertTrue(self.ensemble._acceptance_condition(-10.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(10.0)

    def test_init_with_integer_chemical_potentials(self):
        """Tests init with integer chemical potentials."""

        chemical_potentials = {13: 5, 31: 0}
        ensemble = SemiGrandCanonicalEnsemble(
            calculator=self.calculator, atoms=self.atoms, name='test-ensemble',
            random_seed=42, temperature=self.temperature,
            chemical_potentials=chemical_potentials)
        ensemble._do_trial_step()

        # Test both int and str
        chemical_potentials = {'Al': 5, 31: 0}
        ensemble = SemiGrandCanonicalEnsemble(
            calculator=self.calculator, atoms=self.atoms, name='test-ensemble',
            random_seed=42, temperature=self.temperature,
            chemical_potentials=chemical_potentials)
        ensemble._do_trial_step()

    def test__get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()

        self.assertIn('potential', data.keys())
        self.assertIn('Al_count', data.keys())
        self.assertIn('Ga_count', data.keys())
        self.assertIn('mu_Al', data.keys())
        self.assertIn('mu_Ga', data.keys())
        self.assertIn('temperature', data.keys())

        self.assertEqual(data['Al_count'], 13)
        self.assertEqual(data['Ga_count'], 14)
        self.assertEqual(data['temperature'], 100.0)
        self.assertEqual(data['mu_Al'], 5)
        self.assertEqual(data['mu_Ga'], 0)

    def test_write_interval_and_period(self):
        """Tests interval and period for writing data from ensemble."""
        self.assertEqual(self.ensemble.data_container_write_period, 499.0)
        self.assertEqual(self.ensemble._ensemble_data_write_interval, 25)
        self.assertEqual(self.ensemble._trajectory_write_interval, 40)


if __name__ == '__main__':
    unittest.main()
