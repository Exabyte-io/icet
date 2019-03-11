import unittest

import numpy as np
from ase.build import bulk

from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator

from mchammer.ensembles.canonical_annealing import CanonicalAnnealing, available_cooling_functions


class TestEnsemble(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        # setup supercell
        self.atoms = bulk('Al').repeat(3)
        for i, atom in enumerate(self.atoms):
            if i % 2 == 0:
                atom.symbol = 'Ga'

        # setup cluster expansion
        cutoffs = [5, 5, 4]
        elements = ['Al', 'Ga']
        cs = ClusterSpace(self.atoms, cutoffs, elements)
        parameters = parameters = np.array([1.2] * len(cs))
        self.ce = ClusterExpansion(cs, parameters)

        self.T_start = 1000.0
        self.T_stop = 0.0
        self.n_steps = 500

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.atoms, self.ce)
        self.ensemble = CanonicalAnnealing(
            atoms=self.atoms,
            calculator=self.calculator,
            T_start=self.T_start,
            T_stop=self.T_stop,
            n_steps=self.n_steps,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40)

    def test_different_cooling_functions(self):
        """ Tests that different cooling functions works """

        # make sure init works with available cooling functions
        for f_name in available_cooling_functions.keys():
            mc = CanonicalAnnealing(
                atoms=self.atoms, calculator=self.calculator, T_start=self.T_start,
                T_stop=self.T_stop, n_steps=self.n_steps, cooling_function=f_name)
            mc.run()

        # make sure init works with user-defined cooling function
        def best_annealing_function(step, T_start, T_stop, n_steps):
            return np.random.uniform(T_stop, T_start)
        mc = CanonicalAnnealing(
            atoms=self.atoms, calculator=self.calculator, T_start=self.T_start,
            T_stop=self.T_stop, n_steps=self.n_steps, cooling_function=best_annealing_function)
        mc.run()

    def test_run(self):
        """ Tests that run works and raises if annealing is finished """
        self.ensemble.run()

        with self.assertRaises(Exception) as context:
            self.ensemble.run()
        self.assertIn('Annealing has already finished', str(context.exception))

    def test_do_trial_step(self):
        """Tests the do trial step."""

        # Do it many times and hopefully get both a reject and an accept
        for _ in range(10):
            self.ensemble._do_trial_step()

        self.assertEqual(self.ensemble._total_trials, 10)

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""

        # always accept negative delta potential
        self.assertTrue(self.ensemble._acceptance_condition(-10.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(10.0)

    def test_get_ensemble_data(self):
        """Tests the get ensemble data method."""
        data = self.ensemble._get_ensemble_data()
        self.assertIn('temperature', data.keys())
        self.assertIn('potential', data.keys())

    def test_ensemble_parameters(self):
        """Tests the get ensemble parameters method."""
        self.assertEqual(self.ensemble.ensemble_parameters['n_steps'], self.n_steps)

if __name__ == '__main__':
    unittest.main()
