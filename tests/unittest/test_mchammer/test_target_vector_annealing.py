import unittest

import numpy as np
from ase import Atoms
from ase.build import bulk

from icet import ClusterSpace
from mchammer.calculators import TargetVectorCalculator
from mchammer.ensembles import TargetClusterVectorAnnealing


class TestEnsemble(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestEnsemble, self).__init__(*args, **kwargs)

        # setup supercells
        self.prim = bulk('Al')
        self.atoms = []
        atoms = self.prim.repeat(4)
        for i, atom in enumerate(atoms):
            if i % 2 == 0:
                atom.symbol = 'Ga'
        self.atoms.append(atoms)
        atoms = self.prim.repeat(2)
        for i, atom in enumerate(atoms):
            if i % 2 == 0:
                atom.symbol = 'Ga'
        self.atoms.append(atoms)

        # setup cluster expansion
        cutoffs = [5, 5, 4]
        elements = ['Al', 'Ga']
        cs = ClusterSpace(self.prim, cutoffs, elements)
        parameters = parameters = np.array([1.2] * len(cs))

        target_vector = np.linspace(-1, 1, len(cs))

        self.calculators = []
        for atoms in self.atoms:
            self.calculators.append(TargetVectorCalculator(atoms,
                                                           cs,
                                                           target_vector))

        self.T_start = 3.0
        self.T_stop = 0.01

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.ensemble = TargetClusterVectorAnnealing(
            atoms=self.atoms,
            calculators=self.calculators,
            T_start=self.T_start,
            T_stop=self.T_stop,
            random_seed=42)

    def test_generate_structure(self):
        """ Tests that run works and raises if annealing is finished """
        structure = self.ensemble.generate_structure(number_of_trial_steps=13)
        self.assertEqual(self.ensemble.total_trials, 13)
        self.assertIsInstance(structure, Atoms)

    def test_do_trial_step(self):
        """Tests the do trial step."""

        # Do it many times and hopefully get both a reject and an accept
        for _ in range(10):
            self.ensemble._do_trial_step()
        self.assertEqual(self.ensemble.total_trials, 10)

    def test_acceptance_condition(self):
        """Tests the acceptance condition method."""

        # always accept negative delta potential
        self.assertTrue(self.ensemble._acceptance_condition(-1.0))

        # at least run it for positive energy diff
        self.ensemble._acceptance_condition(1.0)


if __name__ == '__main__':
    unittest.main()
