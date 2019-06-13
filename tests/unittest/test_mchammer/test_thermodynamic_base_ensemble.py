import unittest
import numpy as np
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
from mchammer.ensembles.vcsgc_ensemble import get_phis


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
        parameters = parameters = np.array([1.2] * len(self.cs))
        self.ce = ClusterExpansion(self.cs, parameters)
        self.temperature = 100.0

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.calculator = ClusterExpansionCalculator(self.atoms, self.ce)

        self.atoms = bulk('Al').repeat(3)
        for i, atom in enumerate(self.atoms):
            if i % 2 == 0:
                atom.symbol = 'Ga'

        self.ensemble = SemiGrandCanonicalEnsemble(
            atoms=self.atoms,
            calculator=self.calculator,
            user_tag='test-ensemble', random_seed=42,
            data_container_write_period=499.0,
            ensemble_data_write_interval=25,
            trajectory_write_interval=40,
            temperature=self.temperature,
            chemical_potentials=self.chemical_potentials,
            boltzmann_constant=1e-5)

    def test_do_sgc_trial_step(self):
        """Tests the do trial step."""
        chemical_potentials = self.ensemble._chemical_potentials

        for _ in range(10):
            sl_index = self.ensemble.get_random_sublattice_index(
                self.ensemble._flip_sublattice_probabilities)
            self.ensemble.do_sgc_flip(sublattice_index=sl_index,
                                      chemical_potentials=chemical_potentials)
        self.assertEqual(self.ensemble._total_trials, 10)

    def test_do_canonical_trial_step(self):
        """Tests the do trial step."""

        for _ in range(10):
            sl_index = self.ensemble.get_random_sublattice_index(
                self.ensemble._flip_sublattice_probabilities)
            self.ensemble.do_canonical_swap(sublattice_index=sl_index)
        self.assertEqual(self.ensemble._total_trials, 10)

    def test_do_vcsgc_flip(self):
        """Test the vcsgc flip."""
        kappa = 200
        phis = {'Al': -1, 'Ga': -1}
        phis = get_phis(phis)
        for _ in range(10):
            sl_index = self.ensemble.get_random_sublattice_index(
                self.ensemble._flip_sublattice_probabilities)
            self.ensemble.do_vcsgc_flip(phis=phis, kappa=kappa, sublattice_index=sl_index)
        self.assertEqual(self.ensemble._total_trials, 10)


if __name__ == '__main__':
    unittest.main()
