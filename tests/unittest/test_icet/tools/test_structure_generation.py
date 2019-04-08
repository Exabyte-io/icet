#!/usr/bin/env python3
import unittest
import numpy as np
from icet.tools.structure_generation import (_get_sqs_cluster_vector,
                                             _validate_concentrations,
                                             _concentrations_fit_atom_count,
                                             _concentrations_fit_atoms,
                                             _decorate_atoms_randomly,
                                             generate_target_structure,
                                             generate_sqs)
from icet import ClusterSpace
from ase.build import bulk
from ase import Atom


class TestStructureGenerationBinaryFCC(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestStructureGenerationBinaryFCC, self).__init__(*args, **kwargs)
        self.prim = bulk('Au', a=4.0)
        self.supercell = self.prim.repeat(3)
        self.cs = ClusterSpace(self.prim, [6.0, 5.0], ['Au', 'Pd'])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_sqs_cluster_vector(self):
        target_concentrations = {'Au': 0.5, 'Pd': 0.5}
        target_vector = np.array([1.0] + [0.0] * (len(self.cs) - 1))
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        self.assertTrue(np.allclose(cv, target_vector))

        target_concentrations = {'Au': 0.15, 'Pd': 0.85}
        target_vector = np.array([1., -0.7, 0.49, 0.49, 0.49,
                                  0.49, -0.343, -0.343, -0.343,
                                  -0.343, -0.343, -0.343, -0.343])
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        self.assertTrue(np.allclose(cv, target_vector))

    def test_validate_concentrations(self):
        concentrations = {'Au': 0.5, 'Pd': 0.5}
        _validate_concentrations(concentrations, self.cs)

        concentrations = {'Au': 0.1, 'Pd': 0.7}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertTrue('Concentrations must sum up to 1' in str(cm.exception))

        concentrations = {'Au': 0.1, 'Pd': 0.8, 'Cu': 0.1}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertTrue(
            'not the same as those in the specified' in str(cm.exception))

        concentrations = {'Au': 1.0}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertTrue(
            'not the same as those in the specified' in str(cm.exception))

    def test_concentrations_fit_atom_count(self):
        atom_count = 10
        concentrations = {'Au': 0.5, 'Pd': 0.5}
        self.assertTrue(_concentrations_fit_atom_count(
            atom_count, concentrations))

        concentrations = {'Au': 0.15, 'Pd': 0.85}
        self.assertFalse(_concentrations_fit_atom_count(
            atom_count, concentrations))

    def test_concentrations_fit_atoms(self):
        concentrations = {'Au': 1 / 3, 'Pd': 2 / 3}
        self.assertTrue(_concentrations_fit_atoms(
            self.supercell, self.cs, concentrations))

        concentrations = {'Au': 0.5, 'Pd': 0.5}
        self.assertFalse(_concentrations_fit_atoms(
            self.supercell, self.cs, concentrations))

    def test_decorate_atoms_randomly(self):
        atoms = self.prim.repeat(2)
        target_concentrations = {'Au': 0.5, 'Pd': 0.5}
        _decorate_atoms_randomly(atoms, self.cs,
                                 target_concentrations)
        syms = atoms.get_chemical_symbols()
        self.assertEqual(syms.count('Au'), len(atoms) // 2)

        atoms = self.prim.repeat(3)
        target_concentrations = {'Au': 1 / 3, 'Pd': 2 / 3}
        _decorate_atoms_randomly(atoms, self.cs,
                                 target_concentrations)
        syms = atoms.get_chemical_symbols()
        self.assertEqual(syms.count('Au'), len(atoms) // 3)
        self.assertEqual(syms.count('Pd'), 2 * len(atoms) // 3)

    def test_generate_target_structure(self):

        from icet.tools import enumerate_structures
        # Exact target vector from 2 atoms cell
        target_cv = np.array([1., 0., 0., -1., 0., 1.,
                              0., 0., 0., 0., 0., 0., 0.])
        target_conc = {'Au': 0.5, 'Pd': 0.5}
        # for structure in enumerate_structures(self.prim, [2], ['Au', 'Pd']):
        #    print(self.cs.get_cluster_vector(structure))
        # target_cv =

        # This should be simple enough to always work
        structure = generate_target_structure(cluster_space=self.cs,
                                              max_size=4,
                                              target_concentrations=target_conc,
                                              target_cluster_vector=target_cv,
                                              n_steps=500,
                                              random_seed=42)
        self.assertTrue(np.allclose(
            self.cs.get_cluster_vector(structure), target_cv))

    def test_generate_target_structure(self):

        from icet.tools import enumerate_structures
        # Exact target vector from 2 atoms cell
        target_conc = {'Au': 0.5, 'Pd': 0.5}
        # for structure in enumerate_structures(self.prim, [2], ['Au', 'Pd']):
        #    print(self.cs.get_cluster_vector(structure))
        # target_cv =

        # This should be simple enough to always work
        structure = generate_sqs(cluster_space=self.cs,
                                 max_size=4,
                                 target_concentrations=target_conc,
                                 n_steps=500,
                                 random_seed=42,
                                 optimality_weight=2.0)

        target_cv = [1., 0., -0.16666667, 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(
            self.cs.get_cluster_vector(structure), target_cv))


class TestStructureGenerationTernaryFCC(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestStructureGenerationTernaryFCC,
              self).__init__(*args, **kwargs)
        self.prim = bulk('Au', a=4.0)
        self.supercell = self.prim.repeat(3)
        self.cs = ClusterSpace(self.prim, [5.0, 4.0], ['Au', 'Pd', 'Cu'])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_sqs_cluster_vector(self):
        target_concentrations = {'Au': 0.5, 'Pd': 0.3, 'Cu': 0.2}
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        target_vector = [1., 0.2, 0.17320508, 0.04, 0.03464102, 0.03,
                         0.04, 0.03464102, 0.03, 0.04, 0.03464102, 0.03,
                         0.008, 0.0069282, 0.006, 0.00519615, 0.008, 0.0069282,
                         0.0069282, 0.006, 0.006, 0.00519615]
        self.assertTrue(np.allclose(cv, target_vector))

    def test_validate_concentrations(self):
        concentrations = {'Au': 0.1, 'Pd': 0.8, 'Cu': 0.1}
        _validate_concentrations(concentrations, self.cs)

        concentrations = {'Au': 0.1, 'Pd': 0.7, 'Cu': 0.05}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertTrue('Concentrations must sum up to 1' in str(cm.exception))

        concentrations = {'Au': 0.5, 'Pd': 0.5}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertTrue(
            'not the same as those in the specified' in str(cm.exception))

    def test_concentrations_fit_atom_count(self):
        atom_count = 10
        concentrations = {'Au': 0.5, 'Pd': 0.5, 'Cu': 0.0}
        self.assertTrue(_concentrations_fit_atom_count(
            atom_count, concentrations))

        concentrations = {'Au': 0.15, 'Pd': 0.7, 'Cu': 0.15}
        self.assertFalse(_concentrations_fit_atom_count(
            atom_count, concentrations))

    def test_concentrations_fit_atoms(self):
        concentrations = {'Au': 1 / 3, 'Pd': 1 / 3, 'Cu': 1 / 3}
        self.assertTrue(_concentrations_fit_atoms(
            self.supercell, self.cs, concentrations))

        concentrations = {'Au': 0.5, 'Pd': 0.5, 'Cu': 0.0}
        self.assertFalse(_concentrations_fit_atoms(
            self.supercell, self.cs, concentrations))

    def test_decorate_atoms_randomly(self):
        atoms = self.prim.repeat(2)
        target_concentrations = {'Cu': 0.25, 'Au': 0.25, 'Pd': 0.5}
        _decorate_atoms_randomly(atoms, self.cs,
                                 target_concentrations)
        syms = atoms.get_chemical_symbols()
        self.assertEqual(syms.count('Cu'), len(atoms) // 4)
        self.assertEqual(syms.count('Au'), len(atoms) // 4)
        self.assertEqual(syms.count('Pd'), len(atoms) // 2)


class TestStructureGenerationSublatticesFCC(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestStructureGenerationSublatticesFCC,
              self).__init__(*args, **kwargs)
        self.prim = bulk('Au', a=4.0)
        self.prim.append(Atom('H', position=(2.0, 2.0, 2.0)))
        self.supercell = self.prim.repeat(3)
        self.cs = ClusterSpace(self.prim, [5.0, 4.0], [
                               ['Au', 'Pd', 'Cu'], ['H', 'V']])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_sqs_cluster_vector(self):
        target_concentrations = {'Au': 0.2, 'Pd': 0.1,
                                 'Cu': 0.2, 'H': 0.25, 'V': 0.25}
        cv = _get_sqs_cluster_vector(self.cs, target_concentrations)
        target_vector = [1., -0.1, 0.17320508, 0., 0., 0.,
                         0.01, -0.01732051, 0.03, 0., 0., 0.,
                         0.01, -0.01732051, 0.03, 0., 0., 0.,
                         0.01, -0.01732051, 0.03, 0., 0., 0.,
                         0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0.,
                         0., 0., -0.001, 0.00173205, -0.003, 0.00519615,
                         0., -0.001, 0.00173205, 0.00173205, -0.003, -0.003,
                         0.00519615, 0., 0., 0., 0., 0., 0.]
        self.assertTrue(np.allclose(cv, target_vector))

    def test_validate_concentrations(self):
        concentrations = {'Au': 0.1, 'Pd': 0.3, 'Cu': 0.1, 'H': 0.4, 'V': 0.1}
        _validate_concentrations(concentrations, self.cs)

        concentrations = {'Au': 0.1, 'Pd': 0.7, 'Cu': 0.05, 'H': 0.0, 'V': 0.0}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertTrue('Concentrations must sum up to 1' in str(cm.exception))

        concentrations = {'Au': 0.5, 'Pd': 0.5}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertTrue(
            'not the same as those in the specified' in str(cm.exception))

        concentrations = {'Au': 2 / 6, 'Pd': 1 /
                          6, 'Cu': 1 / 6, 'H': 1 / 6, 'V': 1 / 6}
        with self.assertRaises(ValueError) as cm:
            _validate_concentrations(concentrations, self.cs)
        self.assertTrue(
            'that concentrations per element and per sublattice match' in str(cm.exception))

    def test_concentrations_fit_atom_count(self):
        atom_count = 10
        concentrations = {'Au': 0.2, 'Pd': 0.1,
                          'Cu': 0.2, 'H': 0.3, 'V': 0.2}
        self.assertTrue(_concentrations_fit_atom_count(
            atom_count, concentrations))

        concentrations = {'Au': 0.2, 'Pd': 0.2,
                          'Cu': 0.2, 'H': 0.25, 'V': 0.15}
        self.assertFalse(_concentrations_fit_atom_count(
            atom_count, concentrations))

    def test_concentrations_fit_atoms(self):
        concentrations = {'Au': 1 / 6, 'Pd': 1 /
                          6, 'Cu': 1 / 6, 'H': 2 / 6, 'V': 1 / 6}
        self.assertTrue(_concentrations_fit_atoms(
            self.supercell, self.cs, concentrations))

    def test_decorate_atoms_randomly(self):
        atoms = self.prim.repeat(2)
        target_concentrations = {'Cu': 1 / 8, 'Au': 2 / 8, 'Pd': 1 / 8,
                                 'H': 3 / 8, 'V': 1 / 8}
        _decorate_atoms_randomly(atoms, self.cs,
                                 target_concentrations)
        syms = atoms.get_chemical_symbols()
        self.assertEqual(syms.count('Cu'), len(atoms) // 8)
        self.assertEqual(syms.count('Au'), len(atoms) // 4)
        self.assertEqual(syms.count('Pd'), len(atoms) // 8)
        self.assertEqual(syms.count('H'), 3 * len(atoms) // 8)
        self.assertEqual(syms.count('V'), len(atoms) // 8)


if __name__ == '__main__':
    unittest.main()
