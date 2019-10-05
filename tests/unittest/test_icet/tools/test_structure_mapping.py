#!/usr/bin/env python3

import numpy as np
import unittest
from tempfile import NamedTemporaryFile

from icet.tools import map_structure_to_reference
from icet.tools.structure_mapping import (_get_reference_supercell,
                                          _match_positions)
from icet.io.logging import logger, set_log_config
from ase import Atom


class TestStructureMapping(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        from ase.build import bulk

        reference = bulk('Au', a=4.0)
        reference.append(Atom('H', (2, 2, 2)))
        self.reference = reference

        structure = reference.repeat(2)
        for i in [0, 1, 2, 3, 4, 6, 7]:
            if structure[i].symbol == 'Au':
                structure[i].symbol = 'Pd'
            elif structure[i].symbol == 'H':
                del structure[i]

        # Displace the atoms somewhat
        rattle = [[0.147, -0.037, -0.01],
                  [-0.089, 0.084, -0.063],
                  [0.256, -0.037, 0.097],
                  [-0.048, 0.005, -0.093],
                  [-0.159, -0.194, -0.03],
                  [0.004, -0.041, -0.003],
                  [-0.015, -0.014, -0.007],
                  [-0.023, 0.094, -0.024],
                  [-0.01, 0.075, -0.075],
                  [0.029, -0.024, 0.079],
                  [0.105, 0.172, -0.147]]

        structure.positions = structure.positions + rattle
        structure.set_cell(structure.cell * 1.01, scale_atoms=True)
        self.structure = structure

        super(TestStructureMapping, self).__init__(*args, **kwargs)

    def test_get_reference_supercell(self):
        """
        Tests that retrieval of a reference supercell works.
        """
        supercell = _get_reference_supercell(self.structure, self.reference)
        target_cell = np.array([[0., 4., 4.],
                                [4., 0., 4.],
                                [4., 4., 0.]])
        target_formula = 'H8Au8'
        self.assertTrue(np.allclose(supercell.cell, target_cell))
        self.assertEqual(supercell.get_chemical_formula(), target_formula)

        # Should work with default tol_cell when inert_species are specified
        supercell = _get_reference_supercell(self.structure, self.reference,
                                             inert_species=['Au', 'Pd'])
        self.assertTrue(np.allclose(supercell.cell, target_cell))
        self.assertEqual(supercell.get_chemical_formula(), target_formula)

        # Assuming no cell relaxation, proper match
        structure_scaled = self.structure.copy()
        structure_scaled.set_cell(structure_scaled.cell / 1.01, scale_atoms=True)
        supercell = _get_reference_supercell(structure_scaled, self.reference,
                                             assume_no_cell_relaxation=True)
        self.assertTrue(np.allclose(supercell.cell, target_cell))
        self.assertEqual(supercell.get_chemical_formula(), target_formula)

        # Mismatch in boundary conditions
        structure_nopbc = self.structure.copy()
        structure_nopbc.set_pbc([True, True, False])
        with self.assertRaises(ValueError) as context:
            _get_reference_supercell(structure_nopbc, self.reference)
        self.assertIn('The boundary conditions of', str(context.exception))

    def test_match_positions(self):
        """
        Tests that the final step in mapping works.
        """
        # Mismatching cell metrics
        with self.assertRaises(ValueError) as context:
            _match_positions(self.structure, self.reference.repeat(2))
        self.assertIn('The cell metrics of reference and relaxed',
                      str(context.exception))

        # Mismatching cell metrics, too many atoms in relaxed cell
        reference = self.reference.repeat(2)
        reference.set_cell(reference.cell * 1.01, scale_atoms=True)
        structure = self.structure.copy()
        for x in np.linspace(0, 1, 10):
            structure.append(Atom('Au', position=(x, 0, 0)))
        with self.assertRaises(ValueError) as context:
            _match_positions(structure, reference)
        self.assertIn('The relaxed structure contains more atoms than the reference',
                      str(context.exception))

        # Mismatching boundary conditions
        reference = self.reference.repeat(2)
        reference.set_cell(reference.cell * 1.01, scale_atoms=True)
        reference.set_pbc([True, False, False])
        with self.assertRaises(ValueError) as context:
            _match_positions(structure, reference)
        self.assertIn('The boundary conditions of', str(context.exception))

        # Working example
        reference = self.reference.repeat(2)
        reference.set_cell(reference.cell * 1.01, scale_atoms=True)
        mapped, drmax, dravg = _match_positions(self.structure, reference)
        self.assertAlmostEqual(drmax, 0.279012386)
        self.assertAlmostEqual(dravg, 0.140424392)
        self.assertEqual(mapped.get_chemical_formula(), 'H3Au6Pd2X5')

    def test_map_structure_to_reference(self):
        """
        Tests that mapping algorithm wrapper works.
        """
        def test_mapping(structure,
                         expected_drmax=0.276249887,
                         expected_dravg=0.139034051,
                         **kwargs):
            """
            Convenience wrapper for testing mapping.
            """
            logfile = NamedTemporaryFile(mode='w+', encoding='utf-8')
            set_log_config(filename=logfile.name)
            mapped, info = map_structure_to_reference(structure,
                                                      self.reference,
                                                      **kwargs)
            self.assertEqual(len(info), 5)
            self.assertAlmostEqual(info['drmax'], expected_drmax)
            self.assertAlmostEqual(info['dravg'], expected_dravg)
            self.assertEqual(mapped.get_chemical_formula(), 'H3Au6Pd2X5')
            logfile.seek(0)
            return logfile

        # Log ClusterSpace output to StringIO stream
        for handler in logger.handlers:
            logger.removeHandler(handler)

        # Standard, warning-free mapping
        logfile = test_mapping(self.structure, inert_species=['Au', 'Pd'])
        self.assertEqual(len(logfile.readlines()), 0)

        # Warn when there is a lot of volumetric strain
        structure = self.structure.copy()
        structure.set_cell(1.2 * structure.cell, scale_atoms=True)
        logfile = test_mapping(structure, inert_species=['Au', 'Pd'])
        lines = logfile.readlines()
        self.assertEqual(len(lines), 1)
        self.assertIn('High volumetric strain', lines[0])

        # Do not warn if warnings are suppressed
        structure = self.structure.copy()
        structure.set_cell(1.2 * structure.cell, scale_atoms=True)
        logfile = test_mapping(structure, inert_species=['Au', 'Pd'],
                               suppress_warnings=True)
        lines = logfile.readlines()
        self.assertEqual(len(lines), 0)

        # Warning-free assuming no cell relaxation
        structure = self.structure.copy()
        structure.set_cell((1 / 1.01) * structure.cell, scale_atoms=True)
        logfile = test_mapping(structure, assume_no_cell_relaxation=True)
        lines = logfile.readlines()
        self.assertEqual(len(lines), 0)

        # Warning even with little strain with no cell relaxation assumption
        structure = self.structure.copy()
        structure.set_cell(structure.cell, scale_atoms=True)
        logfile = test_mapping(structure, assume_no_cell_relaxation=True)
        lines = logfile.readlines()
        self.assertEqual(len(lines), 1)
        self.assertIn('High volumetric strain', lines[0])

        # Anisotropic strain
        structure = self.structure.copy()
        A = [[1.2, 0, 0], [0, 1 / 1.2, 0], [0, 0, 1.]]
        structure.set_cell(np.dot(structure.cell, A), scale_atoms=True)
        logfile = test_mapping(structure, inert_species=['Au', 'Pd'])
        lines = logfile.readlines()
        self.assertEqual(len(lines), 1)
        self.assertIn('High anisotropic strain', lines[0])

        # Large deviations
        structure = self.structure.copy()
        structure.positions += [1, 0, 0]
        logfile = test_mapping(structure, inert_species=['Au', 'Pd'],
                               expected_drmax=1.11822844,
                               expected_dravg=0.95130331)
        lines = logfile.readlines()
        self.assertEqual(len(lines), 2)
        self.assertIn('Large maximum relaxation distance', lines[0])
        self.assertIn('Large average relaxation distance', lines[1])


if __name__ == '__main__':
    unittest.main()
