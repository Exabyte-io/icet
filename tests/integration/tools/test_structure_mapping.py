#!/usr/bin/env python3

import numpy as np
import unittest

from icet.tools import map_structure_to_reference
from icet.tools.structure_mapping import (_get_scaled_cell,
                                          _get_transformation_matrix,
                                          _rescale_structures)


class TestStructureMapping(unittest.TestCase):
    '''
    Container for tests of the class functionality
    '''

    def __init__(self, *args, **kwargs):
        from ase import Atom
        from ase.build import bulk

        reference = bulk('Au', a=4.0)
        reference.append(Atom('H', (2, 2, 2)))
        self.reference = reference

        atoms = reference.repeat(2)
        for i in [0, 4, 6, 2, 7, 3]:
            if atoms[i].symbol == 'Au':
                atoms[i].symbol = 'Pd'
            elif atoms[i].symbol == 'H':
                del atoms[i]

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
                  [0.105, 0.172, -0.147],
                  [-0.082, 0.073, 0.015],
                  [-0.062, -0.066, 0.023],
                  [0.081, 0.078, -0.014]]

        atoms.positions = atoms.positions + rattle
        atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)
        self.atoms = atoms

        super(TestStructureMapping, self).__init__(*args, **kwargs)

    def test_get_scaled_cell(self):
        '''
        Testing that the cell scaling retrieval works.
        '''
        modcell = _get_scaled_cell(self.atoms, self.reference,
                                   vacancy_type='V',
                                   inert_species=['Au', 'Pd'])
        target = np.array([[0., 4., 4.],
                           [4., 0., 4.],
                           [4., 4., 0.]])
        self.assertTrue(np.allclose(modcell, target))

        modcell = _get_scaled_cell(self.atoms, self.reference,
                                   vacancy_type='V')
        target = np.array([[0., 4.04, 4.04],
                           [4.04, 0., 4.04],
                           [4.04, 4.04, 0.]])
        self.assertTrue(np.allclose(modcell, target))

        modcell = _get_scaled_cell(self.atoms, self.reference)
        target = np.array([[0., 3.82586237, 3.82586237],
                           [3.82586237, 0., 3.82586237],
                           [3.82586237, 3.82586237, 0.]])
        self.assertTrue(np.allclose(modcell, target))

    def test_get_transformation_matrix(self):
        '''
        Testing that transformation matrix calculation works.
        '''
        P = _get_transformation_matrix(self.atoms.cell, self.reference.cell)
        target = np.array([[2., 0., 0.],
                           [0., 2., 0.],
                           [0., 0., 2.]])
        self.assertTrue(np.allclose(P, target))

    def test_rescale_structures(self):
        '''
        Testing that the structure rescaling works.
        '''
        P = np.array([[2., 0., 0.],
                      [0., 2., 0.],
                      [0., 0., 2.]])
        scaled_structure, ideal_supercell = \
            _rescale_structures(self.atoms, self.reference, P)
        self.assertEqual(len(scaled_structure), 14)
        self.assertEqual(len(ideal_supercell), 16)
        self.assertTrue(np.allclose(scaled_structure.cell,
                                    ideal_supercell.cell))

    def test_map_structure_to_reference(self):
        '''
        Testing that the mapping works. This function will also invoke the
        other ones.
        '''
        mapped, r_max, r_av = \
            map_structure_to_reference(self.atoms, self.reference,
                                       0.7,
                                       vacancy_type='V',
                                       inert_species=['Au', 'Pd'])
        self.assertEqual(mapped.get_chemical_formula(), 'H6Au4Pd4V2')
        self.assertTrue(r_av < r_max)


def suite():
    test_classes_to_run = [TestStructureMapping]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    test_suite = unittest.TestSuite(suites_list)
    return test_suite


if __name__ == '__main__':
    unittest.main()
