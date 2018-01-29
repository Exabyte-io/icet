#!/usr/bin/env python3

import unittest

from icet.tools import ConvexHull
import numpy as np


class TestConvexHull(unittest.TestCase):
    '''
    Container for tests of the class functionality
    '''

    def __init__(self, *args, **kwargs):
        super(TestConvexHull, self).__init__(*args, **kwargs)
        self.concentrations = [0.0, 1.0, 0.4, 0.6]
        self.energies = [0.0, 10.0, -10.0, 20.0]

    def setUp(self):
        '''
        Instantiate class before each test.
        '''
        self.ch = ConvexHull(self.concentrations, self.energies)

    def test_init(self):
        '''
        Testing that the setup
        (initialization) of tested class works.
        '''
        self.assertEqual(self.ch.dimensions, 1)
        self.assertTrue(np.allclose(self.ch.energies,
                                    np.array([0.0, -10.0, 10.0])))

    def test_get_energy_at_convex_hull(self):
        '''
        Testing energy at convex hull retrieval functionality.
        '''
        convex_hull_energies = self.ch.get_energy_at_convex_hull([0.2, 0.7])
        self.assertTrue(np.allclose(convex_hull_energies, np.array([-5, 0])))

    def test_extract_low_energy_structures(self):
        '''
        Test extracting of structures that are sufficiently close to convex
        hull.
        '''
        from ase.build import bulk
        atoms_1 = bulk('Au')
        atoms_2 = bulk('Pd')
        structures = [atoms_1, atoms_2]
        concentrations = [0.3, 0.7]
        energies = [-7.0, 0.1]
        energy_tolerance = 0.3
        extracted = self.ch.extract_low_energy_structures(concentrations,
                                                          energies,
                                                          energy_tolerance,
                                                          structures)
        self.assertEqual(len(extracted), 1)


class TestConvexHullTernary(unittest.TestCase):
    '''
    Container for tests of the class functionality
    '''

    def __init__(self, *args, **kwargs):
        super(TestConvexHullTernary, self).__init__(*args, **kwargs)
        self.concentrations = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                               [0.1, 0.1], [0.3, 0.3]]
        self.energies = [0.0, 10.0, -10.0, 3.0, -7.0]

    def setUp(self):
        '''
        Instantiate class before each test.
        '''
        self.ch = ConvexHull(self.concentrations, self.energies)

    def test_init(self):
        '''
        Testing that the setup
        (initialization) of tested class works.
        '''
        self.assertEqual(self.ch.dimensions, 2)
        self.assertTrue(np.allclose(self.ch.energies,
                                    np.array([0.0, 10.0, -10.0, -7.0])))

    def test_get_energy_at_convex_hull(self):
        '''
        Testing energy at convex hull retrieval functionality.
        '''
        convex_hull_energies = self.ch.get_energy_at_convex_hull(
            [[0.0, 0.0], [0.15, 0.15]])
        self.assertTrue(np.allclose(convex_hull_energies,
                                    np.array([0.0, -3.5])))

    def test_extract_low_energy_structures(self):
        '''
        Test extracting of structures that are sufficiently close to convex
        hull.
        '''
        concentrations = [[0.0, 0.0], [0.15, 0.15]]
        energies = [0.5, -3.3]
        energy_tolerance = 0.4
        extracted = self.ch.extract_low_energy_structures(concentrations,
                                                          energies,
                                                          energy_tolerance)
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0], 1)


def suite():
    test_classes_to_run = [TestConvexHull, TestConvexHullTernary]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    test_suite = unittest.TestSuite(suites_list)
    return test_suite


if __name__ == '__main__':
    unittest.main()
