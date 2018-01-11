#!/usr/bin/env Python3

'''
This file contains unit tests and other tests. It can be executed by
simply executing this file from a shell prompt:

    $ ./test_cluster_space.py

In which case it will use the system's default Python version. If a specific
Python version should be used, run that Python version with this file as input,
e.g.:

    python3 test_cluster_space.py

For a description of the Python unit testing framework, see this link:
https://docs.python.org/3/library/unittest.html

When executing this file doc testing is also performed on all doc tests in
the cluster_space.py file

'''

import unittest

from icet import ClusterSpace
from icet.core.cluster_space import (get_singlet_info,
                                     get_singlet_configuration)
from icet.core.structure import Structure
from icet.core.lattice_site import LatticeSite
from collections import OrderedDict


def strip_surrounding_spaces(input_string):
    '''
    Helper function that removes both leading and trailing spaces from a
    multi-line string.

    Returns
    -------
    str
        original string minus surrounding spaces and empty lines
    '''
    from io import StringIO
    s = []
    for line in StringIO(input_string):
        if len(line.strip()) == 0:
            continue
        s += [line.strip()]
    return '\n'.join(s)


def _assertEqualComplexList(self, retval, target):
    '''
    Helper function that conducts a systematic comparison of a nested list
    with dictionaries.
    '''
    self.assertIsInstance(retval, type(target))
    for row_retval, row_target in zip(retval, target):
        self.assertIsInstance(row_retval, type(row_target))
        for key, val in row_target.items():
            self.assertIn(key, row_retval)
            s = ['key: {}'.format(key)]
            s += ['type: {}'.format(type(key))]
            s += ['retval: {}'.format(row_retval[key])]
            s += ['target: {}'.format(val)]
            info = '   '.join(s)
            if isinstance(val, float):
                self.assertAlmostEqual(val, row_retval[key], places=9,
                                       msg=info)
            else:
                self.assertEqual(row_retval[key], val, msg=info)


unittest.TestCase.assertEqualComplexList = _assertEqualComplexList


def _assertAlmostEqualList(self, retval, target, places=6):
    '''
    Helper function that conducts an element-wise comparison of two lists.
    '''
    self.assertIsInstance(retval, type(target))
    self.assertEqual(len(retval), len(target))
    for k, (r, t) in enumerate(zip(retval, target)):
        s = ['element: {}'.format(k)]
        s += ['retval: {} ({})'.format(r, type(r))]
        s += ['target: {} ({})'.format(t, type(t))]
        info = '   '.join(s)
        self.assertAlmostEqual(r, t, places=places, msg=info)


unittest.TestCase.assertAlmostEqualList = _assertAlmostEqualList


class TestClusterSpace(unittest.TestCase):
    '''
    Container for tests of the class functionality
    '''

    def __init__(self, *args, **kwargs):
        from ase.build import bulk
        super(TestClusterSpace, self).__init__(*args, **kwargs)
        self.subelements = ['Ag', 'Au']
        self.cutoffs = [4.0] * 3
        self.atoms_prim = bulk('Ag', a=4.09)
        self.structure_list = []
        for k in range(4):
            atoms = self.atoms_prim.repeat(2)
            symbols = [self.subelements[0]] * len(atoms)
            symbols[:k] = [self.subelements[1]] * k
            atoms.set_chemical_symbols(symbols)
            self.structure_list.append(atoms)

    def setUp(self):
        '''
        Instantiate class before each test.
        '''
        self.cs = ClusterSpace(self.atoms_prim, self.cutoffs, self.subelements)

    def test_init(self):
        '''
        Just testing that the setup
        (initialization) of tested class work
        '''
        # initialize from ASE Atoms
        cs = ClusterSpace(self.atoms_prim, self.cutoffs, self.subelements)
        self.assertIsInstance(cs, ClusterSpace)
        self.assertEqual(len(cs), len(self.cs))
        # initialize from icet Structure
        cs = ClusterSpace(Structure.from_atoms(self.atoms_prim), self.cutoffs,
                          self.subelements)
        self.assertIsInstance(cs, ClusterSpace)
        self.assertEqual(len(cs), len(self.cs))
        cs = ClusterSpace(Structure.from_atoms(self.atoms_prim), self.cutoffs,
                          self.subelements)
        # check that other types fail
        with self.assertRaises(Exception) as context:
            cs = ClusterSpace('something', self.cutoffs, self.subelements)
        self.assertTrue('Unknown structure format' in str(context.exception))
        # check Mi as int
        cs = ClusterSpace(self.atoms_prim, self.cutoffs,
                          self.subelements, Mi=2)
        self.assertIsInstance(cs, ClusterSpace)
        self.assertEqual(len(cs), len(self.cs))
        # check Mi as dict
        cs = ClusterSpace(self.atoms_prim, self.cutoffs,
                          self.subelements, Mi={0: 2})
        self.assertIsInstance(cs, ClusterSpace)
        self.assertEqual(len(cs), len(self.cs))

    def test_len(self):
        '''
        Testing len functionality
        '''
        number_orbits = self.cs.__len__()
        self.assertEqual(number_orbits, len(self.cs.get_orbit_list()))

    def test_get_orbit_list_info(self):
        '''
        Testing get_orbit_list_info functionality
        '''
        target = [OrderedDict([('index', 0), ('order', 1),
                               ('size', 0.0),
                               ('multiplicity', 1),
                               ('orbit index', 0),
                               ('MC vector', [0])]),
                  OrderedDict([('index', 1), ('order', 2),
                               ('size', 1.4460333675264896),
                               ('multiplicity', 6),
                               ('orbit index', 1),
                               ('MC vector', [0, 0])]),
                  OrderedDict([('index', 2), ('order', 3),
                               ('size', 1.6697355079971996),
                               ('multiplicity', 8),
                               ('orbit index', 2),
                               ('MC vector', [0, 0, 0])]),
                  OrderedDict([('index', 3), ('order', 4),
                               ('size', 1.771021950739177),
                               ('multiplicity', 2),
                               ('orbit index', 3),
                               ('MC vector', [0, 0, 0, 0])])]
        retval = self.cs.get_orbit_list_info()
        self.assertEqualComplexList(retval, target)

    def test_repr(self):
        '''
        Testing repr functionality
        '''
        retval = self.cs.__repr__()
        target = '''
-------------------------- Cluster Space ---------------------------
 subelements: Ag Au
 cutoffs: 4.0000 4.0000 4.0000
 total number of orbits: 4
 number of orbits by order: 1= 1  2= 1  3= 1  4= 1
--------------------------------------------------------------------
index | order |   size   | multiplicity | orbit index |  MC vector
--------------------------------------------------------------------
   0  |   1   |   0.0000 |        1     |       0     |    [0]
   1  |   2   |   1.4460 |        6     |       1     |  [0, 0]
   2  |   3   |   1.6697 |        8     |       2     | [0, 0, 0]
   3  |   4   |   1.7710 |        2     |       3     | [0, 0, 0, 0]
--------------------------------------------------------------------
'''
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_get_string_representation(self):
        '''
        Testing _get_string_representation functionality
        '''
        retval = self.cs._get_string_representation(print_threshold=2,
                                                    print_minimum=1)
        target = '''
-------------------------- Cluster Space ---------------------------
 subelements: Ag Au
 cutoffs: 4.0000 4.0000 4.0000
 total number of orbits: 4
 number of orbits by order: 1= 1  2= 1  3= 1  4= 1
--------------------------------------------------------------------
index | order |   size   | multiplicity | orbit index |  MC vector
--------------------------------------------------------------------
   0  |   1   |   0.0000 |        1     |       0     |    [0]
 ...
   3  |   4   |   1.7710 |        2     |       3     | [0, 0, 0, 0]
--------------------------------------------------------------------
'''
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_print_overview(self):
        '''
        Testing print_overview functionality
        '''
        # this runs the function but since the latter merely invokes another
        # function to print some information to stdout there is not much to
        # test here (other than the function not throwing an exception)
        self.cs.print_overview()

    def test_get_number_of_orbits_by_order(self):
        '''
        Testing get_number_of_orbits_by_order functionality
        '''
        retval = self.cs.get_number_of_orbits_by_order()
        target = OrderedDict([(1, 1), (2, 1), (3, 1), (4, 1)])
        self.assertEqual(target, retval)

    def test_get_cluster_vector(self):
        '''
        Testing get_cluster_vector functionality
        '''
        target_cluster_vectors = [
            [1.0, -1.0, 1.0, -1.0, 1.0],
            [1.0, -0.75, 0.5, -0.25, 0.0],
            [1.0, -0.5, 0.16666666666666666, 0.0, 0.0],
            [1.0, -0.25, 0.0, 0.0, 0.0]]
        s = ['Error in test setup;']
        s += ['number of cluster vectors ({})'.format(
            len(target_cluster_vectors))]
        s += ['does not match']
        s += ['number of structures ({})'.format(len(self.structure_list))]
        info = ' '.join(s)
        self.assertEqual(len(target_cluster_vectors), len(self.structure_list),
                         msg=info)
        # use ASE Atoms
        for atoms, target in zip(self.structure_list, target_cluster_vectors):
            retval = self.cs.get_cluster_vector(atoms)
            self.assertAlmostEqual(retval, target, places=9)
        # use icet Structure
        for atoms, target in zip(self.structure_list, target_cluster_vectors):
            retval = self.cs.get_cluster_vector(Structure.from_atoms(atoms))
            self.assertAlmostEqual(retval, target, places=9)
        # check that other types fail
        with self.assertRaises(Exception) as context:
            retval = self.cs.get_cluster_vector('something')
        self.assertTrue('Unknown structure format' in str(context.exception))

    def test_get_singlet_info(self):
        '''
        Testing get_singlet_info functionality
        '''
        retval = get_singlet_info(self.structure_list[0])
        target = [{'orbit index': 0,
                   'sites': [[LatticeSite(0, [0., 0., 0.])]],
                   'multiplicity': 1,
                   'representative site': [LatticeSite(0, [0., 0., 0.])]}]
        self.assertEqualComplexList(retval, target)
        retval1, retval2 = get_singlet_info(self.structure_list[0],
                                            return_cluster_space=True)
        self.assertEqualComplexList(retval1, target)
        self.assertIsInstance(retval2, type(self.cs))

    def test_get_singlet_configuration(self):
        '''
        Testing get_singlet_configuration functionality
        '''
        from ase import Atoms
        retval = get_singlet_configuration(self.atoms_prim)
        self.assertIsInstance(retval, Atoms)
        self.assertEqual(retval[0].symbol, 'H')
        retval = get_singlet_configuration(self.structure_list[0],
                                           to_primitive=True)
        self.assertIsInstance(retval, Atoms)
        self.assertEqual(len(retval), len(self.atoms_prim))

    def test_get_Mi_from_dict(self):
        '''
        Testing get_Mi_from_dict functionality
        '''
        d = {0: len(self.subelements)}
        Mi = ClusterSpace._get_Mi_from_dict(d, self.atoms_prim)
        self.assertEqual(Mi, [2])
        # check that function fails if dictionary is incomplete
        del d[0]
        with self.assertRaises(Exception) as context:
            Mi = ClusterSpace._get_Mi_from_dict(d, self.atoms_prim)
        self.assertTrue('missing from dictionary' in str(context.exception))

    def test_cutoffs(self):
        ''' Testing cutoffs property '''
        self.assertEqual(self.cs.cutoffs, self.cutoffs)

    def test_structure(self):
        ''' Testing structure property '''
        self.assertEqual(len(self.cs.structure),
                         len(self.atoms_prim))


class TestClusterSpaceSurface(unittest.TestCase):
    '''
    Container for tests of the class functionality for non-periodic structures
    '''

    def __init__(self, *args, **kwargs):
        from ase.build import fcc111
        super(TestClusterSpaceSurface, self).__init__(*args, **kwargs)
        self.subelements = ['Ag', 'Au']
        self.cutoffs = [4.0] * 3
        self.atoms_prim = fcc111('Ag', a=4.09, vacuum=5.0, size=[1, 1, 3])
        self.atoms_prim.pbc = [True, True, False]
        self.structure_list = []
        for k in range(3):
            atoms = self.atoms_prim.repeat((2, 2, 1))
            symbols = [self.subelements[0]] * len(atoms)
            symbols[:k] = [self.subelements[1]] * k
            atoms.set_chemical_symbols(symbols)
            self.structure_list.append(atoms)

    def setUp(self):
        '''
        Instantiate class before each test.
        '''
        self.cs = ClusterSpace(self.atoms_prim, self.cutoffs, self.subelements)

    def test_get_cluster_vector(self):
        '''
        Testing get_cluster_vector functionality
        '''
        target_cluster_vectors = [
            [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
             -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, -0.3333333333333333, -1.0, -1.0, -0.1111111111111111,
             0.5555555555555556, 1.0, 1.0, 1.0, 0.3333333333333333,
             0.3333333333333333, -1.0, -1.0, -1.0, -1.0, 0.1111111111111111,
             -0.5555555555555556, -1.0, -1.0, -0.3333333333333333,
             0.6666666666666666, 1.0, 1.0],
            [1.0, -0.3333333333333333, -0.3333333333333333, -1.0,
             -0.1111111111111111, 0.3333333333333333, -0.1111111111111111,
             0.4444444444444444, 1.0, 0.3333333333333333, 0.3333333333333333,
             0.3333333333333333, 0.3333333333333333, -1.0, -1.0,
             -0.1111111111111111, -0.1111111111111111, 0.1111111111111111,
             -0.4444444444444444, 0.3333333333333333, -0.3333333333333333,
             0.3333333333333333, -0.3333333333333333]]
        s = ['Error in test setup;']
        s += ['number of cluster vectors ({})'.format(
            len(target_cluster_vectors))]
        s += ['does not match']
        s += ['number of structures ({})'.format(len(self.structure_list))]
        info = ' '.join(s)
        self.assertEqual(len(target_cluster_vectors), len(self.structure_list),
                         msg=info)
        # use ASE Atoms
        for atoms, target in zip(self.structure_list, target_cluster_vectors):
            retval = self.cs.get_cluster_vector(atoms)
            self.assertAlmostEqual(retval, target, places=9)
        # use icet Structure
        for atoms, target in zip(self.structure_list, target_cluster_vectors):
            retval = self.cs.get_cluster_vector(Structure.from_atoms(atoms))
            self.assertAlmostEqual(retval, target, places=9)
        # check that other types fail
        with self.assertRaises(Exception) as context:
            retval = self.cs.get_cluster_vector('something')
        self.assertTrue('Unknown structure format' in str(context.exception))

    def test_get_number_of_orbits_by_order(self):
        '''
        Testing get_number_of_orbits_by_order functionality
        '''
        retval = self.cs.get_number_of_orbits_by_order()
        target = OrderedDict([(1, 3), (2, 5), (3, 10), (4, 4)])
        self.assertEqual(target, retval)

    def test_get_Mi_from_dict(self):
        '''
        Testing _get_Mi_from_dict functionality
        '''
        d = {}
        for k in range(len(self.atoms_prim)):
            d[k] = len(self.subelements)
        d[1] = 1
        Mi = ClusterSpace._get_Mi_from_dict(d, self.atoms_prim)
        self.assertEqual(Mi, [2, 1, 2])
        # check that function fails if dictionary is incomplete
        del d[0]
        with self.assertRaises(Exception) as context:
            Mi = ClusterSpace._get_Mi_from_dict(d, self.atoms_prim)
        self.assertTrue('missing from dictionary' in str(context.exception))


def suite():
    test_classes_to_run = [TestClusterSpace, TestClusterSpaceSurface]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    test_suite = unittest.TestSuite(suites_list)
    return test_suite


if __name__ == '__main__':
    unittest.main()
