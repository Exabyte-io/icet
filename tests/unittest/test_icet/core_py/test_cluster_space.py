#!/usr/bin/env Python3

"""
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

"""

import unittest

from icet.core_py.cluster_space import ClusterSpace


def strip_surrounding_spaces(input_string):
    """
    Helper function that removes both leading and trailing spaces from a
    multi-line string.

    Returns
    -------
    str
        original string minus surrounding spaces and empty lines
    """
    from io import StringIO
    s = []
    for line in StringIO(input_string):
        if len(line.strip()) == 0:
            continue
        s += [line.strip()]
    return '\n'.join(s)


def _assertEqualComplexList(self, retval, target):
    """
    Helper function that conducts a systematic comparison of a nested list
    with dictionaries.
    """
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
    """
    Helper function that conducts an element-wise comparison of two lists.
    """
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
    """Container for tests of the class functionality."""

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

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiates class before each test."""
        self.cs = ClusterSpace(self.atoms_prim, self.cutoffs, self.subelements)

    def test_init(self):
        """Tests that initialization of tested class works."""
        # initialize from ASE Atoms
        cs = ClusterSpace(self.atoms_prim, self.cutoffs, self.subelements)
        self.assertIsInstance(cs, ClusterSpace)

    def test_len(self):
        """Tests len functionality."""
        pass

    def test_orbit_data(self):
        """Tests orbit_data property."""
        pass

    def test_repr(self):
        """Tests repr functionality."""
        pass

    def test_get_string_representation(self):
        """Tests _get_string_representation functionality."""
        pass

    def test_print_overview(self):
        """Tests print_overview functionality."""
        pass

    def test_get_number_of_orbits_by_order(self):
        """Tests get_number_of_orbits_by_order functionality."""
        pass

    def test_get_cluster_vector(self):
        """Tests get_cluster_vector functionality."""
        pass

    def test_get_singlet_info(self):
        """Tests get_singlet_info functionality."""
        pass

    def test_get_singlet_configuration(self):
        """Tests get_singlet_configuration functionality."""
        pass

    def test_get_Mi_from_dict(self):
        """Tests get_Mi_from_dict functionality."""
        pass

    def test_cutoffs(self):
        """Tests cutoffs property."""
        self.assertEqual(self.cs.cutoffs, self.cutoffs)

    def test_structure(self):
        """Tests structure property."""
        self.assertEqual(len(self.cs.structure),
                         len(self.atoms_prim))

    def test_chemical_symbols(self):
        """Tests chemical symbols property."""
        self.assertEqual(self.cs.chemical_symbols, self.subelements)


class TestClusterSpaceSurface(unittest.TestCase):
    """
    Container for tests of the class functionality for non-periodic structures
    """

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

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """
        Instantiate class before each test.
        """
        self.cs = ClusterSpace(self.atoms_prim, self.cutoffs, self.subelements)

    def test_get_cluster_vector(self):
        """
        Tests get_cluster_vector functionality
        """
        pass

    def test_get_number_of_orbits_by_order(self):
        """Tests get_number_of_orbits_by_order functionality."""
        pass
        # retval = self.cs.get_number_of_orbits_by_order()
        # target = OrderedDict([(0, 1), (1, 3), (2, 5), (3, 10), (4, 4)])
        # self.assertEqual(target, retval)

    def test_get_Mi_from_dict(self):
        """Tests _get_Mi_from_dict functionality."""
        pass


if __name__ == '__main__':
    unittest.main()
