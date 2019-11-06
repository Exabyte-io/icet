#!/usr/bin/env Python3

"""
This file contains unit tests and other tests. It can be executed by
simply executing this file from a shell prompt:

    $ ./test_ground_state_finder.py

In which case it will use the system's default Python version. If a specific
Python version should be used, run that Python version with this file as input,
e.g.:

    python3 test_ground_state_finder.py

For a description of the Python unit testing framework, see this link:
https://docs.python.org/3/library/unittest.html

When executing this file doc testing is also performed on all doc tests in
the cluster_space.py file

"""

from io import StringIO
import unittest

from ase import Atom
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace
try:
    from icet.extras.ground_state_finder import GroundStateFinder, is_sites_in_orbit
except ImportError as ex:
    module = ex.args[0].split()[0]
    if module == 'Python-MIP':
        raise unittest.SkipTest('no mip module'.format(module))
    else:
        raise


def strip_surrounding_spaces(input_string):
    """
    Helper function that removes both leading and trailing spaces from a
    multi-line string.

    Returns
    -------
    str
        original string minus surrounding spaces and empty lines
    """
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


class TestGroundStateFinder(unittest.TestCase):
    """Container for test of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestGroundStateFinder, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Ag', 'Au']
        self.cutoffs = [4.3]
        self.structure_prim = bulk('Au', a=4.0)
        self.cs = ClusterSpace(self.structure_prim, self.cutoffs, self.chemical_symbols)
        self.ce = ClusterExpansion(self.cs, [0, 0, 0.1, -0.02])
        self.all_possible_structures = []
        self.supercell = self.structure_prim.repeat(2)
        for i in range(len(self.supercell)):
            structure = self.supercell.copy()
            structure.symbols[i] = self.chemical_symbols[0]
            self.all_possible_structures.append(structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.gsf = GroundStateFinder(self.ce)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from GroundStateFinder instance
        gsf = GroundStateFinder(self.ce)
        self.assertIsInstance(gsf, GroundStateFinder)

    def test_init_with_species_to_count(self):
        """Tests that initialization of tested class work."""
        # initialize from GroundStateFinder instance
        gsf = GroundStateFinder(self.ce, species_to_count=self.chemical_symbols[0])
        self.assertIsInstance(gsf, GroundStateFinder)

    def test_init_fails_for_quaternary_with_two_active_sublattices(self):
        """Tests that initialization fails if there are two active sublattices."""
        a = 4.0
        structure_prim = bulk('Au', a=a)
        structure_prim.append(Atom('H', position=(a / 2, a / 2, a / 2)))
        chemical_symbols = [['Au', 'Pd'], ['H', 'V']]
        cs = ClusterSpace(structure_prim, cutoffs=self.cutoffs,
                          chemical_symbols=chemical_symbols)
        ce = ClusterExpansion(cs, [0.0]*len(cs))
        with self.assertRaises(NotImplementedError) as cm:
            GroundStateFinder(ce)
        self.assertTrue('Only binaries are implemented as of yet.' in str(cm.exception))

    def test_init_fails_for_ternary_with_one_active_sublattice(self):
        """Tests that initialization fails for a ternary system with one active sublattice."""
        chemical_symbols = ['Au', 'Ag', 'Pd']
        cs = ClusterSpace(self.structure_prim, cutoffs=self.cutoffs,
                          chemical_symbols=chemical_symbols)
        ce = ClusterExpansion(cs, [0.0]*len(cs))
        with self.assertRaises(NotImplementedError) as cm:
            GroundStateFinder(ce)
        self.assertTrue('Only binaries are implemented as of yet.' in str(cm.exception))

    def test_init_fails_for_faulty_species_to_count(self):
        """Tests that initialization fails if species_to_count is faulty."""
        species_to_count = 'H'
        with self.assertRaises(ValueError) as cm:
            GroundStateFinder(self.ce, species_to_count=species_to_count)
        self.assertTrue('The specified species {} is not found on the active sublattice'
                        ' ({})'.format(species_to_count, self.chemical_symbols)
                        in str(cm.exception))

    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure) for structure in self.all_possible_structures])
        ground_state = self.gsf.get_ground_state(self.supercell, species_count=1, verbose=0)
        predicted_val = self.ce.predict(ground_state)
        self.assertEqual(predicted_val, target_val)

    def test_create_cluster_maps(self):
        """Tests _create_cluster_maps functionality """
        gsf = GroundStateFinder(self.ce)
        gsf._create_cluster_maps(self.structure_prim)

        # Test cluster to sites map
        target = [[0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                  [0, 0], [0, 0]]
        self.assertEqual(target, gsf._cluster_to_sites_map)

        # Test cluster to orbit map
        target = [0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        self.assertEqual(target, gsf._cluster_to_orbit_map)

        # Test ncluster per orbit map
        target = [1, 1, 6, 3]
        self.assertEqual(target, gsf._nclusters_per_orbit)

    def test_get_active_orbit_indices(self):
        """Tests _get_active_orbit_indices functionality """
        retval = self.gsf._get_active_orbit_indices(self.structure_prim)
        target = [0, 1, 2]
        self.assertEqual(target, retval)

    def test_get_transformation_matrix(self):
        """Tests _get_transformation_matrix functionality """
        retval = self.gsf._get_transformation_matrix(self.structure_prim)
        target = [
            [1, 1, 1, 1],
            [0, -2, -4, -4],
            [0, 0, 4, 0],
            [0, 0, 0, 4]]
        self.assertEqual(target, retval.tolist())

    def test_is_sites_in_orbit(self):
        """Tests is_sites_in_orbit functionality """
        orbit = self.cs.get_orbit(2)
        sites = orbit.get_equivalent_sites()[0]
        self.assertTrue(is_sites_in_orbit(orbit, sites))

        orbit = self.cs.get_orbit(1)
        self.assertFalse(is_sites_in_orbit(orbit, sites))

        orbit = self.cs.get_orbit(2)
        sites = orbit.get_equivalent_sites()[0][0:1]
        self.assertFalse(is_sites_in_orbit(orbit, sites))


class TestGroundStateFinderInactiveSublattice(unittest.TestCase):
    """Container for test of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestGroundStateFinderInactiveSublattice, self).__init__(*args, **kwargs)
        self.chemical_symbols = [['Ag', 'Au'], ['H']]
        self.cutoffs = [4.3]
        a = 4.0
        structure_prim = bulk('Au', a=a)
        structure_prim.append(Atom('H', position=(a / 2, a / 2, a / 2)))
        self.structure_prim = structure_prim
        self.cs = ClusterSpace(self.structure_prim, self.cutoffs, self.chemical_symbols)
        self.ce = ClusterExpansion(self.cs, [0, 0, 0.1, -0.02])
        self.all_possible_structures = []
        self.supercell = self.structure_prim.repeat(2)
        for i, sym in enumerate(self.supercell.get_chemical_symbols()):
            if sym not in self.chemical_symbols[0]:
                continue
            structure = self.supercell.copy()
            structure.symbols[i] = self.chemical_symbols[0][0]
            self.all_possible_structures.append(structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.gsf = GroundStateFinder(self.ce)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ClusterExpansion instance
        gsf = GroundStateFinder(self.ce)
        self.assertIsInstance(gsf, GroundStateFinder)

    def test_init_with_species_to_count(self):
        """Tests that initialization of tested class work."""
        # initialize from GroundStateFinder instance
        gsf = GroundStateFinder(self.ce, species_to_count=self.chemical_symbols[0][0])
        self.assertIsInstance(gsf, GroundStateFinder)

    def test_init_fails_for_faulty_species_to_count(self):
        """Tests that initialization fails if species_to_count is faulty."""
        species_to_count = self.chemical_symbols[1][0]
        with self.assertRaises(ValueError) as cm:
            GroundStateFinder(self.ce, species_to_count=species_to_count)
        self.assertTrue('The specified species {} is not found on the active sublattice'
                        ' ({})'.format(species_to_count, self.chemical_symbols[0])
                        in str(cm.exception))

    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure) for structure in self.all_possible_structures])
        ground_state = self.gsf.get_ground_state(self.supercell, species_count=1, verbose=0)
        predicted_val = self.ce.predict(ground_state)
        self.assertEqual(predicted_val, target_val)

    def test_create_cluster_maps(self):
        """Tests _create_cluster_maps functionality """
        gsf = GroundStateFinder(self.ce)
        gsf._create_cluster_maps(self.structure_prim)

        # Test cluster to sites map
        target = [[0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                  [0, 0], [0, 0]]
        self.assertEqual(target, gsf._cluster_to_sites_map)

        # Test cluster to orbit map
        target = [0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        self.assertEqual(target, gsf._cluster_to_orbit_map)

        # Test ncluster per orbit map
        target = [1, 1, 6, 3]
        self.assertEqual(target, gsf._nclusters_per_orbit)

    def test_get_active_orbit_indices(self):
        """Tests _get_active_orbit_indices functionality """
        retval = self.gsf._get_active_orbit_indices(self.structure_prim)
        target = [0, 3, 6]
        self.assertEqual(target, retval)

    def test_get_transformation_matrix(self):
        """Tests _get_transformation_matrix functionality """
        retval = self.gsf._get_transformation_matrix(self.structure_prim)
        target = [
            [1, 1, 1, 1],
            [0, -2, -4, -4],
            [0, 0, 4, 0],
            [0, 0, 0, 4]]
        self.assertEqual(target, retval.tolist())


if __name__ == '__main__':
    unittest.main()
