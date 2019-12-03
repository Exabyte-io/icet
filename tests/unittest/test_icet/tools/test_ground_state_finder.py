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

import unittest
from io import StringIO

from ase import Atom
from ase.build import bulk
from ase.build import fcc111
from icet import ClusterExpansion, ClusterSpace
try:
    from icet.tools.ground_state_finder import GroundStateFinder
except ImportError as ex:
    module = ex.args[0].split()[0]
    if module == 'Python-MIP':
        raise unittest.SkipTest('no mip module'.format(module))
    else:
        raise


def find_orbit_and_equivalent_site_with_indices(orbit_list, site_indices):
    """
    Go through the orbit list and find the equivalent with the specified list
    of site indices
    ----------
    orbit_list
        list of orbits
    site_indices
        list of lattice sites indices
    """

    for i in range(len(orbit_list)):
        orbit = orbit_list.get_orbit(i)

        # Check if the number of sites matches the order of the orbit
        if len(site_indices) != orbit.order:
            continue

        for sites in orbit.get_equivalent_sites():

            # Check if the list of site indices matches those for the equivalent site
            if all(sites[j].index == site_indices[j] for j in range(len(site_indices))):
                return orbit, sites

        return None, None


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
        self.cs = ClusterSpace(self.structure_prim, self.cutoffs,
                               self.chemical_symbols)
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
        self.gsf = GroundStateFinder(self.ce, self.supercell, verbose=False)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from GroundStateFinder instance
        gsf = GroundStateFinder(self.ce, self.supercell, verbose=False)
        self.assertIsInstance(gsf, GroundStateFinder)

    def test_init_solver(self):
        """Tests that initialization of tested class work."""
        # initialize from GroundStateFinder instance
        # Set the solver explicitely
        gsf = GroundStateFinder(self.ce, self.supercell, solver_name='CBC',
                                verbose=False)
        self.assertEqual('CBC', gsf._model.solver_name.upper())

    def test_init_fails_for_quaternary_with_two_active_sublattices(self):
        """Tests that initialization fails if there are two active
        sublattices."""
        a = 4.0
        structure_prim = bulk('Au', a=a)
        structure_prim.append(Atom('H', position=(a / 2, a / 2, a / 2)))
        chemical_symbols = [['Au', 'Pd'], ['H', 'V']]
        cs = ClusterSpace(structure_prim, cutoffs=self.cutoffs,
                          chemical_symbols=chemical_symbols)
        ce = ClusterExpansion(cs, [0.0]*len(cs))
        with self.assertRaises(NotImplementedError) as cm:
            GroundStateFinder(ce, self.supercell, verbose=False)
        self.assertTrue('Currently, only one active sublattice is allowed.'
                        in str(cm.exception))

    def test_init_fails_for_ternary_with_one_active_sublattice(self):
        """Tests that initialization fails for a ternary system with one active
        sublattice."""
        chemical_symbols = ['Au', 'Ag', 'Pd']
        cs = ClusterSpace(self.structure_prim, cutoffs=self.cutoffs,
                          chemical_symbols=chemical_symbols)
        ce = ClusterExpansion(cs, [0.0]*len(cs))
        with self.assertRaises(NotImplementedError) as cm:
            GroundStateFinder(ce, self.supercell, verbose=False)
        self.assertTrue('Only binaries are implemented as of yet.'
                        in str(cm.exception))

    def test_optimization_status_property(self):
        """Tests the optimization_status property."""

        # Check that the optimization_status is None initially
        self.assertIsNone(self.gsf.optimization_status)

        # Check that the optimization_status is OPTIMAL if a ground state is found
        species_count = {self.chemical_symbols[0]: 1}
        self.gsf.get_ground_state(species_count=species_count, threads=1)
        self.assertEqual(str(self.gsf.optimization_status),
                         'OptimizationStatus.OPTIMAL')

    def test_model_property(self):
        """Tests the model property."""
        self.assertEqual(self.gsf.model.name, 'CE')

    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure)
                          for structure in self.all_possible_structures])

        # Provide counts for first species
        species_count = {self.chemical_symbols[0]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species0 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, target_val)

        # Provide counts for second species
        species_count = {self.chemical_symbols[1]: len(self.supercell) - 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species1 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, predicted_species1)

        # Set the maximum run time
        species_count = {self.chemical_symbols[0]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count,
                                                 max_seconds=0.5, threads=1)
        predicted_max_seconds = self.ce.predict(ground_state)
        self.assertGreaterEqual(predicted_max_seconds, predicted_species0)

    def test_get_ground_state_fails_for_faulty_species_to_count(self):
        """Tests that get_ground_state fails if species_to_count is faulty."""
        # Check that get_ground_state fails if counts are provided for multiple
        # species
        species_count = {self.chemical_symbols[0]: 1,
                         self.chemical_symbols[1]: len(self.supercell) - 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count, threads=1)
        self.assertTrue('Provide counts for one of the species on the active sublattice ({}),'
                        ' not {}!'.format(self.gsf._species, list(species_count.keys()))
                        in str(cm.exception))

        # Check that get_ground_state fails if counts are provided for a
        # species not found on the active sublattice
        species_count = {'H': 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count, threads=1)
        self.assertTrue('The species {} is not present on the active sublattice'
                        ' ({})'.format(list(species_count.keys())[0], self.gsf._species)
                        in str(cm.exception))

    def test_create_cluster_maps(self):
        """Tests _create_cluster_maps functionality """
        gsf = GroundStateFinder(self.ce, self.supercell, verbose=False)
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


class TestGroundStateFinderInactiveSublattice(unittest.TestCase):
    """Container for test of the class functionality for a system with an
    inactive sublattice."""

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
        self.gsf = GroundStateFinder(self.ce, self.supercell, verbose=False)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ClusterExpansion instance
        gsf = GroundStateFinder(self.ce, self.supercell, verbose=False)
        self.assertIsInstance(gsf, GroundStateFinder)

    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure)
                          for structure in self.all_possible_structures])

        # Provide counts for first species
        species_count = {self.chemical_symbols[0][0]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species0 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, target_val)

        # Provide counts for second species
        species_count = {self.chemical_symbols[0][1]:
                         len([sym for sym in self.supercell.get_chemical_symbols() if sym in
                              self.chemical_symbols[0]]) - 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species1 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, predicted_species1)

    def test_get_ground_state_fails_for_faulty_species_to_count(self):
        """Tests that get_ground_state fails if species_to_count is faulty."""
        # Check that get_ground_state fails if counts are provided for multiple species
        species_count = {self.chemical_symbols[0][0]: 1,
                         self.chemical_symbols[0][1]: len([sym for sym in self.supercell if sym in
                                                           self.chemical_symbols[0]]) - 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count, threads=1)
        self.assertTrue('Provide counts for one of the species on the active '
                        'sublattice ({}), '
                        'not {}!'.format(self.gsf._species,
                                         list(species_count.keys()))
                        in str(cm.exception))

        # Check that get_ground_state fails if counts are provided for a
        # species not found on the active sublattice
        species_count = {'H': 1}
        with self.assertRaises(ValueError) as cm:
            self.gsf.get_ground_state(species_count=species_count, threads=1)
        self.assertTrue('The species {} is not present on the active sublattice'
                        ' ({})'.format(list(species_count.keys())[0],
                                       self.gsf._species)
                        in str(cm.exception))

    def test_create_cluster_maps(self):
        """Tests _create_cluster_maps functionality """
        gsf = GroundStateFinder(self.ce, self.supercell, verbose=False)
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


class TestGroundStateFinderTriplets(unittest.TestCase):
    """Container for test of the class functionality for a system with
    triplets."""

    def __init__(self, *args, **kwargs):
        super(TestGroundStateFinderTriplets, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Au', 'Pd']
        self.cutoffs = [3.0, 3.0]
        structure_prim = fcc111('Au', a=4.0, size=(1, 1, 6), vacuum=10,
                                periodic=True)
        structure_prim.wrap()
        self.structure_prim = structure_prim
        self.cs = ClusterSpace(self.structure_prim, self.cutoffs,
                               self.chemical_symbols)
        ecis = [0.0] * 4 + [0.1] * 6 + [-0.02] * 11
        self.ce = ClusterExpansion(self.cs, ecis)
        self.all_possible_structures = []
        self.supercell = self.structure_prim.repeat((2, 2, 1))
        for i in range(len(self.supercell)):
            structure = self.supercell.copy()
            structure.symbols[i] = self.chemical_symbols[1]
            self.all_possible_structures.append(structure)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.gsf = GroundStateFinder(self.ce, self.supercell, verbose=False)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ClusterExpansion instance
        gsf = GroundStateFinder(self.ce, self.supercell, verbose=False)
        self.assertIsInstance(gsf, GroundStateFinder)

    def test_get_ground_state(self):
        """Tests get_ground_state functionality."""
        target_val = min([self.ce.predict(structure)
                          for structure in self.all_possible_structures])

        # Provide counts for first species
        species_count = {self.chemical_symbols[0]: len(self.supercell) - 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species0 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, target_val)

        # Provide counts for second species
        species_count = {self.chemical_symbols[1]: 1}
        ground_state = self.gsf.get_ground_state(species_count=species_count, threads=1)
        predicted_species1 = self.ce.predict(ground_state)
        self.assertEqual(predicted_species0, predicted_species1)


if __name__ == '__main__':
    unittest.main()
