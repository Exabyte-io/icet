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

from collections import OrderedDict
from io import StringIO
import inspect
import os
import sys
import tempfile
import unittest

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111
from ase.db import connect as db_connect
from icet import ClusterSpace
from icet.core.cluster_space import (get_singlet_info,
                                     get_singlet_configuration)
from icet.core.lattice_site import LatticeSite


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


class TestClusterSpace(unittest.TestCase):
    """Container for test of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterSpace, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Ag', 'Au']
        self.cutoffs = [4.0] * 3
        self.atoms_prim = bulk('Ag', a=4.09)
        self.structure_list = []
        for k in range(4):
            atoms = self.atoms_prim.repeat(2)
            symbols = [self.chemical_symbols[0]] * len(atoms)
            symbols[:k] = [self.chemical_symbols[1]] * k
            atoms.set_chemical_symbols(symbols)
            self.structure_list.append(atoms)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.cs = ClusterSpace(self.atoms_prim, self.cutoffs,
                               self.chemical_symbols)

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ASE Atoms
        cs = ClusterSpace(self.atoms_prim, self.cutoffs, self.chemical_symbols)
        self.assertIsInstance(cs, ClusterSpace)
        self.assertEqual(len(cs), len(self.cs))

    def test_len(self):
        """Tests length functionality."""
        number_orbits = self.cs.__len__()
        self.assertEqual(number_orbits, len(self.cs._get_orbit_list()) + 1)

    def test_orbit_data(self):
        """Tests orbit_data property."""
        target = [OrderedDict([('index', 0),
                               ('order', 0),
                               ('radius', 0),
                               ('multiplicity', 1),
                               ('orbit_index', -1)]),
                  OrderedDict([('index', 1), ('order', 1),
                               ('radius', 0.0),
                               ('multiplicity', 1),
                               ('orbit_index', 0),
                               ('multi_component_vector', [0])]),
                  OrderedDict([('index', 2), ('order', 2),
                               ('radius', 1.4460333675264896),
                               ('multiplicity', 6),
                               ('orbit_index', 1),
                               ('multi_component_vector', [0, 0])]),
                  OrderedDict([('index', 3), ('order', 3),
                               ('radius', 1.6697355079971996),
                               ('multiplicity', 8),
                               ('orbit_index', 2),
                               ('multi_component_vector', [0, 0, 0])]),
                  OrderedDict([('index', 4), ('order', 4),
                               ('radius', 1.771021950739177),
                               ('multiplicity', 2),
                               ('orbit_index', 3),
                               ('multi_component_vector', [0, 0, 0, 0])])]
        self.assertEqualComplexList(self.cs.orbit_data, target)

    def test_repr(self):
        """Tests string representation functionality."""
        retval = self.cs.__repr__()
        target = """
=============================== Cluster Space ================================
 chemical species: ['Ag', 'Au']
 cutoffs: 4.0000 4.0000 4.0000
 total number of orbits: 5
 number of orbits by order: 0= 1  1= 1  2= 1  3= 1  4= 1
------------------------------------------------------------------------------
index | order |  radius  | multiplicity | orbit_index | multi_component_vector
------------------------------------------------------------------------------
   0  |   0   |   0.0000 |        1     |      -1     |           .
   1  |   1   |   0.0000 |        1     |       0     |          [0]
   2  |   2   |   1.4460 |        6     |       1     |         [0, 0]
   3  |   3   |   1.6697 |        8     |       2     |       [0, 0, 0]
   4  |   4   |   1.7710 |        2     |       3     |      [0, 0, 0, 0]
==============================================================================
"""
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_get_string_representation(self):
        """Tests _get_string_representation functionality."""
        retval = self.cs._get_string_representation(print_threshold=2,
                                                    print_minimum=1)
        target = """
=============================== Cluster Space ================================
 chemical species: ['Ag', 'Au']
 cutoffs: 4.0000 4.0000 4.0000
 total number of orbits: 5
 number of orbits by order: 0= 1  1= 1  2= 1  3= 1  4= 1
------------------------------------------------------------------------------
index | order |  radius  | multiplicity | orbit_index | multi_component_vector
------------------------------------------------------------------------------
   0  |   0   |   0.0000 |        1     |      -1     |           .
 ...
   4  |   4   |   1.7710 |        2     |       3     |      [0, 0, 0, 0]
==============================================================================
"""
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_print_overview(self):
        """Tests print_overview functionality."""
        with StringIO() as capturedOutput:
            sys.stdout = capturedOutput  # redirect stdout
            self.cs.print_overview()
            sys.stdout = sys.__stdout__  # reset redirect
            self.assertTrue('Cluster Space' in capturedOutput.getvalue())

    def test_get_number_of_orbits_by_order(self):
        """Tests get_number_of_orbits_by_order functionality """
        retval = self.cs.get_number_of_orbits_by_order()
        target = OrderedDict([(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
        self.assertEqual(target, retval)

    def test_get_cluster_vector(self):
        """Tests get_cluster_vector functionality."""
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
        for atoms, target in zip(self.structure_list, target_cluster_vectors):
            retval = list(self.cs.get_cluster_vector(atoms))
            self.assertAlmostEqual(retval, target, places=9)

    def test_get_singlet_info(self):
        """Tests get_singlet_info functionality."""
        retval = get_singlet_info(self.structure_list[0])
        target = [{'orbit_index': 0,
                   'sites': [[LatticeSite(0, [0., 0., 0.])]],
                   'multiplicity': 1,
                   'representative_site': [LatticeSite(0, [0., 0., 0.])]}]
        self.assertEqualComplexList(retval, target)
        retval1, retval2 = get_singlet_info(self.structure_list[0],
                                            return_cluster_space=True)
        self.assertEqualComplexList(retval1, target)
        self.assertIsInstance(retval2, type(self.cs))

    def test_get_singlet_configuration(self):
        """Tests get_singlet_configuration functionality."""
        retval = get_singlet_configuration(self.atoms_prim)
        self.assertIsInstance(retval, Atoms)
        self.assertEqual(retval[0].symbol, 'H')
        retval = get_singlet_configuration(self.structure_list[0],
                                           to_primitive=True)
        self.assertIsInstance(retval, Atoms)
        self.assertEqual(len(retval), len(self.atoms_prim))

    def test_cutoffs(self):
        """Tests cutoffs property."""
        self.assertEqual(self.cs.cutoffs, self.cutoffs)

    def _test_cluster_vectors_in_database(self, db_name):
        """Tests the cluster vectors in the database."""

        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path = os.path.dirname(os.path.abspath(filename))
        db = db_connect(os.path.join(path, db_name))

        entry1 = db.get(id=1)
        atoms = entry1.toatoms()
        elements = entry1.data.elements
        cutoffs = entry1.data.cutoffs
        cs = ClusterSpace(atoms, cutoffs, elements)

        for row in db.select():
            atoms = row.toatoms()
            retval = cs.get_cluster_vector(atoms)
            target = np.array(row.data.target_cv)
            self.assertTrue(np.all(np.isclose(target, retval)))

    def test_cluster_vectors(self):
        """
        Tests the calculation of cluster vectors against databases
        of structures with known cluster vectors.
        """
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/fcc_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/fcc_skew_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/fcc_ternary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/fcc_quaternary.db')

        self._test_cluster_vectors_in_database(
            '../../../structure_databases/bcc_longedge_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/bcc_ternary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/bcc_quaternary.db')

        self._test_cluster_vectors_in_database(
            '../../../structure_databases/hcp_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/hcp_skew_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/hcp_ternary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/hcp_quaternary.db')

        self._test_cluster_vectors_in_database(
            '../../../structure_databases/tetragonal_binary.db')
        self._test_cluster_vectors_in_database(
            '../../../structure_databases/tetragonal_ternary.db')

    def test_read_write(self):
        """Tests read/write functionality."""
        f = tempfile.NamedTemporaryFile()
        self.cs.write(f.name)
        f.seek(0)
        cs_read = ClusterSpace.read(f.name)
        self.assertEqual(self.cs._atoms, cs_read._atoms)
        self.assertEqual(list(self.cs._cutoffs), list(cs_read._cutoffs))
        self.assertEqual(self.cs._chemical_symbols, cs_read._chemical_symbols)

    def test_chemical_symbols(self):
        """Tests chemical_symbols property."""
        target = [['Ag', 'Au']]
        self.assertEqual(self.cs.chemical_symbols, target)


class TestClusterSpaceSurface(unittest.TestCase):
    """
    Container for tests of the class functionality for non-periodic structures.
    """

    def __init__(self, *args, **kwargs):
        super(TestClusterSpaceSurface, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Ag', 'Au']
        self.cutoffs = [4.0] * 3
        self.atoms_prim = fcc111('Ag', a=4.09, vacuum=5.0, size=[1, 1, 3])
        self.atoms_prim.pbc = [True, True, False]
        self.structure_list = []
        for k in range(3):
            atoms = self.atoms_prim.repeat((2, 2, 1))
            symbols = [self.chemical_symbols[0]] * len(atoms)
            symbols[:k] = [self.chemical_symbols[1]] * k
            atoms.set_chemical_symbols(symbols)
            self.structure_list.append(atoms)

    def setUp(self):
        """
        Instantiates class before each test.
        """
        self.cs = ClusterSpace(self.atoms_prim, self.cutoffs,
                               self.chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    @unittest.expectedFailure
    def test_get_cluster_vector(self):
        """Tests get_cluster_vector functionality."""
        target_cluster_vectors = np.array([
            [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
             -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            [1., -0.5, -1., -1., 0., 0.5, 1., 1., 1., 0.5, 0.5,
             -1., -1., -1., -1., 0., -0.5, -1., -1., -0.5, 0.5, 1.,
             1.],
            [1., -0.5, -0.5,
             -1., 0., 0.3333333333,
             0., 0.5, 1.,
             0.5, 0.5, 0.5,
             0.5, -1., -1.,
             -0.1666666667, -0.1666666667, 0.,
             -0.5, 0., 0.,
             0.5, -0.5]])
        s = ['Error in test setup;']
        s += ['number of cluster vectors ({})'.format(
            len(target_cluster_vectors))]
        s += ['does not match']
        s += ['number of structures ({})'.format(len(self.structure_list))]
        info = ' '.join(s)
        self.assertEqual(len(target_cluster_vectors), len(self.structure_list),
                         msg=info)
        for atoms, target in zip(self.structure_list, target_cluster_vectors):
            retval = self.cs.get_cluster_vector(atoms)
            self.assertAlmostEqualList(retval, target, places=9)

    def test_get_number_of_orbits_by_order(self):
        """Tests get_number_of_orbits_by_order functionality."""
        retval = self.cs.get_number_of_orbits_by_order()
        target = OrderedDict([(0, 1), (1, 3), (2, 5), (3, 10), (4, 4)])
        self.assertEqual(target, retval)


class TestClusterSpaceTernary(unittest.TestCase):
    """
    Container for tests of the class functionality for non-periodic structures
    """

    def __init__(self, *args, **kwargs):
        super(TestClusterSpaceTernary, self).__init__(*args, **kwargs)
        self.chemical_symbols = ['Ag', 'Au', 'Pd']
        self.cutoffs = [4.0] * 3
        self.atoms_prim = bulk('Ag', 'fcc')

    def setUp(self):
        """Instantiates class before each test."""
        self.cs = ClusterSpace(self.atoms_prim, self.cutoffs,
                               self.chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def _get_mc_vector(self, cluster_space, orbit_index):
        """
        Helper function to  return the mc vectors for a
        particular orbit.

        Parameters
        ----------
        cluster_space : icet cluster space
        orbit_index : int
            The orbit which the mc vectors should be returned from.
        """
        orbit = cluster_space.get_orbit(orbit_index)
        local_Mi = \
            cluster_space.get_number_of_allowed_species_by_site(
                cluster_space._get_primitive_structure(),
                orbit.representative_sites)

        mc_vectors = orbit.get_mc_vectors(local_Mi)
        return mc_vectors

    def test_multi_component_cluster_vector_permutation(self):
        """Tests the multicomponent permutation functionality."""
        # Test orbit number 1
        orbit_index = 1
        mc_vector_target = [[0, 0], [0, 1], [1, 1]]
        mc_vector_retval = self._get_mc_vector(self.cs, orbit_index)
        self.assertEqual(mc_vector_retval, mc_vector_target)

        permutations_target = [[[0, 1]],
                               [[0, 1], [1, 0]],
                               [[0, 1]]]
        permutation_retval = self.cs.get_multi_component_vector_permutations(
            mc_vector_target, orbit_index)
        self.assertEqual(permutations_target, permutation_retval)

        # Test orbit number 2
        orbit_index = 2
        mc_vector_target = [[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 1],
                            [1, 1, 1]]
        mc_vector_retval = self._get_mc_vector(self.cs, orbit_index)
        self.assertEqual(mc_vector_retval, mc_vector_target)

        permutations_target = [[[0, 1, 2]],
                               [[0, 1, 2], [0, 2, 1], [2, 1, 0]],
                               [[0, 1, 2]],
                               [[0, 1, 2]]]
        permutations_target = [[[0, 1, 2]],
                               [[0, 1, 2], [1, 2, 0], [2, 1, 0]],
                               [[0, 1, 2], [2, 0, 1], [2, 1, 0]],
                               [[0, 1, 2]]]
        permutation_retval = self.cs.get_multi_component_vector_permutations(
            mc_vector_target, orbit_index)
        self.assertEqual(permutations_target, permutation_retval)

        # Test orbit 3
        orbit_index = 3

        mc_vector_target = [[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1]]
        mc_vector_retval = self._get_mc_vector(self.cs, orbit_index)
        self.assertEqual(mc_vector_retval, mc_vector_target)
        permutations_target = [[[0, 1, 2, 3]],
                               [[0, 1, 2, 3],
                                [2, 1, 3, 0], [2, 3, 1, 0],
                                [3, 1, 2, 0]],
                               [[0, 1, 2, 3],
                                [0, 3, 1, 2], [
                                   1, 2, 3, 0],
                                [2, 0, 1, 3], [
                                   2, 3, 1, 0],
                                [3, 1, 2, 0]],
                               [[0, 1, 2, 3],
                                [2, 0, 3, 1], [
                                   2, 3, 1, 0],
                                [3, 2, 0, 1]],
                               [[0, 1, 2, 3]]]
        permutation_retval = self.cs.get_multi_component_vector_permutations(
            mc_vector_target, orbit_index)
        self.assertEqual(permutations_target, permutation_retval)


class TestClusterSpaceMultiSublattice(unittest.TestCase):
    """Container for test of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterSpaceMultiSublattice, self).__init__(*args, **kwargs)
        self.chemical_symbols = [['Ag', 'Au'],
                                 ['H', 'V']]
        self.cutoffs = [5] * 2
        self.atoms_prim = bulk(
            'Ag', a=4.09, crystalstructure='bcc', cubic=True).repeat([1, 1, 1])

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.cs = ClusterSpace(self.atoms_prim, self.cutoffs,
                               self.chemical_symbols)
        self.cluster_space_binary = ClusterSpace(
            self.atoms_prim, self.cutoffs, ['Ag', 'Au'])

    def test_init(self):
        """Tests that initialization of tested class work."""
        # initialize from ASE Atoms
        cs = ClusterSpace(self.atoms_prim, self.cutoffs, self.chemical_symbols)
        self.assertIsInstance(cs, ClusterSpace)
        self.assertEqual(len(cs), len(self.cs))

    def test_correct_number_of_singlets(self):
        """Tests that we get two singlets."""
        singlet_count = 0
        for orbit in self.cs.orbit_list.orbits:
            if orbit.order == 1:
                singlet_count += 1
        self.assertEqual(singlet_count, 2)

    def test_correct_number_of_pairs(self):
        """Tests that we get correct number of pairs."""
        pair_counts = OrderedDict()
        pair_counts_binary = OrderedDict()
        for orbit in self.cs.orbit_list.orbits:
            if orbit.order == 2:
                radius = np.round(orbit.representative_cluster.radius, 3)
                if radius in pair_counts.keys():
                    pair_counts[radius] += 1
                else:
                    pair_counts[radius] = 1

        for orbit in self.cluster_space_binary.orbit_list.orbits:
            if orbit.order == 2:
                radius = np.round(orbit.representative_cluster.radius, 3)
                if radius in pair_counts_binary.keys():
                    pair_counts_binary[radius] += 1
                else:
                    pair_counts_binary[radius] = 1

        self.assertEqual(len(pair_counts.keys()),
                         len(pair_counts_binary.keys()))

        # origin to center atom only one since these are only
        # sublattice 1 -> sublattice 2 interactions
        self.assertEqual(pair_counts_binary[1.771], pair_counts[1.771])

        # Twice as many pairs in 100 direction since they can be both
        # sublattice-1 -> sublatice 1 and sublattice 2  -> sublattice 2
        self.assertEqual(pair_counts_binary[2.045] * 2, pair_counts[2.045])


if __name__ == '__main__':
    unittest.main()
