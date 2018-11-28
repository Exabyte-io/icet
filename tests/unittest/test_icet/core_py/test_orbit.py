#!/usr/bin/env Python3


import unittest

from icet.core_py.orbit import Orbit
from icet.core_py.lattice_site import LatticeSite
from icet.core.lattice_site import LatticeSite as LatticeSite_cpp
from icet.core.cluster import Cluster
from icet.core.orbit import Orbit as Orbit_cpp
from ase.build import bulk
import itertools
import numpy as np


class TestOrbit(unittest.TestCase):
    """
    Container for tests of the class functionality

    TODO
    ---
    *   revise test test_property_radius when
        cluster is in orbit and the geometrical size can
        be retrieved from the orbit.
    """

    def __init__(self, *args, **kwargs):
        super(TestOrbit, self).__init__(*args, **kwargs)

        self.lattice_sites_pairs = []
        indices = [i for i in range(8)]
        unitcell_offsets = []
        cartesian_product_lists = [[0., 1.], [0., 1.], [0., 1.]]
        for element in itertools.product(*cartesian_product_lists):
            unitcell_offsets.append(list(element))
        self.lattice_sites_pairs = [[LatticeSite(index, unitcell_offset),
                                     LatticeSite(index + 1, unitcell_offset)]
                                    for index, unitcell_offset in
                                    zip(indices, unitcell_offsets)]
        self.lattice_sites_triplets = [[LatticeSite(index, unitcell_offset),
                                        LatticeSite(
                                            index + 1, unitcell_offset),
                                        LatticeSite(
                                            index + 3, unitcell_offset)]
                                       for index, unitcell_offset in
                                       zip(indices, unitcell_offsets)]
        self.lattice_sites_pairs_cpp = [[LatticeSite_cpp(index,
                                                         unitcell_offset),
                                         LatticeSite_cpp(index + 1,
                                                         unitcell_offset)]
                                        for index, unitcell_offset in
                                        zip(indices, unitcell_offsets)]

        self.lattice_sites_triplets_cpp = [[LatticeSite_cpp(index,
                                                            unitcell_offset),
                                            LatticeSite_cpp(
                                            index + 1, unitcell_offset),
                                            LatticeSite_cpp(
                                            index + 3, unitcell_offset)]
                                           for index, unitcell_offset in
                                           zip(indices, unitcell_offsets)]

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiates class before each test."""
        atoms = bulk("Al")
        lattice_site_for_cluster = [
            LatticeSite(0, [i, 0, 0]) for i in range(3)]

        self.pair_cluster = Cluster.from_python(
            atoms, [lattice_site_for_cluster[0],
                    lattice_site_for_cluster[1]], True)
        self.triplet_cluster = Cluster.from_python(
            atoms, lattice_site_for_cluster, True)

        self.orbit_pair = Orbit(self.pair_cluster)
        self.orbit_triplet = Orbit(self.triplet_cluster)

        self.orbit_pair_cpp = Orbit_cpp(self.pair_cluster)
        self.orbit_triplet_cpp = Orbit_cpp(self.triplet_cluster)

    def test_init(self):
        """Tests that initialization of tested class works."""
        # initialize from ASE Atoms
        orbit = Orbit(self.pair_cluster)
        self.assertIsInstance(orbit, Orbit)

    def test_property_equivalent_sites(self):
        """Tests equivalent_sites property."""
        self.assertEqual(self.orbit_pair.equivalent_sites, [])
        self.orbit_pair.equivalent_sites = self.lattice_sites_pairs
        self.assertEqual(self.orbit_pair.equivalent_sites,
                         self.lattice_sites_pairs)

        # Test C++ and Python together
        self.orbit_pair_cpp.equivalent_sites = self.lattice_sites_pairs_cpp
        self.assertEqual(
            self.orbit_pair_cpp.equivalent_sites,
            self.orbit_pair.equivalent_sites)

    def test_property_representative_cluster(self):
        """
        Tests retrieving the representative cluster.

        TODO
        ----
        * implement this more thoroughly when cluster is added
        """
        self.orbit_pair.representative_cluster

    def test_property_representative_sites(self):
        """Tests getting the representative sites."""
        with self.assertRaises(IndexError):
            self.orbit_pair.representative_sites

        self.orbit_pair.equivalent_sites = self.lattice_sites_pairs
        self.assertEqual(self.orbit_pair.representative_sites,
                         self.lattice_sites_pairs[0])

        # Test C++ and Python together
        self.orbit_pair_cpp.equivalent_sites = self.lattice_sites_pairs_cpp
        self.assertEqual(self.orbit_pair_cpp.representative_sites,
                         self.orbit_pair.representative_sites)

    def test_property_order(self):
        """Tests getting the order from an orbit."""
        with self.assertRaises(IndexError):
            self.orbit_pair.order

        self.orbit_pair.equivalent_sites = self.lattice_sites_pairs
        self.orbit_triplet.equivalent_sites = self.lattice_sites_triplets
        self.assertEqual(self.orbit_pair.order,
                         2)
        self.assertEqual(self.orbit_triplet.order,
                         3)

    def test_len(self):
        """Tests length of orbit."""
        self.assertEqual(len(self.orbit_pair), 0)
        self.orbit_pair.equivalent_sites = self.lattice_sites_pairs
        self.assertEqual(len(self.orbit_pair), len(self.lattice_sites_pairs))

    def test_property_radius(self):
        """Tests the property geometrical size."""
        self.orbit_pair.radius

    def test_sort(self):
        """
        Tests the sort functionality.

        The test will set the equivalent sites
        as the the reversed sorted list of lattice sites.
        Then it will call the orbit.sort(), i.e. the function
        we are testing and then assert that the property
        equivalent sites are equivalent to the sorted list of
        lattice sites.
        """

        self.orbit_pair.equivalent_sites = sorted(self.lattice_sites_pairs,
                                                  reverse=True)
        self.orbit_pair.sort()
        self.assertEqual(self.orbit_pair.equivalent_sites,
                         sorted(self.lattice_sites_pairs))

        # sorted orbit from C++ version
        self.orbit_pair_cpp.equivalent_sites = sorted(
            self.lattice_sites_pairs_cpp, reverse=True)
        self.orbit_pair_cpp.sort()

        # Test C++ and Python together
        self.assertEqual(
            self.orbit_pair_cpp.equivalent_sites,
            self.orbit_pair.equivalent_sites)

    def test_eq(self):
        """Tests equality functionality."""
        self.orbit_pair.equivalent_sites = self.lattice_sites_pairs
        self.orbit_triplet.equivalent_sites = self.lattice_sites_triplets

        self.assertEqual(self.orbit_pair, self.orbit_pair)
        self.assertEqual(self.orbit_triplet, self.orbit_triplet)
        self.assertNotEqual(self.orbit_pair, self.orbit_triplet)
        self.assertNotEqual(self.orbit_triplet, self.orbit_pair)

    def test_lt(self):
        """Tests less than functionality."""
        self.orbit_pair.equivalent_sites = self.lattice_sites_pairs
        self.orbit_triplet.equivalent_sites = self.lattice_sites_triplets
        self.assertFalse(self.orbit_pair < self.orbit_pair)
        self.assertFalse(self.orbit_triplet < self.orbit_triplet)

        self.assertTrue(self.orbit_pair < self.orbit_triplet)
        self.assertTrue(self.orbit_triplet > self.orbit_pair)

    def test_add(self):
        """
        Tests add operator.

        This test the orbit_offset = orbit + offset
        functionality.

        It will test this by creating
        orbit = self.orbit_pair + added_offset
        and then loop through the equivalent sites
        to see that all lattice sites offset have been
        this added_offset added to it.
        There are also some assertions that you can not add
        a list or something that is not a three long list
        """
        # Test python
        added_offset = np.array((1., 1., 1.))
        self.orbit_pair.equivalent_sites = self.lattice_sites_pairs
        # create a new orbit with offseted equivalent lattice sites
        orbit = self.orbit_pair + added_offset
        self.assertNotEqual(orbit, self.orbit_pair)

        # Loop through sites check index is same and offset has been added
        for sites, sites_with_offset in zip(self.orbit_pair.equivalent_sites,
                                            orbit.equivalent_sites):
            for site, site_with_offset in zip(sites, sites_with_offset):
                self.assertEqual(site.index, site_with_offset.index)
                self.assertListEqual(list(site.unitcell_offset),
                                     list(site_with_offset.unitcell_offset -
                                          added_offset))

        # offset as list or array not of size 3 is not allowed
        with self.assertRaises(TypeError):
            orbit + [1, 1, 1]
        with self.assertRaises(TypeError):
            orbit + np.array([1, 1, 1, 1])

        # Test C++
        added_offset = np.array((1., 1., 1.))
        self.orbit_pair_cpp.equivalent_sites =\
            self.lattice_sites_pairs_cpp
        # create a new orbit with offseted equivalent lattice sites
        orbit_cpp = self.orbit_pair_cpp + added_offset
        self.assertNotEqual(orbit_cpp, self.orbit_pair_cpp)

        # Test C++ and Python together
        self.assertEqual(orbit_cpp.equivalent_sites,
                         orbit.equivalent_sites)

    def test_property_permutations_to_representative(self):
        """Tests the permutations to representative property."""
        allowed_permutations = [[1, 2, 3]]

        self.assertEqual(self.orbit_pair.permutations_to_representative, [])
        self.orbit_pair.permutations_to_representative = allowed_permutations
        self.assertListEqual(
            self.orbit_pair.permutations_to_representative,
            allowed_permutations)

        # This functionality is not exposed to python
    def test_property_allowed_permutations(self):
        """Tests the allowed permutations property."""
        allowed_permutations = [[1, 2, 3]]
        self.assertEqual(self.orbit_pair.allowed_permutations, [])
        self.orbit_pair.allowed_permutations = allowed_permutations
        self.assertListEqual(
            self.orbit_pair.allowed_permutations, allowed_permutations)

        self.orbit_pair_cpp.allowed_permutations = allowed_permutations

        # Test C++ and Python together
        self.assertEqual(self.orbit_pair_cpp.allowed_permutations,
                         self.orbit_pair.allowed_permutations)

    def test_property_permuted_sites(self):
        """Tests the permuted sites property."""
        self.orbit_pair.equivalent_sites = self.lattice_sites_pairs
        # Raises IndexError when permutations to primitive is not set
        with self.assertRaises(IndexError):
            self.orbit_pair.permuted_sites

        # Provide the identity permutation
        self.orbit_pair.permutations_to_representative = [
            [i for i in range(self.orbit_pair.order)]] * len(self.orbit_pair)

        self.assertEqual(self.orbit_pair.permuted_sites,
                         self.orbit_pair.equivalent_sites)

        # Provide a completely reversed permutation, [i,j,k] ->[k,j,i]
        self.orbit_pair.permutations_to_representative = [
            [i for i in reversed(range(self.orbit_pair.order))]] * \
            len(self.orbit_pair)

        for perm_sites, sites in zip(self.orbit_pair.permuted_sites,
                                     self.orbit_pair.equivalent_sites):
            self.assertEqual(perm_sites, list(reversed(sites)))

    def test_get_mc_vectors_pairs(self):
        """Tests  the get mc vectors functionality for a pair orbit."""
        self.orbit_pair.equivalent_sites = self.lattice_sites_pairs
        # Binary mc vectors
        # Allow only identity permutation
        self.orbit_pair.allowed_permutations = [
            [i for i in range(self.orbit_pair.order)]]
        mc_vectors = self.orbit_pair.get_mc_vectors(
            [2] * self.orbit_pair.order)
        self.assertEqual(mc_vectors, [(0, 0)])

        # Ternary mc vectors
        mc_vectors = self.orbit_pair.get_mc_vectors(
            [3] * self.orbit_pair.order)
        target = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.assertEqual(mc_vectors, target)

        # Allow the permutation [1,0] permutation
        self.orbit_pair.allowed_permutations = ([0, 1], [1, 0])
        mc_vectors = self.orbit_pair.get_mc_vectors(
            [3] * self.orbit_pair.order)
        target = [(0, 0), (0, 1), (1, 1)]
        self.assertEqual(mc_vectors, target)

    def test_get_mc_vectors_triplets(self):
        """Tests  the get mc vectors functionality for a triplet orbit."""
        self.orbit_pair.equivalent_sites = self.lattice_sites_triplets
        # Binary mc vectors
        # Allow only identity permutation
        self.orbit_pair.allowed_permutations = [
            [i for i in range(self.orbit_pair.order)]]

        # Ternary mc vectors
        mc_vectors = self.orbit_pair.get_mc_vectors(
            [3] * self.orbit_pair.order)
        target = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                  (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        self.assertEqual(mc_vectors, target)

        # Allow the permutation [0,2,1] permutation
        self.orbit_pair.allowed_permutations = ([0, 1, 2], [0, 2, 1])
        mc_vectors = self.orbit_pair.get_mc_vectors(
            [3] * self.orbit_pair.order)
        target = [(0, 0, 0), (0, 0, 1), (0, 1, 1),
                  (1, 0, 0), (1, 0, 1), (1, 1, 1)]
        self.assertEqual(mc_vectors, target)


if __name__ == '__main__':
    unittest.main()
