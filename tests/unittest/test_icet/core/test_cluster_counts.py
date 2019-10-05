#!/usr/bin/env python

import unittest

from ase.build import bulk
from icet import Structure
from icet.core.cluster import Cluster
from icet.core.orbit_list import OrbitList
from icet.core.cluster_counts import ClusterCounts
from icet.core.lattice_site import LatticeSite
from io import StringIO


def strip_surrounding_spaces(input_string):
    """
    Removes both leading and trailing spaces from a multi-line string.

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


class TestClusterCounts(unittest.TestCase):
    """ Container for test of the module functionality. """

    def __init__(self, *args, **kwargs):
        super(TestClusterCounts, self).__init__(*args, **kwargs)
        self.structure = bulk('Ni', 'hcp', a=2.0).repeat([2, 1, 1])
        self.structure_prim = bulk('Ni', 'hcp', a=2.0)
        self.structure.set_chemical_symbols('NiFeNi2')
        self.icet_structure = Structure.from_atoms(self.structure)
        self.cutoffs = [2.2]
        self.orbit_list = OrbitList(self.structure_prim, self.cutoffs)
        self.orbit_list.sort()

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """ Sets up an empty cluster counts object. """
        self.cluster_counts = ClusterCounts(self.orbit_list, self.structure)

    def test_count_lattice_sites(self):
        """
        Tests cluster_counts counts a pair given a set of
        lattice neighbors.
        """
        lattice_sites = []
        lattice_sites.append(LatticeSite(0, [0., 0., 0.]))
        lattice_sites.append(LatticeSite(1, [0., 0., 0.]))

        cluster = Cluster(self.icet_structure, lattice_sites)

        self.cluster_counts.count(self.icet_structure, lattice_sites)
        cluster_map = self.cluster_counts.get_cluster_counts()

        count = cluster_map[cluster]
        self.assertEqual(count, {('Fe', 'Ni'): 1})

    def test_count_list_lattice_sites(self):
        """
        Tests whether cluster_counts returns the correct number of pairs
        given a list of lattice neighbors.
        """
        lattice_sites = []
        lattice_sites.append(LatticeSite(0, [0., 0., 0.]))
        lattice_sites.append(LatticeSite(1, [0., 0., 0.]))

        lattice_sites2 = []
        lattice_sites2.append(LatticeSite(0, [0., 0., 0.]))
        lattice_sites2.append(LatticeSite(2, [0., 0., 0.]))

        cluster = Cluster(self.icet_structure, lattice_sites)

        lattice_neighbors = [lattice_sites, lattice_sites2]

        self.cluster_counts.count(self.icet_structure, lattice_neighbors,
                                  cluster, True)
        cluster_map = self.cluster_counts.get_cluster_counts()

        count = cluster_map[cluster]
        self.assertEqual(count, {('Ni', 'Fe'): 1, ('Ni', 'Ni'): 1})

    def test_count_orbit_list(self):
        """Tests cluster_counts given orbits in an orbit list."""
        cluster_singlet = Cluster(self.icet_structure, [], False, 0)
        cluster_pair = Cluster(self.icet_structure, [], False, 1)
        clusters = [cluster_singlet, cluster_pair]

        expected_counts = [{('Fe',): 1, ('Ni',): 3},
                           {('Fe', 'Fe'): 1, ('Fe', 'Ni'): 4,
                            ('Ni', 'Ni'): 7}]

        for k, cluster in enumerate(clusters):
            count = self.cluster_counts[cluster]
            self.assertEqual(count, expected_counts[k])

    @unittest.expectedFailure
    def test_count_orbit_list_non_pbc(self):
        """Tests cluster counts using orbit_list for a non-pbc structure."""
        structure_non_pbc = self.structure.copy()
        structure_non_pbc.set_pbc(False)
        orbit_list = OrbitList(structure_non_pbc, self.cutoffs)

        cluster_singlet = Cluster(self.icet_structure, [], False, 0)
        cluster_pair = Cluster(self.icet_structure, [], False, 1)
        clusters = [cluster_singlet, cluster_pair]

        expected_counts = [{('Fe',): 1, ('Ni',): 3},
                           {('Fe', 'Ni'): 2, ('Ni', 'Ni'): 2}]

        cluster_counts = ClusterCounts(orbit_list, structure_non_pbc)

        for k, cluster in enumerate(clusters):
            count = cluster_counts[cluster]
            self.assertEqual(count, expected_counts[k])

    def test_len(self):
        """Tests total size of counts."""
        self.assertEqual(len(self.cluster_counts),
                         len(self.orbit_list))

    def test_reset(self):
        """Tests reset functionality."""
        self.cluster_counts.reset()
        self.assertEqual(len(self.cluster_counts), 0)

    def test_getitem(self):
        """Tests __getitem__ functionality."""
        # Test with integer as key
        self.assertEqual(self.cluster_counts[2],
                         {('Fe', 'Ni'): 6, ('Ni', 'Ni'): 6})
        # Test with icet Cluster as key
        cluster = list(self.cluster_counts.cluster_counts.keys())[0]
        self.assertEqual(self.cluster_counts[cluster], {
                         ('Fe',): 1, ('Ni',): 3})

    def test_str(self):
        """Tests representation of cluster_counts."""
        retval = self.cluster_counts.__str__()
        target = """
====================== Cluster Counts ======================
Singlet: [0] [] 0.0000
Fe   1
Ni   3

Pair: [0, 0] [2.0] 1.0000
Fe  Fe   1
Fe  Ni   4
Ni  Ni   7

Pair: [0, 0] [2.0] 1.0000
Fe  Ni   6
Ni  Ni   6
============================================================
"""
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_get_cluster_counts(self):
        """Tests get_cluster_counts functionality."""
        counts = self.cluster_counts.get_cluster_counts()
        for cluster, cluster_info in counts.items():
            # check size of cluster match the number of elements in counts
            for elements in cluster_info:
                self.assertEqual(len(cluster), len(elements))


if __name__ == '__main__':
    unittest.main()
