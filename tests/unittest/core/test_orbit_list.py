import unittest
from itertools import permutations
from ase.build import bulk
from ase.db import connect as ase_connect

from icet.core.lattice_site import LatticeSite
from icet.core.cluster import Cluster
from icet.core.orbit import Orbit
from icet import OrbitList
from icet import Structure
from icet.tools.geometry import get_permutation
from icet.core.permutation_matrix import (_get_lattice_site_permutation_matrix,
                                          permutation_matrix_from_atoms)


class TestOrbitList(unittest.TestCase):
    """Container for test of the module functionality."""

    def __init__(self, *args, **kwargs):
        super(TestOrbitList, self).__init__(*args, **kwargs)
        self.cutoffs = [4.2]
        self.atoms = bulk('Ag', 'sc', a=4.09)

        # representative clusters for testing
        structure = Structure.from_atoms(self.atoms)
        # for singlet
        self.cluster_singlet = Cluster(
            structure, [LatticeSite(0, [0, 0, 0])])
        # for pair
        lattice_sites = [LatticeSite(0, [i, 0, 0]) for i in range(3)]
        self.cluster_pair = Cluster(
            structure, [lattice_sites[0], lattice_sites[1]], True)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiate class before each test."""
        self.orbit_list = OrbitList(self.atoms, self.cutoffs)

    def test_init(self):
        """Test the different initializers."""
        orbit_list = OrbitList(
            self.atoms, self.cutoffs)
        self.assertIsInstance(orbit_list, OrbitList)

    def test_property_permutation_matrix(self):
        """Tests permutation matrix property."""
        permutation_matrix, prim_structure, _ = \
            permutation_matrix_from_atoms(self.atoms, self.cutoffs[0])
        pm_lattice_site = _get_lattice_site_permutation_matrix(
            prim_structure, permutation_matrix, prune=True)

        self.assertEqual(self.orbit_list.permutation_matrix, pm_lattice_site)

    def test_add_orbit(self):
        """Tests add_orbit funcionality."""
        orbit = Orbit(self.cluster_pair)
        self.orbit_list.add_orbit(orbit)
        self.assertEqual(len(self.orbit_list), 3)

    def test_get_number_of_NClusters(self):
        """Tests that only a pair is counted in the orbit list."""
        NPairs = self.orbit_list.get_number_of_NClusters(2)
        self.assertEqual(NPairs, 1)

    def test_get_orbit(self):
        """Tests function returns the number of orbits of a given order."""
        # get singlet
        orbit = self.orbit_list.get_orbit(0)
        self.assertEqual(orbit.order, 1)
        # get pair
        orbit = self.orbit_list.get_orbit(1)
        self.assertEqual(orbit.order, 2)
        # check higher order raises an error
        with self.assertRaises(IndexError):
            self.orbit_list.get_orbit(3)

    def test_clear(self):
        """Tests orbit list is empty after calling this function."""
        self.orbit_list.clear()
        with self.assertRaises(IndexError):
            self.orbit_list.get_orbit(0)

    def test_sort(self):
        """Tests orbits in orbit list are sorted."""
        self.orbit_list.sort()
        for i in range(len(self.orbit_list) - 1):
            self.assertLess(
                self.orbit_list.get_orbit(i), self.orbit_list.get_orbit(i + 1))

    def test_find_orbit(self):
        """Tests orbit index returned for the given representative cluster."""
        # TODO: test a non-representative cluster returns -1
        self.assertEqual(
            self.orbit_list._find_orbit(self.cluster_singlet), 0)
        self.assertEqual(
            self.orbit_list._find_orbit(self.cluster_pair), 1)

    def test_is_row_taken(self):
        """Tests is_row_taken (private) functionality."""
        taken_rows = set()
        row_indices = tuple([0, 1, 2])
        self.assertFalse(self.orbit_list._is_row_taken(
            taken_rows, row_indices))

        taken_rows = set([row_indices])
        self.assertTrue(self.orbit_list._is_row_taken(
            taken_rows, row_indices))

    def test_get_orbit_list(self):
        """Tests a list of orbits is returned from this function."""
        orbit_list = self.orbit_list.get_orbit_list()
        # clusters for testing
        repr_clusters = [self.cluster_singlet, self.cluster_pair]

        for k, orbit in enumerate(orbit_list):
            with self.subTest(orbit=orbit):
                self.assertEqual(orbit.get_representative_cluster(),
                                 repr_clusters[k])

    def test_remove_all_orbits(self):
        """Tests removing all orbits"""

        mi = [1]*len(self.orbit_list.get_primitive_structure())
        len_before = len(self.orbit_list)
        self.assertNotEqual(len_before, 0)
        self.orbit_list.remove_inactive_orbits(mi)
        len_after = len(self.orbit_list)
        self.assertEqual(len_after, 0)

    def test_get_primitive_structure(self):
        """
        Tests get primitive structure functionality.

        Todo
        ----
        Test fails
        """
        self.assertIsInstance(
            self.orbit_list.get_primitive_structure(), Structure)

    def test_len(self):
        """Tests length of orbit list."""
        self.assertEqual(len(self.orbit_list), 2)

    def test_get_supercell_orbit_list(self):
        """Tests orbit list is returned for the given supercell."""
        # TODO : Tests fails for an actual supercell of the testing structure
        atoms_supercell = self.atoms.copy()
        orbit_list_super = \
            self.orbit_list.get_supercell_orbit_list(atoms_supercell)
        orbit_list_super.sort()
        self.orbit_list.sort()
        for k in range(len(orbit_list_super)):
            orbit_super = orbit_list_super.get_orbit(k)
            orbit = self.orbit_list.get_orbit(k)
            self.assertEqual(orbit, orbit_super)

    def test_translate_sites_to_unitcell(self):
        """Tests the get all translated sites functionality."""
        # no offset site shoud get itself as translated
        sites = [LatticeSite(0, [0, 0, 0])]
        target = [[LatticeSite(0, [0, 0, 0])]]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unitcell(sites, False),
            target)

        # test a singlet site with offset
        sites = [LatticeSite(3, [0, 0, -1])]
        target = [[LatticeSite(3, [0, 0, -1])],
                  [LatticeSite(3, [0, 0, 0])]]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unitcell(sites, False),
            target)

        # sort output
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unitcell(sites, True),
            sorted(target))

        # Does it break when the offset is floats?
        sites = [LatticeSite(0, [0.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unitcell(sites, False),
            target)

        # Test two sites with floats
        sites = [LatticeSite(0, [1.0, 0.0, 0.0]),
                 LatticeSite(0, [0.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [0.0, 0.0, 0.0]),
                   LatticeSite(0, [-1., 0.0, 0.0])],
                  sites]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unitcell(sites, False),
            target)

        # Test sites where none is inside unit cell
        sites = [LatticeSite(0, [1.0, 2.0, -1.0]),
                 LatticeSite(2, [2.0, 0.0, 0.0])]

        target = [[LatticeSite(0, [-1.0, 2.0, -1.0]),
                   LatticeSite(2, [0.0, 0.0, 0.0])],
                  [LatticeSite(0, [0.0, 0.0, 0.0]),
                   LatticeSite(2, [1.0, -2.0, 1.0])],
                  sites]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unitcell(sites, False),
            target)

    def test_get_all_columns_from_sites(self):
        """Tests get_all_columns_from_sites functionality."""
        # These sites are first and last elements in column1
        sites = [LatticeSite(0, [0., 0., 0.]),
                 LatticeSite(0, [1., 0., 0.])]

        pm = self.orbit_list.permutation_matrix
        column1 = [row[0] for row in pm]

        columns = \
            self.orbit_list._get_all_columns_from_sites(sites, column1, pm)
        for i in range(len(pm[0])):
            perm_sites = [pm[0][i], pm[-1][i]]
            translated_sites = \
                self.orbit_list._get_sites_translated_to_unitcell(perm_sites,
                                                                  False)
            for k, sites in enumerate(translated_sites):
                self.assertEqual(columns[k+2*i], sites)

    def _test_allowed_permutations(self, atoms):
        """Tests allowed permutations of orbits in orbit list.

        This test works in the following fashion.
        For each orbit in orbit_list:
        1- Translate representative sites to unitcell
        2- Permute translated sites
        3- Get sites from all columns of permutation matrix
        that map simultaneusly to the permuted sites.
        3- If permutation is not allowed then check that any of translated
        sites cannot be found in columns obtained in previous step.
        4- If at least one of translated sites is found in columns
        then append the respective permutation to allowed_perm list.
        5. Check allowed_perm list is equal to orbit.allowed_permutation.
        """
        cutoffs = [1.4, 1.4]
        orbit_list = OrbitList(atoms, cutoffs)

        pm = orbit_list.permutation_matrix
        column1 = [row[0] for row in pm]

        for orbit in orbit_list.orbits:
            # Set up all possible permutations
            allowed_perm = []
            all_perm = \
                [list(perm) for perm in permutations(range(orbit.order))]
            # Get representative site of orbit
            repr_sites = orbit.get_representative_sites()
            translated_sites = \
                orbit_list._get_sites_translated_to_unitcell(repr_sites, False)
            for sites in translated_sites:
                for perm in all_perm:
                    # Permute translated sites
                    perm_sites = get_permutation(sites, perm)
                    # Get from all columns those sites at the rows
                    # where permuted sites is found in column1.
                    columns = \
                        orbit_list._get_all_columns_from_sites(perm_sites,
                                                               column1, pm)
                    # Any translated sites will be find in columns since
                    # permutation is not allowed
                    if perm not in orbit.allowed_permutations:
                        self.assertTrue(
                            any(s not in columns for s in translated_sites))
                    # If translated sites is found then save permutation
                    for s in translated_sites:
                        if s in columns and perm not in allowed_perm:
                            allowed_perm.append(perm)
            # Check all collected permutations match allowed_permutations
            self.assertEqual(sorted(allowed_perm),
                             sorted(orbit.allowed_permutations))

    def _test_equivalent_sites(self, atoms):
        """
        Tests permutations taken equivalent sites to representative sites.
        """
        cutoffs = [1.4, 1.4]
        orbit_list = OrbitList(atoms, cutoffs)

        pm = orbit_list.permutation_matrix
        column1 = [row[0] for row in pm]

        for orbit in orbit_list.orbits:
            match_repr_site = False
            # Take representative sites and translated them into unitcell
            repr_sites = orbit.get_representative_sites()
            print(len(repr_sites))
            # Take equivalent sites and its permutations_to_representative
            for eq_sites, perm in zip(orbit.equivalent_sites,
                                      orbit.permutations_to_representative):
                trans_eq_sites = \
                    orbit_list._get_sites_translated_to_unitcell(eq_sites,
                                                                 False)
                # Permute equivalent sites and get all columns from those sites
                for sites in trans_eq_sites:
                    perm_sites = get_permutation(sites, perm)
                    columns = \
                        orbit_list._get_all_columns_from_sites(perm_sites,
                                                               column1, pm)
                    # Check representative sites can be found in columns
                    if repr_sites in columns:
                        match_repr_site = True
            self.assertTrue(match_repr_site)

    def test_orbit_permutations_for_atoms_in_database(self):
        """
        Tests allowed_permutation and equivalent_sites of orbits in orbit_list
        for atoms in database (only atoms with pbc=True).
        """
        db = ase_connect("structures_for_testing.db")
        for row in db.select('pbc=TTT'):
            atoms = row.toatoms()
            with self.subTest(atoms_tag=row.tag):
                self._test_allowed_permutations(atoms)
                self._test_equivalent_sites(atoms)

    def test_orbit_list_non_pbc(self):
        """
        Tests that singlets in orbit list retrieves the right number of unique
        sites of the structure with different periodic boundary conditions.
        """
        # TODO: Returned results are incorrect for simple-cubic structures.
        atoms = bulk('Al', 'sc', a=4.0).repeat(4)
        # [True, True, False]
        atoms.set_pbc([True, True, False])
        orbit_list = OrbitList(atoms, [0.])
        self.assertEqual(len(orbit_list), 4)
        # [True, False, False]
        atoms.set_pbc([True, False, False])
        orbit_list = OrbitList(atoms, [0.])
        self.assertEqual(len(orbit_list), 7)
        # [False]
        atoms.set_pbc([False, False, False])
        orbit_list = OrbitList(atoms, [0.])
        self.assertEqual(len(orbit_list), 20)

    def test_orbit_list_fcc(self):
        """
        Tests orbit list has the right number of singlet and pairs for
        a fcc structure.
        """
        atoms = bulk('Al', 'fcc', a=3.0)
        cutoffs = [2.5]
        orbit_list = OrbitList(atoms, cutoffs)
        # only a singlet and a pair are expected
        self.assertEqual(len(orbit_list), 2)
        # singlet
        singlet = orbit_list.get_orbit(0)
        self.assertEqual(len(singlet), 1)
        # pair has multiplicity equal to 4
        pairs = orbit_list.get_orbit(1)
        self.assertEqual(len(pairs), 6)
        # not more orbits listed
        with self.assertRaises(IndexError):
            orbit_list.get_orbit(2)

    def test_orbit_list_bcc(self):
        """
        Tests orbit list has the right number  of singlet and pairs for
        a bcc structure.
        """
        atoms = bulk('Al', 'bcc', a=3.0)
        cutoffs = [3.0]
        orbit_list = OrbitList(atoms, cutoffs)
        # one singlet and two pairs expected
        self.assertEqual(len(orbit_list), 3)
        # singlet
        singlet = orbit_list.get_orbit(0)
        self.assertEqual(len(singlet), 1)
        # first pair has multiplicity equal to 4
        pairs = orbit_list.get_orbit(1)
        self.assertEqual(len(pairs), 4)
        # first pair has multiplicity equal to 3
        pairs = orbit_list.get_orbit(2)
        self.assertEqual(len(pairs), 3)
        # not more orbits listed
        with self.assertRaises(IndexError):
            orbit_list.get_orbit(3)

    def test_orbit_list_hcp(self):
        """
        Tests orbit list has the right number of singlet and pairs for
        a hcp structure.
        """
        atoms = bulk('Ni', 'hcp', a=3.0)
        cutoffs = [3.1]
        orbit_list = OrbitList(atoms, cutoffs)
        # only one singlet and one pair expected
        self.assertEqual(len(orbit_list), 3)
        # singlet
        singlet = orbit_list.get_orbit(0)
        self.assertEqual(len(singlet), 2)
        # pair has multiplicity equal to 4
        pairs = orbit_list.get_orbit(1)
        self.assertEqual(len(pairs), 6)
        # pair has multiplicity equal to 4
        pairs = orbit_list.get_orbit(2)
        self.assertEqual(len(pairs), 6)
        # not more orbits listed
        with self.assertRaises(IndexError):
            orbit_list.get_orbit(3)


if __name__ == '__main__':
    unittest.main()
