import numpy as np

from icet.core_py.permutation_matrix import PermutationMatrix
from icet.core_py.orbit import Orbit
from icet.core_py.many_body_neighbor_list import ManyBodyNeighborList
from icet.core_py.lattice_site import LatticeSite
from icet.core.cluster import Cluster
from icet.tools.geometry import find_permutation, get_permutation

from itertools import permutations
import copy

from icet.io.logging import logger
logger = logger.getChild('orbit_list_py')


class OrbitList(object):
    """
    The orbit list object has an internal list of orbits.

    An orbit has a list of equivalent sites with the restriction
    that at least one site is in the primitive cell of the
    primitive structure.

    parameters
    ----------
    atoms : ASE Atoms object
            This atoms object will be used
            to construct a primitive structure
            on which all the lattice sites in the orbits
            are based on.
    cutoffs : list of float
              cutoffs[i] is the cutoff for
              orbits with order i+2.
    """

    def __init__(self, atoms, cutoffs):
        self._permutation_matrix = PermutationMatrix(atoms, max(cutoffs))

        for i, row in enumerate(self.permutation_matrix.pm_lattice_sites):
            for j, site in enumerate(row):
                if not isinstance(site, LatticeSite):
                    raise TypeError(
                        "Type {} is not type "
                        "LatticeSite in row {},"
                        " col {} permutation"
                        " matrix with len {} ".format(
                            type(site), i, j,
                            len(self.permutation_matrix.pm_lattice_sites)))

        self.column1 = self.permutation_matrix.column1
        self.column1_dict = {}
        for i, lattice_site in enumerate(self.column1):
            self.column1_dict[lattice_site] = i
        assert len(set(self.column1)) == len(self.column1)

        self._primitive_structure = self.permutation_matrix.primitive_structure
        mbnl = ManyBodyNeighborList(self.primitive_structure, cutoffs)
        self.taken_rows = set()
        self._orbits = []
        for index in range(len(self.primitive_structure)):
            mb_neigbhors_index = mbnl.build(index)
            for compressed_sites in mb_neigbhors_index:
                for sites in mbnl.unzip(compressed_sites):
                    sites.sort()
                    zero_cells = 0
                    for site in sites:
                        if np.linalg.norm(site.unitcell_offset) < 0.1:
                            zero_cells += 1
                    assert zero_cells != 0

                    if self.is_new_orbit(sites):
                        orbit = self.make_orbit(sites)
                        self._orbits.append(orbit)
        self.sort()

        self.find_orbit_permutation_info()

    def sort(self):
        """
        Sort the list of orbits.
        """
        self._orbits.sort()

    @property
    def primitive_structure(self):
        """
        Returns the primitive structure to
        which the lattice sites in the
        orbits are referenced to.
        """
        return self._primitive_structure.copy()

    @property
    def permutation_matrix(self):
        """
        Return icet PermutationMatrix object.
        """
        return self._permutation_matrix

    @property
    def orbits(self):
        """
        Return the internal list of orbits
        """
        return self._orbits

    def is_new_orbit(self, sites):
        """
        Checks if these sites has been added into
        an orbit already.
        """
        if len(sites) == 0:
            raise RuntimeError("sites is empty in is_new_orbit")

        translated_eq_sites = self.get_all_translated_sites(sites)
        if len(translated_eq_sites) == 0:
            raise RuntimeError("translated_eq_sites is empty in is_new_orbit")
        try:
            sites_indices_match = self.get_matches_in_pm(translated_eq_sites)
        except RuntimeError as e:
            if False and "Did not find any of the " in str(e):
                return False
            else:
                raise RuntimeError(e)
        for site_index in sites_indices_match:
            if self.is_rows_taken(site_index[1]):
                return False
        return True

    def make_orbit(self, sites):
        """
        Creates a new orbit from the sites.
        It will find the rows of the first column in
        the permutation matrix that correspond
        to the sites in the sites and add all
        equivalent sites in an Orbit by traversing the
        columns in the permutation matrix

        Parameters
        ----------
        sites : list of icet Lattice Site objects

        returns orbit : icet Orbit object
        """

        rows = self.get_rows(sites)
        assert len(rows) == len(sites)
        assert isinstance(rows, list)
        assert isinstance(rows[0], list)
        assert isinstance(rows[0][0], LatticeSite), "{} != LatticeSite".format(
            type(rows[0][0]))

        for i in range(len(sites)):
            assert len(rows[i]) == len(rows[0])

        cluster = Cluster.from_python(self.primitive_structure, sites)
        orbit = Orbit(cluster)
        for i in range(len(rows[0])):
            eq_sites = [row[i] for row in rows]

            translated_eq_sites = self.get_all_translated_sites(eq_sites)
            sites_indices_match = self.get_matches_in_pm(translated_eq_sites)
            new_sites = True
            for site_index in sites_indices_match:
                if self.is_rows_taken(site_index[1]):
                    new_sites = False
                    break

            if new_sites:
                for valid_sites in sites_indices_match:
                    if self.is_valid_sites(valid_sites[0]):
                        orbit.equivalent_sites.append(
                            copy.deepcopy(valid_sites[0]))
                        break
                else:
                    raise RuntimeError("No sites were valid")
            for site_index in sites_indices_match:
                self.take_row(site_index[1])

        orbit.sort()
        return orbit

    def is_valid_sites(self, sites):
        """
        Check if sites are valid, i.e. at least one
        lattice site has the offset [0,0,0]
        """
        for site in sites:
            if np.linalg.norm(site.unitcell_offset) < 0.1:
                return True
        return False

    def get_rows(self, sites):
        """
        Returns rows of the permutation matrix
        corresponding to the sites in the sites.

        Parameters
        ----------

        sites : list of icet Lattice Site objects

        """
        row_indices = self.get_row_indices(sites)

        return [copy.deepcopy(self.permutation_matrix.pm_lattice_sites[index])
                for index in row_indices]

    def get_row_indices(self, sites):
        """
        Return the indices for the rows that the sites
        in the sites matches to.

        TODO : think if this should be sorted
        """
        try:
            indices = [self.column1_dict[site] for site in sites]
        except KeyError:
            raise RuntimeError("index not found for sites")

        # TODO check if this should be done elsewhere
        return tuple(indices)

    def get_all_translated_sites(self, sites_in):
        """
        Construct a list of lists of sites.
        Will for each site that has
        a unitcell offset !=[0,0,0] translate all sites so
        *that* site is in [0,0,0] and the others sides just
        go along for the ride.

        The resulted list are all equivalent sites that
        the permutation matrix doesn't really catch on.

        Parameters
        ----------
        sites : list of icet Lattice Sites object
        """
        sites = copy.deepcopy(sites_in)
        for site in sites:
            if not isinstance(site, LatticeSite):
                raise TypeError(
                    "Type {} is not type LatticeSite"
                    " in get_all_translated_sites".format(type(site)))

        translated_sites = [sites]
        for site in sites:
            if not np.allclose(site.unitcell_offset, np.array([0., 0., 0.])):
                translated_sites.append(
                    [ls - site.unitcell_offset for ls in sites])

        return sorted(translated_sites)

    def __len__(self):
        """
        Length of an orbit list is number of orbits in orbit list.
        """
        return len(self._orbits)

    def get_matches_in_pm(self, list_of_sites):
        """
        Returns a list of tuple of lattice sites and
        matched rows in permutation matrix (pm) of those that has a
        match in column 1 of permutation matrix.

        Parameters
        ----------
        list_of_sites : list of list of LatticeSite

        Return
        ------
        matched_sites : the elements in sites that had
        a match in column 1
        """
        if len(list_of_sites) == 0:
            raise RuntimeError("List of sites empty")
        matched_sites = []
        for sites in list_of_sites:
            try:
                rows = self.get_row_indices(sites)
                matched_sites.append(tuple((sites, rows)))
            except RuntimeError as e:
                if "index not found for sites" in str(e):
                    continue
                else:
                    raise RuntimeError(e)
        if len(matched_sites) > 0:
            return sorted(matched_sites)
        else:
            raise RuntimeError("Did not find any of the "
                               "translated sites in col1 "
                               "of permutation matrix in "
                               "function get_matches_in_pm "
                               "in orbit list")

    def __str__(self):
        """String representation."""
        nice_str = ""
        for i, orbit in enumerate(self.orbits):
            nice_str += "Orbit {} - Multiplicity {}\n".format(i, len(orbit))
        return nice_str

    def is_rows_taken(self, rows):
        """
        Checks if these particular rows in the
        permutation matrix has been used for
        constructing an orbit.

        parameters
        ---------
        rows : tuple of ints
            Refers to row indices of
            the permutation matrix
        """
        if tuple(sorted(rows)) in self.taken_rows:
            return True
        return False

    def take_row(self, row):
        """
        Add this row to the list of taken rows
        in the permtuation matrix.

        row : list (tuple) of int
        """
        self.taken_rows.add(tuple(sorted(row)))

    def find_orbit_permutation_info(self):
        """
        Finds the permutations info
        for each orbit.
        Will set both permutation to representative
        and allowed permutations

        For each orbit:

        1. Take representative sites
        2. Find the rows these sites belong to
           (also find the unit cell offsets equivalent sites)

        3. Get all columns for these rows, i.e the sites that
           are directly equivalent, call these p_equal.
           Don't sort the actual list of lattice sites
           (ok to sort list of lists of lattice sites)

        4. Construct all possible permutations for the
           representative sites, call these p_all

        5. Construct the intersect of p_equal and
           p_all, call this p_allowed_permutations.
        6. Get the index version of p_allowed_permutations
           and these are then the allowed permutations for this orbit.

        Start to find the permutation to representative sites now
        7. take the sites in the orbit:
            site exist in p_all?:
                those sites are then related to
                representative_sites through the permutation
            else:
                loop over permutations of the sites:
                    does the permutation exist in p_all?:
                        that permutation is then related
                        to rep_sites through that permutation
                    else:
                        continue
        """
        for orbit in self.orbits:
            # step one  Take representative sites
            eq_sites = orbit.representative_sites
            # get all translated variations of eq sites
            translated_eq_sites = self.get_all_translated_sites(eq_sites)
            # Step 2: find the rows these sites belong to and:
            # Step 3: get all rows for all the translated sites
            all_translated_p_equal = []
            for sites in translated_eq_sites:
                rows = self.get_rows(sites)
                for i in range(len(rows[0])):
                    row_sites = [row[i] for row in rows]
                    all_translated_p_equal.append(tuple(row_sites))
                    translated_row_sites = self.get_all_translated_sites(
                        row_sites)
                    for trans_row_sites in translated_row_sites:
                        all_translated_p_equal.append(tuple(trans_row_sites))

            all_translated_p_equal.sort()  # check what this does

            # Step four: Construct all possible permutations
            #  for the representative sites

            p_all_with_translated_equivalent = []
            all_possible_permutations = list(
                permutations(range(len(eq_sites)), len(eq_sites)))
            for sites in translated_eq_sites:
                for permutation in all_possible_permutations:
                    p_all_with_translated_equivalent.append(
                        tuple(get_permutation(sites, permutation)))

            p_all_with_translated_equivalent.sort()
            # Step five:  Construct the intersect of p_equal and p_all

            p_allowed_permutations = set(all_translated_p_equal).intersection(
                set(copy.deepcopy(p_all_with_translated_equivalent)))

            # Step six: Get the indice version of p_allowed_permutations
            # TODO: find_permutation does not raise any exception
            allowed_permutations = set()
            for sites in p_allowed_permutations:
                failed_loops = 0
                for translated_rep_sites in translated_eq_sites:
                    try:
                        perm = find_permutation(translated_rep_sites, sites)
                        allowed_permutations.add(tuple(perm))
                    except Exception as e:
                        failed_loops += 1
                        logger.debug("Caught exception {}".format(str(e)))
                        if failed_loops == len(translated_eq_sites):
                            raise Exception(
                                " did not find any integer permutation"
                                " from allowed permutation to any"
                                " translated representative site ")

            # step 7
            p_equal_set = set()
            [p_equal_set.add(sites) for sites in all_translated_p_equal]
            site_permutations = []

            for sites in orbit.equivalent_sites:
                any_translated_equivalents = False
                translated_sites = self.get_all_translated_sites(sites)
                for site in translated_sites:
                    if tuple(site) in p_equal_set:
                        any_translated_equivalents = True

                if not tuple(sites) in p_equal_set \
                        and not any_translated_equivalents:
                    #  Did not find the orbit.eq_sites in p_equal
                    #  meaning that this eq site does not have an
                    #  allowed permutation

                    all_permutation_of_sites = []
                    for trans_sites in translated_sites:
                        for permutation in all_possible_permutations:
                            all_permutation_of_sites.append(
                                [get_permutation(trans_sites,
                                                 permutation),
                                 trans_sites])

                    for perm_sites in all_permutation_of_sites:
                        if tuple(perm_sites[0]) in p_equal_set:
                            # one perm is one of the equivalent sites.
                            #  This means that eqOrbitSites is
                            #  associated to p_equal
                            permutation = find_permutation(
                                perm_sites[0], perm_sites[1])
                            site_permutations.append(permutation)
                            break
                    else:
                        raise RuntimeError(
                            "did not find a permutation of the"
                            " orbit sites to the permutations"
                            " of the representative sites")

                else:
                    # Direct match. Score!
                    permutation_to_eq_sites = find_permutation(
                        sites, sites)  # identical permutation
                    site_permutations.append(permutation_to_eq_sites)

            if len(site_permutations) != len(orbit):
                raise RuntimeError("each set of site did not"
                                   " get a permutations {} != {}".format(
                                       len(site_permutations), len(orbit)))

            orbit.permutations_to_representative = site_permutations
            orbit.allowed_permutations = allowed_permutations
