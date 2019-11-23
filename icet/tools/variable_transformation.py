from itertools import combinations, permutations
from typing import List

import numpy as np
from ase import Atoms
from ..core.orbit import Orbit
from ..core.orbit_list import OrbitList
from ..core.lattice_site import LatticeSite


def _is_sites_in_orbit(orbit: Orbit, sites: List[LatticeSite]) -> bool:
    """Checks if the list of lattice sites is found among the equivalent
    sites for the orbit.

    Parameters
    ----------
    orbit
        orbit
    sites
        list of lattice sites
    """

    # Ensure that the number of sites matches the order of the orbit
    if len(sites) != orbit.order:
        return False

    equivalent_sites = orbit.get_equivalent_sites()

    # Check if the set of lattice sites is found among the equivalent sites
    if set(sites) in [set(es) for es in equivalent_sites]:
        return True

    # Go through all equivalent sites
    sites_indices = [s.index for s in sites]
    for orbit_sites in equivalent_sites:
        orbit_sites_indices = [s.index for s in orbit_sites]

        # Skip if the site indices do not match
        if set(sites_indices) != set(orbit_sites_indices):
            continue

        # Loop over all possible ways of pairing sites from the two lists
        for comb_sites in [list(zip(sites, pos))
                           for pos in permutations(orbit_sites)]:

            # Skip all cases that include pairs of sites with different site
            # indices
            if any(cs[0].index != cs[1].index for cs in comb_sites):
                continue

            # If the relative offsets for all pairs of sites match, the two
            # clusters are equivalent
            relative_offsets = [cs[0].unitcell_offset - cs[1].unitcell_offset
                                for cs in comb_sites]
            if all(np.array_equal(ro, relative_offsets[0])
                   for ro in relative_offsets):
                return True
    return False


def get_transformation_matrix(structure: Atoms,
                              full_orbit_list: OrbitList,
                              active_orbit_indices: List[int]) -> np.ndarray:
    """
    Determines the matrix that transforms the cluster functions in the form
    of spin variables, (:math:`\\sigma_i\\in\\{-1,1\\}`), to their binary
    equivalents, (:math:`x_i\\in\\{0,1\\}`). The form is obtained by
    performing the substitution (:math:`\\sigma_i=1-2x_i`) in the
    cluster expansion expression of the total energy.

    Parameters
    ----------
    structure
        atomic configuration
    """
    # Go through all clusters associated with each active orbit and
    # determine its contribution to each orbit
    transformation = np.zeros((len(active_orbit_indices) + 1,
                               len(active_orbit_indices) + 1))
    transformation[0, 0] = 1.0
    for i, orb_index in enumerate(active_orbit_indices, 1):
        orbit = full_orbit_list.get_orbit(orb_index)
        rep_sites = orbit.get_representative_sites()
        # add contributions to the lower order orbits to which the
        # subclusters belong
        for sub_order in range(orbit.order + 1):
            n_terms_target = len(list(combinations(rep_sites, sub_order)))
            n_terms_actual = 0
            if sub_order == 0:
                transformation[0, i] += 1.0
                n_terms_actual += 1
            if sub_order == orbit.order:
                transformation[i, i] += (-2.0) ** (sub_order)
                n_terms_actual += 1
            else:
                comb_sub_sites = combinations(rep_sites, sub_order)
                for sub_sites in comb_sub_sites:
                    for j, sub_index in enumerate(active_orbit_indices, 1):
                        sub_orbit = full_orbit_list.get_orbit(sub_index)
                        if sub_orbit.order != sub_order:
                            continue
                        if _is_sites_in_orbit(sub_orbit, sub_sites):
                            transformation[j, i] += (-2.0) ** (sub_order)
                            n_terms_actual += 1
            # check that the number of contributions matches the number
            # of subclusters
            if n_terms_actual != n_terms_target:
                raise Exception("Fewer matches ({}) than expected ({}) for"
                                " sub-cluster: {}".format(n_terms_actual,
                                                          n_terms_target,
                                                          sub_sites))
    return transformation


def transform_ECIs(structure: Atoms,
                   full_orbit_list: OrbitList,
                   active_orbit_indices: List[int],
                   ecis: np.ndarray) -> np.ndarray:
    """
    Transforms the list of ECIs, obtained using cluster functions in the
    form of of spin variables, (:math:`\\sigma_i\\in\\{-1,1\\}`), to their
    equivalents for the case of binary variables,
    (:math:`x_i\\in\\{0,1\\}`).

    Parameters
    ----------
    structure
        atomic configuration
    full_orbit_list
        full orbit list
    active_orbit_indices
        active orbit indices
    ecis
        spin variable ECIs
    """
    A = get_transformation_matrix(structure, full_orbit_list, active_orbit_indices)
    return np.dot(A, ecis)
