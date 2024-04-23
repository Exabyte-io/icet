from itertools import combinations, permutations
from typing import List

import numpy as np
from ase import Atoms
from icet.core.orbit import Orbit
from icet.core.orbit_list import OrbitList
from icet.core.lattice_site import LatticeSite


def _is_site_group_in_orbit(orbit: Orbit, site_group: List[LatticeSite]) -> bool:
    """Checks if a list of sites is found among the clusters in an orbit.
    The number of sites must match the order of the orbit.

    Parameters
    ----------
    orbit
        Orbit.
    site_group
        Sites to be searched for.
    """

    # Ensure that the number of sites matches the order of the orbit
    if len(site_group) != orbit.order:
        return False

    # Check if the set of lattice sites is found among the equivalent sites
    if set(site_group) in [set(cl.lattice_sites) for cl in orbit.clusters]:
        return True

    # Go through all clusters
    site_indices = [site.index for site in site_group]
    for cluster in orbit.clusters:
        cluster_site_indices = [s.index for s in cluster.lattice_sites]

        # Skip if the site indices do not match
        if set(site_indices) != set(cluster_site_indices):
            continue

        # Loop over all permutations of the lattice sites in cluster
        for cluster_site_group in permutations(cluster.lattice_sites):

            # Skip all cases that include pairs of sites with different site indices
            if any(site1.index != site2.index
                   for site1, site2 in zip(site_group, cluster_site_group)):
                continue

            # If the relative offsets for all pairs of sites match, the two
            # clusters are equivalent
            relative_offsets = [site1.unitcell_offset - site2.unitcell_offset
                                for site1, site2 in zip(site_group, cluster_site_group)]
            if all(np.array_equal(ro, relative_offsets[0]) for ro in relative_offsets):
                return True
    return False


def get_transformation_matrix(structure: Atoms,
                              full_orbit_list: OrbitList) -> np.ndarray:
    r"""
    Determines the matrix that transforms the cluster functions in the form
    of spin variables, :math:`\sigma_i\in\{-1,1\}`, to their binary
    equivalents, :math:`x_i\in\{0,1\}`.  The form is obtained by
    performing the substitution (:math:`\sigma_i=1-2x_i`) in the
    cluster expansion expression of the predicted property (commonly the energy).

    Parameters
    ----------
    structure
        Atomic configuration.
    full_orbit_list
        Full orbit list.
    """
    # Go through all clusters associated with each active orbit and
    # determine its contribution to each orbit
    orbit_indices = range(len(full_orbit_list))
    transformation = np.zeros((len(orbit_indices) + 1,
                               len(orbit_indices) + 1))
    transformation[0, 0] = 1.0
    for i, orb_index in enumerate(orbit_indices, 1):
        orbit = full_orbit_list.get_orbit(orb_index)
        repr_sites = orbit.representative_cluster.lattice_sites
        # add contributions to the lower order orbits to which the
        # subclusters belong
        for sub_order in range(orbit.order + 1):
            n_terms_target = len(list(combinations(orbit.representative_cluster.lattice_sites,
                                                   sub_order)))
            n_terms_actual = 0
            if sub_order == 0:
                transformation[0, i] += 1.0
                n_terms_actual += 1
            if sub_order == orbit.order:
                transformation[i, i] += (-2.0) ** (sub_order)
                n_terms_actual += 1
            else:
                comb_sub_sites = combinations(repr_sites, sub_order)
                for sub_sites in comb_sub_sites:
                    for j, sub_index in enumerate(orbit_indices, 1):
                        sub_orbit = full_orbit_list.get_orbit(sub_index)
                        if sub_orbit.order != sub_order:
                            continue
                        if _is_site_group_in_orbit(sub_orbit, sub_sites):
                            transformation[j, i] += (-2.0) ** (sub_order)
                            n_terms_actual += 1
            # If the number of contributions does not match the number of subclusters,
            # this orbit list is incompatible with the ground state finder
            # of subclusters
            if n_terms_actual != n_terms_target:
                raise ValueError('At least one cluster had subclusters that were not included'
                                 ' in the cluster space. This is typically caused by cutoffs'
                                 ' that are longer for a higher-order orbit than lower-order one'
                                 ' (such as 8 Angstrom for triplets and 6 Angstrom for pairs).'
                                 ' Please use a different cluster space for the ground state '
                                 ' finder.')

    return transformation


def transform_parameters(structure: Atoms,
                         full_orbit_list: OrbitList,
                         parameters: np.ndarray) -> np.ndarray:
    r"""
    Transforms the list of parameters, obtained using cluster functions in the
    form of of spin variables, :math:`\sigma_i\in\{-1,1\}`, to their
    equivalents for the case of binary variables,
    :math:`x_i\in\{0,1\}`.

    Parameters
    ----------
    structure
        Atomic configuration.
    full_orbit_list
        Full orbit list.
    parameters
        Parameter vector (spin variables).
    """
    A = get_transformation_matrix(structure, full_orbit_list)
    return np.dot(A, parameters)
