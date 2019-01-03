import itertools
import numpy as np
import time

from ase.build import bulk
from icet.core.cluster import Cluster
from icet.core.lattice_site import LatticeSite
from icet.core.orbit import Orbit


def init_cpp_orbit(number_of_sites):
    """
    Returns a pair and triplet C++ orbit with a size equal to number_of_sites.

    Parameters
    ----------
    number_of_sites : int
        number of equivalent sites in the orbit
    """

    indices = [i for i in range(number_of_sites)]
    unitcell_offsets = []
    cartesian_product_lists = [[i for i in range(
        number_of_sites // 3)], [0, 2], [-number_of_sites, number_of_sites]]
    for element in itertools.product(*cartesian_product_lists):
        unitcell_offsets.append(list(element))
    lattice_sites_pairs_cpp = [[LatticeSite(index,
                                            unitcell_offset),
                                LatticeSite(index + 1,
                                            unitcell_offset)]
                               for index, unitcell_offset in
                               zip(indices, unitcell_offsets)]

    lattice_sites_triplets_cpp = [[LatticeSite(index,
                                               unitcell_offset),
                                   LatticeSite(
        index + 1, unitcell_offset),
        LatticeSite(
        index + 3, unitcell_offset)]
        for index, unitcell_offset in
        zip(indices, unitcell_offsets)]

    lattice_site_for_cluster = [
        LatticeSite(0, [i, 0, 0]) for i in range(3)]
    atoms = bulk('Al')

    pair_cluster = Cluster.from_python(
        atoms, [lattice_site_for_cluster[0],
                lattice_site_for_cluster[1]], True)
    triplet_cluster = Cluster.from_python(
        atoms, lattice_site_for_cluster, True)

    orbit_pair_cpp = Orbit(pair_cluster)
    orbit_pair_cpp.equivalent_sites = lattice_sites_pairs_cpp

    orbit_triplet_cpp = Orbit(triplet_cluster)
    orbit_triplet_cpp.equivalent_sites = lattice_sites_triplets_cpp

    return orbit_pair_cpp, orbit_triplet_cpp


def time_orbit_translating(orbit, iterations=1000):
    """
    Times orbit + offset operation.
    if you doubt this doesn't work try:
    adding assert \
    (orbit_copy.equivalent_sites[0][0].unitcell_offset).all() == \
       (orbit_copy.equivalent_sites[0][0].unitcell_offset + offset).all()
    in the loop
    """
    offset = np.array([3, 2, 1])
    t = time.process_time()
    for i in range(iterations):
        orbit_copy = orbit + offset  # noqa

    total_time = time.process_time() - t
    return total_time / iterations


def time_orbit_sites_permutations(orbit, iterations=5000):
    """ Times retrieving equivalent sites through permutations. """
    # Get the reversed permutation so we actually do some work
    permutations = [[i for i in reversed(range(orbit.order))]] * len(orbit)
    orbit.permutations_to_representative = permutations

    t = time.process_time()
    for i in range(iterations):
        orbit.permuted_sites

    total_time = time.process_time() - t
    return total_time / iterations


if __name__ == '__main__':

    orbit_pair_cpp, orbit_triplet_cpp = init_cpp_orbit(100)

    # Pair Orbit offset translation
    cpp_timing = time_orbit_translating(orbit_pair_cpp)
    print('Time for pair orbit translation: {:.8f} sec'
          .format(cpp_timing))

    # Triplet Orbit offset translation
    cpp_timing = time_orbit_translating(orbit_triplet_cpp)
    print('Time for triplet orbit translation: {:.8f} sec'
          .format(cpp_timing))

    # Pair permutation timing
    cpp_timing = time_orbit_sites_permutations(orbit_pair_cpp)
    print('Time for pair orbit permutations: {:.8f} sec'
          .format(cpp_timing))

    # Triplet permutation timing
    cpp_timing = time_orbit_sites_permutations(orbit_triplet_cpp)
    print('Time for triplet orbit permutations: {:.8f} sec'
          .format(cpp_timing))
