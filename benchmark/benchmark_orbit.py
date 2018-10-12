import itertools
from icet.core_py.lattice_site import LatticeSite
from icet.core.lattice_site import LatticeSite as LatticeSite_cpp
from icet.core_py.orbit import Orbit
from icet.core.orbit import Orbit as Orbit_cpp
import time
from icet.core.cluster import Cluster
from ase.build import bulk
import numpy as np


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
    lattice_sites_pairs_cpp = [[LatticeSite_cpp(index,
                                                unitcell_offset),
                                LatticeSite_cpp(index + 1,
                                                unitcell_offset)]
                               for index, unitcell_offset in
                               zip(indices, unitcell_offsets)]

    lattice_sites_triplets_cpp = [[LatticeSite_cpp(index,
                                                   unitcell_offset),
                                   LatticeSite_cpp(
        index + 1, unitcell_offset),
        LatticeSite_cpp(
        index + 3, unitcell_offset)]
        for index, unitcell_offset in
        zip(indices, unitcell_offsets)]

    lattice_site_for_cluster = [
        LatticeSite(0, [i, 0, 0]) for i in range(3)]
    atoms = bulk("Al")

    pair_cluster = Cluster.from_python(
        atoms, [lattice_site_for_cluster[0],
                lattice_site_for_cluster[1]], True)
    triplet_cluster = Cluster.from_python(
        atoms, lattice_site_for_cluster, True)

    orbit_pair_cpp = Orbit_cpp(pair_cluster)
    orbit_pair_cpp.equivalent_sites = lattice_sites_pairs_cpp

    orbit_triplet_cpp = Orbit_cpp(triplet_cluster)
    orbit_triplet_cpp.equivalent_sites = lattice_sites_triplets_cpp

    return orbit_pair_cpp, orbit_triplet_cpp


def init_python_orbit(number_of_sites):
    """
    Returns a pair and triplet Python orbit with size equal to number_of_sites.

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
    lattice_sites_pairs = [[LatticeSite(index, unitcell_offset),
                            LatticeSite(index + 1, unitcell_offset)]
                           for index, unitcell_offset in
                           zip(indices, unitcell_offsets)]
    lattice_sites_triplets = [[LatticeSite(index, unitcell_offset),
                               LatticeSite(
        index + 1, unitcell_offset),
        LatticeSite(
        index + 3, unitcell_offset)]
        for index, unitcell_offset in
        zip(indices, unitcell_offsets)]
    lattice_site_for_cluster = [
        LatticeSite(0, [i, 0, 0]) for i in range(3)]
    atoms = bulk("Al")

    pair_cluster = Cluster.from_python(
        atoms, [lattice_site_for_cluster[0],
                lattice_site_for_cluster[1]], True)
    triplet_cluster = Cluster.from_python(
        atoms, lattice_site_for_cluster, True)

    orbit_pair = Orbit(pair_cluster)
    orbit_pair.equivalent_sites = lattice_sites_pairs
    orbit_triplet = Orbit(triplet_cluster)
    orbit_triplet.equivalent_sites = lattice_sites_triplets
    return orbit_pair, orbit_triplet


def time_orbit_translating(orbit, iterations=1000):
    """
    Time orbit + offset operation.
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
    """
    Test retrieving equivalent sites through permutations
    """
    # Get the reversed permutation so we actually do some work
    permutations = [[i for i in reversed(range(orbit.order))]] * len(orbit)
    orbit.permutations_to_representative = permutations

    t = time.process_time()
    for i in range(iterations):
        orbit.permuted_sites

    total_time = time.process_time() - t
    return total_time / iterations


if __name__ == '__main__':
    orbit_pair, orbit_triplet = init_python_orbit(100)
    orbit_pair_cpp, orbit_triplet_cpp = init_cpp_orbit(100)

    # Pair Orbit offset translation
    py_timing = time_orbit_translating(orbit_pair)
    cpp_timing = time_orbit_translating(orbit_pair_cpp)
    try:
        cpp_speedup = py_timing / cpp_timing
    except:  # noqa
        cpp_speedup = np.inf
    print("Time for python pair orbit translating: {:.8f}s".format(py_timing))
    print("Time for C++ pair orbit translating: {:.8f}s".format(cpp_timing))
    print("Cpp speedup {:3.3f}".format(cpp_speedup))

    print()
    # Triplet Orbit offset translation
    py_timing = time_orbit_translating(orbit_triplet)
    cpp_timing = time_orbit_translating(orbit_pair_cpp)
    try:
        cpp_speedup = py_timing / cpp_timing
    except:  # noqa
        cpp_speedup = np.inf
    print("Time for python pair orbit translating: {:.8f}s".format(py_timing))
    print("Time for C++ pair orbit translating: {:.8f}s".format(cpp_timing))
    print("Cpp speedup {:3.3f}".format(cpp_speedup))

    print()
    # Pair permutation timing
    py_timing = time_orbit_sites_permutations(orbit_pair)
    cpp_timing = time_orbit_sites_permutations(orbit_pair_cpp)
    try:
        cpp_speedup = py_timing / cpp_timing
    except:  # noqa
        cpp_speedup = np.inf
    print("Time for python pair orbit permutation: {:.8f}s".format(py_timing))
    print("Time for C++ pair orbit permutation: {:.8f}s".format(cpp_timing))
    print("Cpp speedup {:3.3f}".format(cpp_speedup))

    print()
    # Triplet permutation timing
    py_timing = time_orbit_sites_permutations(orbit_triplet)
    cpp_timing = time_orbit_sites_permutations(orbit_triplet_cpp)
    try:
        cpp_speedup = py_timing / cpp_timing
    except:  # noqa
        cpp_speedup = np.inf
    print('''Time for python triplet orbit permutation:
     {:.8f}s'''.format(py_timing))
    print("Time for C++ triplet orbit permutation: {:.8f}s".format(cpp_timing))
    print("Cpp speedup {:3.3f}".format(cpp_speedup))
