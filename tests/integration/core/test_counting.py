"""
This is not a well written test.

Todo
----
* replace with proper unit test(s)
"""

import random
import numpy as np

from ase.db import connect
from icet.core.cluster_counts import ClusterCounts
from icet import OrbitList


def get_equivalent_clustermap_key(key1, clustermap_keys, tol=1e-3):
    """
    Will search for key1 in clustermap keys such that the distances are all
    roughly the same and the geometrical size is rougly the same. It will
    return the key in clustermap_keys that are within this threshhold if none
    is find it will throw an exception
    """

    for key2 in clustermap_keys:
        if len(key1.sites) == len(key2.sites):
            if np.linalg.norm(np.array(key1.distances) -
                              np.array(key2.distances)) < tol:
                if np.abs(key1.radius -
                          key2.radius) < tol:
                    if np.linalg.norm(np.array(key1.sites) -
                                      np.array(key2.sites)) < tol:
                        return key2
    raise Exception('Did not find a matching key')


def test_no_symmetry_local_orbit_list_counting(atoms_prim, atoms_tag,
                                               cutoffs, repeat):
    """
    This function creates a primitive orbit list (no symmetry) and an orbit
    list on supercell (symmetry) and then compares the cluster counting over
    both of them.

    Parameters
    ----------
    atoms_prim : ASE atoms object
        structure to test
    atoms_tag : string
        description of ASE atom object
    cutoffs : list of floats
        list of cutoff radii, one for each order of cluster (pairs, triplet,
        quadruplets, etc)
    repeat : int
        create a number of repeated ASE atom object

    Todo
    ----
    Please fix the description; it is barely comprehensible.
    """
    atoms = atoms_prim.copy()
    atoms_repeat = atoms_prim.copy().repeat(repeat)
    for atom in atoms_repeat:
        if random.random() < 0.5:
            atom.symbol = 'H'

    # orbit list primitive
    orbit_list_primitive = OrbitList(atoms, cutoffs)
    orbit_list_primitive.sort()

    # orbit list supercell
    orbit_list_supercell = OrbitList(atoms_repeat, cutoffs)
    orbit_list_supercell.sort()

    # set up cluster counts and count clusters
    # with local orbit list
    cluster_count_local = ClusterCounts(orbit_list_primitive,
                                        atoms_repeat)

    # set up cluster counts and count cluster
    # with non-local orbit list
    cluster_count_supercell = ClusterCounts(orbit_list_supercell,
                                            atoms_repeat)

    # get the cluster count map
    cluster_map_local = cluster_count_local.get_cluster_counts()
    cluster_map_supercell = cluster_count_supercell.get_cluster_counts()

    base = 'Cluster counts for local and non-local orbit list '

    msg = base + 'return different number of clusters for {}'.format(atoms_tag)
    assert len(cluster_map_local) == len(cluster_map_supercell), msg

    msg = base + 'return different cluster keys for {}'.format(atoms_tag)
    assert cluster_map_local.keys() == cluster_map_supercell.keys(), msg

    for key1, key2 in zip(sorted(cluster_map_local.keys()),
                          sorted(cluster_map_supercell.keys())):
        key2 = get_equivalent_clustermap_key(key1,
                                             cluster_map_supercell.keys())
        msg = base + 'for {} count {} != {} respectively'.format(
            atoms_tag, cluster_map_local[key1], cluster_map_supercell[key2])
        assert cluster_map_local[key1] == cluster_map_supercell[key2], msg


def get_total_count(cluster_count_dict, cluster):
    """
    Returns the total count of a particular cluster

    Parameters
    ----------
    cluster_count_dict : dictionary
        cluster counting dictionary
    cluster : string
        key of a particular cluster
    """
    count = 0
    for counts in cluster_count_dict[cluster]:
        count += cluster_count_dict[cluster][counts]
    return count


def test_no_symmetry_vs_symmetry_count(atoms_prim, atoms_tag,
                                       cutoffs, repeat):
    """
    This function creates a no-symmetry local orbit list from the neighbor list
    of a primitive cell and compares the cluster counting with the one obtained
    from symmetry orbit list created from the primitive structure and the
    cutoff.

    atoms_prim : ASE atoms object
        structure to test
    atoms_tag : string
        description of ASE atom object
    cutoffs : list of floats
        list of cutoff radii, one for each order of cluster (pairs, triplet,
        quadruplets, etc)
    repeat : int
        repeat ASE Atoms object N `repeat` times

    Raises
    ------
    AssertionError
        If the total count or keys or count of clusters obtained from symmetry
        and no-symmetry orbit list are not equal.
    RuntimeError
        Testing for rocksalt, hcp, perovskite, non-pbc structures

    Todo
    ----
    Using `random` in a test is a terrible idea.
    """
    atoms = atoms_prim.copy()
    atoms_repeat = atoms_prim.copy().repeat(repeat)
    for atom in atoms_repeat:
        if random.random() < 0.3:
            atom.symbol = 'H'

    """ Get orbit list no symmetry case """
    orbit_list_no_symmetry = OrbitList(atoms_repeat, cutoffs)
    orbit_list_no_symmetry.sort()
    """Set up cluster count and count. Here we use a cutoff so that no extra
    clusters are found in the symmetry case and compare the counts found in
    both methods."""
    cluster_count_no_symmetry = ClusterCounts(orbit_list_no_symmetry,
                                              atoms_repeat)

    """ Get orbit list symmetry case """
    orbit_list_symmetry = OrbitList(atoms, cutoffs)
    orbit_list_symmetry.sort()

    """ Set up cluster count """
    cluster_count_symmetry = ClusterCounts(orbit_list_symmetry, atoms_repeat)

    msg = 'Testing count orbit list of symmetry case failed'
    msg += ' for {} when counts  {}!={}'.format(atoms_tag,
                                                len(orbit_list_symmetry),
                                                len(orbit_list_no_symmetry))
    assert len(orbit_list_symmetry) == len(orbit_list_no_symmetry), msg

    for key1, key2 in zip(
            sorted(cluster_count_no_symmetry.cluster_counts.keys()),
            sorted(cluster_count_symmetry.cluster_counts.keys())):
        msg = 'sizes for keys are not equal'
        msg += ' {} != {}'.format(key1.radius,
                                  key2.radius)
        assert (key1.radius -
                key2.radius) < 1e-3, msg

        msg = 'clusters are not of the same order.'
        msg += ' {} != {}'.format(key1.distances, key2.distances)
        assert len(key1.distances) == len(key2.distances), msg

        msg = 'sizes for keys are not equal'
        msg += ' {} ! {} '.format(key1.distances, key2.distances)
        assert np.linalg.norm(np.array(key1.distances) -
                              np.array(key2.distances)) < 1e-3, msg

        cluster_map_symmetry = cluster_count_symmetry.cluster_counts
        cluster_map_no_symmetry = cluster_count_no_symmetry.cluster_counts
        msg = 'Testing cluster multiplicity of symmetry case failed'
        msg = ' for {} when total count'.format(atoms_tag)
        msg += ' {}!={}'.format(get_total_count(cluster_map_symmetry, key1),
                                get_total_count(cluster_map_no_symmetry, key1))
        assert (get_total_count(cluster_map_no_symmetry, key1) ==
                get_total_count(cluster_map_symmetry, key2)), msg

        for element_key in cluster_map_no_symmetry[key1]:
            msg = 'Testing element combination in cluster count of symmetry'
            msg += ' case failed for {}.'.format(atoms_tag)
            msg += ' {} != {}'.format(cluster_map_no_symmetry[key1],
                                      cluster_map_symmetry[key2])
            assert (cluster_map_no_symmetry[key1][element_key] ==
                    cluster_map_symmetry[key2][element_key]), msg


db = connect('structures_for_testing.db')
for row in db.select():
    atoms_row = row.toatoms()
    atoms_tag = row.tag
    N = 2
    cutoffs = [1.4] * 3
    if 'NaCl' in atoms_tag:
        cutoffs = [1.1] * 4
    if 'Ni-hcp-hexagonal' in atoms_tag:
        cutoffs = [1.4, 0.5, 0.8]
    # The following systems have muyltiple Wyckoff sites such that even with
    # zero cutoffs the singlets give different sizes of the orbit list
    if 'BaZrO3-perovskite' in atoms_tag or 'distorted' in atoms_tag:
        continue
    if atoms_row.get_pbc().all():
        atoms_row.wrap()
        test_no_symmetry_local_orbit_list_counting(
            atoms_row, atoms_tag, cutoffs, N)
        test_no_symmetry_vs_symmetry_count(atoms_row, atoms_tag, cutoffs, N)
