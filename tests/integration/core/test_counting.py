'''
This is not a well written test.

Todo
----
* replace with proper unit test(s)
'''

import random
import numpy as np

from ase.db import connect
from icet import Structure
from icet.core.neighbor_list import get_neighbor_lists
from icet.core.cluster_counts import ClusterCounts
from icet.core import orbit_list


def get_equivalent_clustermap_key(key1, clustermap_keys, tol=1e-3):
    '''
    Will search for key1 in clustermap keys such that the distances are all
    roughly the same and the geometrical size is rougly the same. It will
    return the key in clustermap_keys that are within this threshhold if none
    is find it will throw an exception
    '''

    for key2 in clustermap_keys:
        if len(key1.sites) == len(key2.sites):
            if np.linalg.norm(np.array(key1.distances) -
                              np.array(key2.distances)) < tol:
                if np.abs(key1.geometrical_size -
                          key2.geometrical_size) < tol:
                    if np.linalg.norm(np.array(key1.sites) -
                                      np.array(key2.sites)) < tol:
                        return key2
    raise Exception('Did not find a matching key')


def test_no_symmetry_local_orbit_list_counting(atoms_primitive, atoms_tag,
                                               cutoffs, repeat):
    '''
    This function creates a primitive orbit list (no symmetry) and an orbit
    list on supercell (symmetry) and then compares the cluster counting over
    both of them.

    Parameters
    ----------
    atoms_primitive : ASE atoms object
        structure to test
    atoms_tag : string
        description of ASE atom object
    cutoffs : list of floats
        list of cutoff radii, one for each order of cluster (pairs, triplet,
        quadruplets, etc)
    repeat : int
        create a number of repeated ASE atom object

    Raises
    ------
    AssertionError
        If the total count or keys or count of clusters obtained from symmetry
        and no-symmetry orbit list are not equal.
    RuntimeError
        Testing for rocksalt, hcp, perovskite, non-pbc structures

    Todo
    ----
    Please fix the description; it is barely comprehensible.
    '''
    atoms = atoms_primitive.copy()
    atoms_repeat = atoms_primitive.copy().repeat(repeat)
    for atom in atoms_repeat:
        if random.random() < 0.5:
            atom.symbol = 'H'

    ''' Set up neighbor_lists '''
    neighbor_lists_primitive = get_neighbor_lists(atoms=atoms, cutoffs=cutoffs)
    structure = Structure.from_atoms(atoms)
    structure_repeat = Structure.from_atoms(atoms_repeat)
    neighbor_lists_supercell = get_neighbor_lists(
        atoms=atoms_repeat, cutoffs=cutoffs)

    ''' Orbit list primitive '''
    orbit_list_primitive = orbit_list.OrbitList(
        neighbor_lists_primitive, structure)
    orbit_list_primitive.sort()

    ''' Orbit list supercell '''
    orbit_list_supercell = orbit_list.OrbitList(
        neighbor_lists_supercell, structure_repeat)
    orbit_list_supercell.sort()

    ''' Set up cluster counts and count clusters '''
    clusterCount_local = ClusterCounts()

    clusterCount_local.count_clusters(
        structure_repeat, orbit_list_primitive, False)

    clusterCount_supercell = ClusterCounts()

    clusterCount_supercell.count_orbit_list(
        structure_repeat, orbit_list_supercell, False)

    ''' Get the clustercount map '''
    cluster_map_local = clusterCount_local.get_cluster_counts()

    cluster_map_supercell = clusterCount_supercell.get_cluster_counts()

    msg = 'Testing total count of clusters using no symmetry'
    msg += ' local orbit list failed for structure {}'.format(atoms_tag)
    assert len(cluster_map_local) == len(cluster_map_supercell), msg

    msg = 'Testing keys of cluster count using no symmetry local orbit list'
    msg += ' failed for structure {}'.format(atoms_tag)
    assert cluster_map_local.keys() == cluster_map_supercell.keys(), msg

    for key1, keyt in zip(sorted(cluster_map_local.keys()),
                          sorted(cluster_map_supercell.keys())):
        key2 = get_equivalent_clustermap_key(key1,
                                             cluster_map_supercell.keys())
        msg = 'Testing cluster count using no symmetry local orbit list failed'
        msg += ' for {} when counts {} != {}'.format(
            atoms_tag, cluster_map_local[key1], cluster_map_supercell[key2])
        assert cluster_map_local[key1] == cluster_map_supercell[key2], msg


def get_total_count(cluster_count_dict, cluster):
    '''
    Returns the total count of a particular cluster

    Parameters
    ----------
    cluster_count_dict : dictionary
        cluster counting dictionary
    cluster : string
        key of a particular cluster
    '''
    count = 0
    for counts in cluster_count_dict[cluster]:
        count += cluster_count_dict[cluster][counts]
    return count


def test_no_symmetry_vs_symmetry_count(atoms_primitive, atoms_tag,
                                       cutoffs, repeat):
    '''
    This function creates a no-symmetry local orbit list from the neighbor list
    of a primitive cell and compares the cluster counting with the one obtained
    from symmetry orbit list created from the primitive structure and the
    cutoff.

    atoms_primitive : ASE atoms object
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
    '''
    atoms = atoms_primitive.copy()
    atoms_repeat = atoms_primitive.copy().repeat(repeat)
    for atom in atoms_repeat:
        if random.random() < 0.3:
            atom.symbol = 'H'

    structure_repeat = Structure.from_atoms(atoms_repeat)
    structure = Structure.from_atoms(atoms)

    ####################################
    #                                  #
    #      no symmetry counting        #
    #                                  #
    ####################################

    # get neighbor_list
    neighbor_lists = get_neighbor_lists(atoms=atoms, cutoffs=cutoffs)

    structure = Structure.from_atoms(atoms)

    neighbor_lists = get_neighbor_lists(atoms=atoms, cutoffs=cutoffs)

    ''' Get orbit list no symmetry case '''
    orbit_list_no_symmetry = orbit_list.OrbitList(neighbor_lists, structure)
    orbit_list_no_symmetry.sort()
    ''' Set up cluster count and count '''
    clusterCount_no_symmetry = ClusterCounts()
    msg = 'Here we use a cutoff so that no extra clusters are found in the'
    msg += 'symmetry case and compare the counts found in both methods.'
    clusterCount_no_symmetry.count_clusters(
        structure_repeat, orbit_list_no_symmetry, False), msg

    ''' Get the cluster_count map '''
    cluster_map_no_symmetry = clusterCount_no_symmetry.get_cluster_counts()

    ''' Get orbit list symmetry case '''
    orbit_list_symmetry = orbit_list.create_orbit_list(structure, cutoffs,
                                                       verbosity=0)
    orbit_list_symmetry.sort()

    ''' Set up cluster_count and count '''
    cluster_count_symmetry = ClusterCounts()
    cluster_count_symmetry.count_clusters(
        structure_repeat, orbit_list_symmetry, False)

    ''' Get the cluster_count map '''
    cluster_map_symmetry = cluster_count_symmetry.get_cluster_counts()

    msg = 'Testing count orbit list of symmetry case failed'
    msg += ' for {} when counts  {}!={}'.format(atoms_tag,
                                                len(orbit_list_symmetry),
                                                len(orbit_list_no_symmetry))
    assert len(orbit_list_symmetry) == len(orbit_list_no_symmetry), msg

    for key1, key2 in zip(sorted(cluster_map_no_symmetry.keys()),
                          sorted(cluster_map_symmetry.keys())):
        msg = 'sizes for keys are not equal'
        msg += ' {} != {}'.format(key1.geometrical_size,
                                  key2.geometrical_size)
        assert (key1.geometrical_size -
                key2.geometrical_size) < 1e-3, msg

        msg = 'clusters are not of the same order.'
        msg += ' {} != {}'.format(key1.distances, key2.distances)
        assert len(key1.distances) == len(key2.distances), msg

        msg = 'sizes for keys are not equal'
        msg += ' {} ! {} '.format(key1.distances, key2.distances)
        assert np.linalg.norm(np.array(key1.distances) -
                              np.array(key2.distances)) < 1e-3, msg

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


print('')
db = connect('structures_for_testing.db')
for row in db.select():
    atoms_row = row.toatoms()
    atoms_tag = row.tag
    N = 3
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
        print(' structure: {}'.format(row.tag))
        atoms_row.wrap()
        test_no_symmetry_local_orbit_list_counting(
            atoms_row, atoms_tag, cutoffs, N)
        test_no_symmetry_vs_symmetry_count(atoms_row, atoms_tag, cutoffs, N)
