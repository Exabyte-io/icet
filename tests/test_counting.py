from icetdev import orbitList
from icetdev.structure import structure_from_atoms, Structure
from icetdev.manybodyNeighborlist import *
from ase import Atoms
from ase.db import connect
from icetdev.neighborlist import get_neighborlists
from icetdev.orbitList import create_orbit_list
from ase.spacegroup import crystal
import time
from icetdev.clusterCounts import *
import random

"""
BUG: AssertionError: Testing clusters of cluster map failed for structure Al-fcc-primitive_cell
"""

def setup_test_orbitlist(atoms, cutoffs):
    """
    A helper for getting structure, mbnl and neighborlists used for setting up a orbitlist without symmetry
    """
    structure = structure_from_atoms(atoms)
    mbnl = ManybodyNeighborlist()
    neighborlists = get_neighborlists(atoms=atoms, cutoffs=cutoffs)
    mbnl.build(neighborlists, 0, True)
    return structure, mbnl, neighborlists
 
def test_no_symmetry_local_orbitlist_counting(atoms_primitive, atoms_tag,  cutoffs, repeatInteger):
    """
    Creates a primitive orbitlist and create an orbitlist on supercell and
    then compares the counting over both of them.
    """
    atoms = atoms_primitive.copy()
    atoms_repeat = atoms_primitive.copy().repeat(repeatInteger)
    for atom in atoms_repeat:
        if random.random() < 0.5:
            atom.symbol = "H"

    structure_supercell = structure_from_atoms(atoms_repeat)

    structure, mbnl, neighborlists = setup_test_orbitlist(atoms, cutoffs)

    orbitlist_primitive = orbitList.OrbitList(neighborlists, structure)

    clusterCount_local = ClusterCounts()

    clusterCount_local.count_each_local_orbitlist(
        structure_supercell, orbitlist_primitive)

    structure_supercell, mbnl, neighborlists = setup_test_orbitlist(
        atoms_repeat, cutoffs)

    orbitlist_supercell = orbitList.OrbitList(neighborlists, structure_supercell)

    clusterCount_supercell = ClusterCounts()

    clusterCount_supercell.count_orbitlist(
        structure_supercell, orbitlist_supercell)

    cluster_map_local = clusterCount_local.get_cluster_counts()

    cluster_map_supercell = clusterCount_supercell.get_cluster_counts()

    assert len(cluster_map_local) == len(
        cluster_map_supercell), "Testing len of cluster map failed for structure {}".format(atoms_tag)

    assert cluster_map_local.keys() == cluster_map_supercell.keys(
        ), "Testing keys of cluster map failed for structure {}".format(atoms_tag) 
mbnl.build and mbnl.tester
    for key in cluster_map_local.keys():
        print(cluster_map_local[key])
        print(cluster_map_supercell[key])
        assert cluster_map_local[key] == cluster_map_supercell[key], "Testing cluster mapping failed for structure {}".format(atoms_tag)
        print(" ")


def get_total_count(cluster_count_dict, cluster):
    """
    Returns the total count of a particular cluster
    """
    count = 0
    for counts in cluster_count_dict[cluster]:
        count += cluster_count_dict[cluster][counts]
    return count


def test_no_symmetry_vs_symmetry_count(primitive_atoms, atoms_tag, cutoffs, repeatInteger):
    """
    Here we use a cutoff so that no extra clusters are found in the symmetry case
    and compare the counts found in both methods (should be equal)
    """
    atoms = primitive_atoms.copy()
    atoms_repeat = primitive_atoms.copy().repeat(repeatInteger)
    for atom in atoms_repeat:
        if random.random() < 0.3:
            atom.symbol = "H"

    structure_repeat = structure_from_atoms(atoms_repeat)
mbnl.build and mbnl.tester
    # counting without symmetry
    structure, mbnl, neighborlists = setup_test_orbitlist(atoms, cutoffs)
    orbitlist_no_symmetry = orbitList.OrbitList(neighborlists, structure)
    clusterCount_no_symmetry = ClusterCounts()
    clusterCount_no_symmetry.count_each_local_orbitlist(
        structure_repeat, orbitlist_no_symmetry)
    clusterCountMap_no_symmetry = clusterCount_no_symmetry.get_cluster_counts()

    # counting with symmetry
    orbitlist_symmetry = create_orbit_list(structure, cutoffs, verbosity=0)
    clustercounts_symmetry = ClusterCounts()
    clustercounts_symmetry.count_each_local_orbitlist(
        structure_repeat, orbitlist_symmetry)
    clusterCountMap_symmetry = clustercounts_symmetry.get_cluster_counts()

    assert orbitlist_symmetry.size() == orbitlist_no_symmetry.size(
    ), "testing count orbitlist symmetry failed for structure {}".format(atoms_tag)

    for key in clusterCountMap_no_symmetry.keys():
        assert get_total_count(clusterCountMap_no_symmetry, key) == get_total_count(
            clusterCountMap_symmetry, key), "testing cluster multiplicity failed for structure {}".format(atoms_tag)

        for element_key in clusterCountMap_no_symmetry[key]:
            assert clusterCountMap_no_symmetry[key][element_key] == clusterCountMap_symmetry[key][
                element_key], "testing element combination failed for structure {}".format(atoms_tag)

db = connect('structures_for_testing.db')

for row in db.select():
    atoms_row = row.toatoms()
    atoms_tag = row.tag
    N = 6
    cutoffs = [1.6] * 3
    if atoms_row.get_pbc().all() == True:
        test_no_symmetry_local_orbitlist_counting(atoms_row, atoms_tag, cutoffs, N)
        test_no_symmetry_vs_symmetry_count(atoms_row, atoms_tag, cutoffs, N)

