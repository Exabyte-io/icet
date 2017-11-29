
import numpy as np
import time
from _icetdev import OrbitList
from icetdev.neighborlist import get_neighborlists
from icetdev.permutation_map import permutation_matrix_from_atoms
from icetdev.local_orbitlist_generator import LocalOrbitlistGenerator
from icetdev.structure import Structure
def __fractional_to_cartesian(fractional_coordinates, cell):
    '''
    Convert from fractional to Cartesian coordinates.

    Parameters
    ----------
    fractional_coordinates : list of 3d-vectors
        list fractional coordinates
    cell : 3x3 matrix
        cell metric
    '''
    cartesian_coordinates = [np.dot(frac, cell)
                             for frac in fractional_coordinates]
    return cartesian_coordinates


def __get_lattice_site_permutation_matrix(structure, permutation_matrix,
                                          prune=True, verbosity=0):
    '''
    Return a transformed permutation matrix with lattice sites instead of
    fractional coordinates.

    Permutation matrix is in row major format which we will keep
    '''
    pm_frac = permutation_matrix.get_permutated_positions()

    pm_lattice_sites = []
    for row in pm_frac:
        positions = __fractional_to_cartesian(row, structure.cell)
        lat_nbrs = []
        if np.all(structure.pbc):
            lat_nbrs = structure.find_lattice_sites_by_positions(positions)
        else:
            for pos in positions:
                try:
                    lat_nbr = structure.find_lattice_site_by_position(pos)
                    lat_nbrs.append(lat_nbr)
                except:
                    continue
        if len(lat_nbrs) > 0:
            pm_lattice_sites.append(lat_nbrs)
        else:
            print('lat nbrs are zero')
    if prune:
        if verbosity > 2:
            print('size before pruning {} '.format(len(pm_lattice_sites)))
        pm_lattice_sites = __prune_permutation_matrix(pm_lattice_sites,
                                                      verbosity=0)
        if verbosity > 2:
            print('size after pruning {} '.format(len(pm_lattice_sites)))

    return pm_lattice_sites


def __prune_permutation_matrix(permutation_matrix, verbosity=0):
    '''
    Prunes the matrix so that the first column only contains unique elements.
    '''
    for i in range(len(permutation_matrix)):
        for j in reversed(range(len(permutation_matrix))):
            if j <= i:
                continue
            if permutation_matrix[i][0] == permutation_matrix[j][0]:
                permutation_matrix.pop(j)
                if verbosity > 2:
                    msg = ['Removing duplicate in permutation matrix']
                    msg += ['i: {} j: {}'.format(i, j)]
                    print(' '.join(msg))
    return permutation_matrix


def _get_supercell_orbitlist(self, atoms):
    """
    Returns a orbitlist for a supercell structure

    atoms: ASE atoms object
    """
    structure = Structure.from_atoms(atoms)
    log = LocalOrbitlistGenerator(self, structure)

    supercell_orbitlist = log.generate_full_orbitlist()

    return supercell_orbitlist
OrbitList.get_supercell_orbitlist = _get_supercell_orbitlist


def create_orbit_list(structure, cutoffs, verbosity=0):
    '''
    Build an orbit list.

    Parameters
    ----------
    structure: icet structure object
        input configuration used to initialize mbnl and permutation matrix
    cutoffs : list of float
        cutoff radii for each order
    verbosity : int
        verbosity level

    Returns
    -------
    OrbitList object
    '''
    max_cutoff = np.max(cutoffs)
    total_time_spent = 0

    t0 = time.time()
    permutation_matrix, prim_structure, neighborlist \
        = permutation_matrix_from_atoms(structure.to_atoms(), max_cutoff)
    t1 = time.time()
    time_spent = t1 - t0
    total_time_spent += time_spent

    if verbosity > 3:
        print('Done getting permutation_matrix. Time {} s'.format(time_spent))
    total_time_spent += time_spent

    t0 = time.time()
    neighborlists = get_neighborlists(prim_structure, cutoffs=cutoffs)
    t1 = time.time()
    time_spent = t1 - t0
    total_time_spent += time_spent

    if verbosity > 3:
        print('Done getting neighborlists. Time {} s'.format(time_spent))

    t0 = time.time()
    # transform permutation_matrix to be in lattice site format
    pm_lattice_sites \
        = __get_lattice_site_permutation_matrix(prim_structure,
                                                permutation_matrix,
                                                prune=True,
                                                verbosity=verbosity)
    t1 = time.time()
    time_spent = t1 - t0
    total_time_spent += time_spent

    if verbosity > 3:
        msg = ['Transformation of permutation matrix to lattice neighbor']
        msg += ['format completed (time: {} s)'.format(time_spent)]
        print(' '.join(msg))

    t0 = time.time()
    orbitlist = OrbitList(prim_structure, pm_lattice_sites, neighborlists)
    t1 = time.time()
    time_spent = t1 - t0
    total_time_spent += time_spent

    if verbosity > 3:
        print('Finished construction of orbitlist.'
              ' Time {} s'.format(time_spent))

    if verbosity > 3:
        print('Total time {} s'.format(total_time_spent))

    return orbitlist
