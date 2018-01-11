from ase import Atoms
from _icet import ManyBodyNeighborList
from .lattice_site import LatticeSite
from .neighbor_list import get_neighbor_lists
from .structure import Structure


def get_all_lattice_neighbors(atoms, neighbor_lists=None, cutoffs=None):
    '''
    Generate lattice neighbors for a configuration.

    Parameters
    ----------
    atoms : ASE Atomss object / icet Structure object (bi-optional)
        atomic configuration
    neighbor_lists : array/list of icet NeighborList objects
        neighbor lists; if None neighbor lists will be created, which
        requires cutoffs to be specified
    cutoffs : list of floats
        cutoffs for the clusters of increasing order; only used if
        `neighbor_lists` is None; note that the first value in the list refers
        to pairs

    Returns
    -------
    list of tuples
        the first part of each tuple identifies the lattice neighbor, the
        second part is a list of neighbors (by index)
    '''

    bothways = False
    lattice_neighbors = []

    # deal with different types of structure objects
    if isinstance(atoms, Atoms):
        structure = Structure.from_atoms(atoms)
    elif isinstance(atoms, Structure):
        structure = atoms
    else:
        msg = ['Unknown structure format']
        msg += ['{} (ClusterSpace)'.format(type(atoms))]
        raise Exception(' '.join(msg))

    # get neigbhor lists
    if neighbor_lists is None:
        neighbor_lists = []
        if cutoffs is None:
            raise Exception('Need to specify either neighbor list or cutoffs')
        else:
            neighbor_lists = get_neighbor_lists(
                structure=structure, cutoffs=cutoffs)
    else:
        # build the neighbor lists
        for nl in neighbor_lists:
            nl.build(structure)

    order = len(cutoffs) + 1

    mbnl = ManyBodyNeighborList()
    # add the mbnl lattice neighbors
    if order >= 2:
        for lattice_index in range(len(structure)):
            lattice_neighbor = mbnl.build(neighbor_lists,
                                          lattice_index, bothways)
            for lat_nbrs in lattice_neighbor:
                lattice_neighbors.append(lat_nbrs)

    # add the pairs and singlets
    for lattice_index in range(len(structure)):
        lat_nbr_i = LatticeSite(lattice_index, [0.0, 0.0, 0.0])
        lattice_neighbors.append(([lat_nbr_i], []))  # singlet
        lattice_neighbors.append(
            ([lat_nbr_i], neighbor_lists[0].get_neighbors(lattice_index)))

    return lattice_neighbors
