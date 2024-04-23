"""
This module provides a simple wrapper of the ASE Neighbor List class,
returning a list of Lattice Sites.
"""
from typing import List, Union
from ase import Atoms
from ase.neighborlist import NeighborList as ASENeighborList
from .lattice_site import LatticeSite
from .structure import Structure


def get_neighbor_lists(structure: Union[Atoms, Structure],
                       cutoffs: List,
                       position_tolerance: float = 1e-5) -> List[List]:
    """
    Returns a list of :program:`icet` neighbor lists given a configuration and cutoffs.

    Parameters
    ----------
    structure
        Atomic configuration.
    cutoffs
        Positive floats indicating the cutoffs for the various clusters.
    position_tolerance
        Tolerance applied when comparing positions in Cartesian coordinates.
    """
    # deal with different types of structure objects
    if isinstance(structure, Structure):
        structure = Structure.to_atoms(structure)
    elif not isinstance(structure, Atoms):
        msg = ['Unknown structure format']
        msg += [f'{type(structure)} (get_neighbor_list)']
        raise Exception(' '.join(msg))

    neighbor_lists = []
    for cutoff in cutoffs:
        neighbor_list = []
        ase_nl = ASENeighborList(len(structure)*[cutoff/2.0],
                                 skin=2*position_tolerance, bothways=True,
                                 self_interaction=False)
        ase_nl.update(structure)

        for i in range(len(structure)):
            ase_indices, ase_offsets = ase_nl.get_neighbors(i)
            site = []
            # Update the final list of sites with LatticeSite objects
            for index, offset in zip(ase_indices, ase_offsets):
                element = LatticeSite(index, offset)
                site.append(element)
            # sort by unitcell_offset first
            # have to cast to a list, since numpy array isn't iterable
            site = sorted(site, key=lambda s: list(s.unitcell_offset))
            # now sort by index
            site = sorted(site, key=lambda s: s.index)

            neighbor_list.append(site)
        neighbor_lists.append(neighbor_list)

    # This returns list[list[list[LatticeSite]]], which corresponds to
    # <std::vector<std::vector<std::vector<LatticeSite>>>>
    return neighbor_lists
