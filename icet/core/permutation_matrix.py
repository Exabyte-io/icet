"""
This module provides a Python interface to the PermutationMatrix
class with supplementary functions.
"""

from typing import List, Tuple

import numpy as np
import spglib

from ase import Atoms
from _icet import PermutationMatrix
from icet.core.lattice_site import LatticeSite
from icet.core.neighbor_list import NeighborList
from icet.core.structure import Structure
from icet.io.logging import logger
from icet.tools.geometry import (ase_atoms_to_spglib_cell,
                                 get_fractional_positions_from_neighbor_list,
                                 get_primitive_structure)

logger = logger.getChild('permutation_matrix')


def permutation_matrix_from_structure(structure: Atoms, cutoff: float,
                                      find_prim: bool = True) \
        -> Tuple[np.ndarray, Structure, NeighborList]:
    """Sets up a list of permutation maps from an Atoms object.

    Parameters
    ----------
    structure
        input structure
    cutoff
        cutoff radius
    find_primitive
        if True the symmetries of the primitive structure will be employed

    Returns
    -------
    the tuple that is returned comprises the permutation matrix, the
    primitive structure, and the neighbor list
    """

    structure = structure.copy()

    structure_prim = structure
    if find_prim:
        structure_prim = get_primitive_structure(structure)

    logger.debug('Size of primitive structure: {}'.format(len(structure_prim)))

    structure_as_tuple = ase_atoms_to_spglib_cell(structure_prim)

    # get symmetry information
    symmetry = spglib.get_symmetry(structure_as_tuple)
    translations = symmetry['translations']
    rotations = symmetry['rotations']

    # set up a permutation map object
    permutation_matrix = PermutationMatrix(translations, rotations)

    # create neighbor_lists from the different cutoffs
    prim_icet_structure = Structure.from_atoms(structure_prim)
    neighbor_list = NeighborList(cutoff)
    neighbor_list.build(prim_icet_structure)

    # get fractional positions for neighbor_list
    frac_positions = get_fractional_positions_from_neighbor_list(
        prim_icet_structure, neighbor_list)

    logger.debug('Number of fractional positions:'
                 ' {}'.format(len(frac_positions)))
    if frac_positions is not None:
        permutation_matrix.build(frac_positions)

    return permutation_matrix, prim_icet_structure, neighbor_list


def _get_lattice_site_permutation_matrix(structure: Structure,
                                         permutation_matrix: PermutationMatrix,
                                         prune: bool = True) -> np.ndarray:
    """
    Returns a transformed permutation matrix with lattice sites as entries
    instead of fractional coordinates.

    Parameters
    ----------
    structure
        primitive atomic icet structure
    permutation_matrix
        permutation matrix with fractional coordinates format entries
    prune
        if True the permutation matrix will be pruned

    Returns
    -------
    Permutation matrix in a row major order with lattice site format entries
    """
    pm_frac = permutation_matrix.get_permuted_positions()

    pm_lattice_sites = []
    for row in pm_frac:
        positions = _fractional_to_cartesian(row, structure.cell)
        lat_neighbors = []
        if np.all(structure.pbc):
            lat_neighbors = \
                structure.find_lattice_sites_by_positions(positions)
        else:
            for pos in positions:
                try:
                    lat_neighbor = \
                        structure.find_lattice_site_by_position(pos)
                except RuntimeError:
                    continue
                lat_neighbors.append(lat_neighbor)
        if lat_neighbors is not None:
            pm_lattice_sites.append(lat_neighbors)
        else:
            logger.warning('Unable to transform any element in a column of the'
                           ' fractional permutation matrix to lattice site')
    if prune:
        logger.debug('Size of columns of the permutation matrix before'
                     ' pruning {}'.format(len(pm_lattice_sites)))

        pm_lattice_sites = _prune_permutation_matrix(pm_lattice_sites)

        logger.debug('Size of columns of the permutation matrix after'
                     ' pruning {}'.format(len(pm_lattice_sites)))

    return pm_lattice_sites


def _prune_permutation_matrix(permutation_matrix: List[List[LatticeSite]]):
    """
    Prunes the matrix so that the first column only contains unique elements.

    Parameters
    ----------
    permutation_matrix
        permutation matrix with LatticeSite type entries
    """

    for i in range(len(permutation_matrix)):
        for j in reversed(range(len(permutation_matrix))):
            if j <= i:
                continue
            if permutation_matrix[i][0] == permutation_matrix[j][0]:
                permutation_matrix.pop(j)
                logger.debug('Removing duplicate in permutation matrix'
                             'i: {} j: {}'.format(i, j))

    return permutation_matrix


def _fractional_to_cartesian(fractional_coordinates: List[List[float]],
                             cell: np.ndarray) -> List[float]:
    """
    Converts cell metrics from fractional to cartesian coordinates.

    Parameters
    ----------
    fractional_coordinates
        list of fractional coordinates

    cell
        cell metric
    """
    cartesian_coordinates = [np.dot(frac, cell)
                             for frac in fractional_coordinates]
    return cartesian_coordinates
