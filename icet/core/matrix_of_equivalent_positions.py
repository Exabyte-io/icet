"""
This module provides a Python interface to the MatrixOfEquivalentPositions
class with supplementary functions.
"""

from typing import List, Tuple

import numpy as np
import spglib

from ase import Atoms
from _icet import MatrixOfEquivalentPositions
from icet.core.lattice_site import LatticeSite
from icet.core.neighbor_list import NeighborList
from icet.core.structure import Structure
from icet.input_output.logging_tools import logger
from icet.tools.geometry import (ase_atoms_to_spglib_cell,
                                 get_fractional_positions_from_neighbor_list,
                                 get_primitive_structure)

logger = logger.getChild('matrix_of_equivalent_positions')


def matrix_of_equivalent_positions_from_structure(structure: Atoms,
                                                  cutoff: float,
                                                  position_tolerance: float,
                                                  symprec: float,
                                                  find_primitive: bool = True) \
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
    symprec
        tolerance imposed when analyzing the symmetry using spglib
    position_tolerance
        tolerance applied when comparing positions in Cartesian coordinates

    Returns
    -------
    The tuple that is returned comprises the permutation matrix, the
    primitive structure, and the neighbor list.
    """

    structure = structure.copy()
    structure_prim = structure
    if find_primitive:
        structure_prim = get_primitive_structure(structure, symprec=symprec)
    logger.debug('Size of primitive structure: {}'.format(len(structure_prim)))

    # get symmetry information
    structure_as_tuple = ase_atoms_to_spglib_cell(structure_prim)
    symmetry = spglib.get_symmetry(structure_as_tuple, symprec=symprec)
    translations = symmetry['translations']
    rotations = symmetry['rotations']

    # set up a permutation map object
    matrix_of_equivalent_positions = MatrixOfEquivalentPositions(translations, rotations)

    # create neighbor_lists from the different cutoffs
    prim_icet_structure = Structure.from_atoms(structure_prim)
    neighbor_list = NeighborList(cutoff)
    neighbor_list.build(prim_icet_structure, position_tolerance)

    # get fractional positions for neighbor_list
    frac_positions = get_fractional_positions_from_neighbor_list(
        prim_icet_structure, neighbor_list)

    logger.debug('Number of fractional positions: {}'.format(len(frac_positions)))
    if frac_positions is not None:
        matrix_of_equivalent_positions.build(frac_positions)

    return matrix_of_equivalent_positions, prim_icet_structure, neighbor_list


def _get_lattice_site_matrix_of_equivalent_positions(
        structure: Structure,
        matrix_of_equivalent_positions: MatrixOfEquivalentPositions,
        fractional_position_tolerance: float,
        prune: bool = True) -> np.ndarray:
    """
    Returns a transformed permutation matrix with lattice sites as entries
    instead of fractional coordinates.

    Parameters
    ----------
    structure
        primitive atomic icet structure
    matrix_of_equivalent_positions
        permutation matrix with fractional coordinates format entries
    fractional_position_tolerance
        tolerance applied when evaluating distances in fractional coordinates
    prune
        if True the permutation matrix will be pruned

    Returns
    -------
    permutation matrix in a row major order with lattice site format entries
    """
    pm_frac = matrix_of_equivalent_positions.get_equivalent_positions()

    pm_lattice_sites = []
    for row in pm_frac:
        positions = _fractional_to_cartesian(row, structure.cell)
        lattice_sites = []
        if np.all(structure.pbc):
            lattice_sites = \
                structure.find_lattice_sites_by_positions(positions, fractional_position_tolerance)
        else:
            for pos in positions:
                try:
                    lattice_site = \
                        structure.find_lattice_site_by_position(pos, fractional_position_tolerance)
                except RuntimeError:
                    continue
                lattice_sites.append(lattice_site)
        if lattice_sites is not None:
            pm_lattice_sites.append(lattice_sites)
        else:
            logger.warning('Unable to transform any element in a column of the'
                           ' fractional permutation matrix to lattice site')
    if prune:
        logger.debug('Size of columns of the permutation matrix before'
                     ' pruning {}'.format(len(pm_lattice_sites)))

        pm_lattice_sites = _prune_matrix_of_equivalent_positions(pm_lattice_sites)

        logger.debug('Size of columns of the permutation matrix after'
                     ' pruning {}'.format(len(pm_lattice_sites)))

    return pm_lattice_sites


def _prune_matrix_of_equivalent_positions(matrix_of_equivalent_positions: List[List[LatticeSite]]):
    """
    Prunes the matrix so that the first column only contains unique elements.

    Parameters
    ----------
    matrix_of_equivalent_positions
        permutation matrix with LatticeSite type entries
    """

    for i in range(len(matrix_of_equivalent_positions)):
        for j in reversed(range(len(matrix_of_equivalent_positions))):
            if j <= i:
                continue
            if matrix_of_equivalent_positions[i][0] == matrix_of_equivalent_positions[j][0]:
                matrix_of_equivalent_positions.pop(j)
                logger.debug('Removing duplicate in permutation matrix'
                             'i: {} j: {}'.format(i, j))
    return matrix_of_equivalent_positions


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
