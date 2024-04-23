"""
This module provides a Python interface to the MatrixOfEquivalentPositions
class with supplementary functions.
"""

from typing import List, Tuple

import numpy as np
import spglib

from ase import Atoms
from icet.core.lattice_site import LatticeSite
from icet.core.neighbor_list import get_neighbor_lists
from icet.core.structure import Structure
from icet.input_output.logging_tools import logger
from icet.tools.geometry import (ase_atoms_to_spglib_cell,
                                 get_fractional_positions_from_neighbor_list,
                                 get_primitive_structure)

logger = logger.getChild('matrix_of_equivalent_positions')


class MatrixOfEquivalentPositions:
    """
    This class handles a matrix of equivalent positions given the symmetry
    elements of an atomic structure.

    Note
    ----
    As a user you will usually not interact directly with objects of this type.

    Parameters
    ----------
    translations
        Translational symmetry elements.
    rotations
        Rotational symmetry elements.
    """

    def __init__(self, translations: np.ndarray, rotations: np.ndarray):
        if len(translations) != len(rotations):
            raise ValueError(f'The number of translations ({len(translations)})'
                             f' must equal the number of rotations ({len(rotations)}).')
        self.n_symmetries = len(rotations)
        self.translations = np.array(translations)
        self.rotations = np.array(rotations)

    def build(self, fractional_positions: np.ndarray) -> None:
        """
        Builds a matrix of symmetry equivalent positions given a set of input
        coordinates using the rotational and translational symmetries provided upon
        initialization of the object.

        Parameters
        ----------
        fractional_positions
            Atomic positions in fractional coordinates.
            Dimensions: (number of atoms, 3 fractional coordinates).
        """
        positions = np.dot(self.rotations, fractional_positions.transpose())
        positions = np.moveaxis(positions, 2, 0)
        translations = self.translations[np.newaxis, :].repeat(len(fractional_positions), axis=0)
        positions += translations
        self.positions = positions

    def get_equivalent_positions(self) -> np.ndarray:
        """
        Returns the matrix of equivalent positions. Each row corresponds
        to a set of symmetry equivalent positions. The entry in the
        first column is commonly treated as the representative position.
        Dimensions: (number of atoms, number of symmetries, 3 fractional coordinates)
        """
        return self.positions


def matrix_of_equivalent_positions_from_structure(structure: Atoms,
                                                  cutoff: float,
                                                  position_tolerance: float,
                                                  symprec: float,
                                                  find_primitive: bool = True) \
        -> Tuple[np.ndarray, Structure, List]:
    """Sets up a matrix of equivalent positions from an :class:`Atoms <ase.Atoms>` object.

    Parameters
    ----------
    structure
        Input structure.
    cutoff
        Cutoff radius.
    find_primitive
        If ``True`` the symmetries of the primitive structure will be employed.
    symprec
        Tolerance imposed when analyzing the symmetry using spglib.
    position_tolerance
        Tolerance applied when comparing positions in Cartesian coordinates.

    Returns
    -------
        The tuple that is returned comprises the matrix of equivalent positions,
        the primitive structure, and the neighbor list.
    """

    structure = structure.copy()
    structure_prim = structure
    if find_primitive:
        structure_prim = get_primitive_structure(structure, symprec=symprec)
    logger.debug(f'Size of primitive structure: {len(structure_prim)}')

    # get symmetry information
    structure_as_tuple = ase_atoms_to_spglib_cell(structure_prim)
    symmetry = spglib.get_symmetry(structure_as_tuple, symprec=symprec)
    translations = symmetry['translations']
    rotations = symmetry['rotations']

    # set up a MatrixOfEquivalentPositions object
    matrix_of_equivalent_positions = MatrixOfEquivalentPositions(translations, rotations)

    # create neighbor lists
    prim_icet_structure = Structure.from_atoms(structure_prim)

    neighbor_list = get_neighbor_lists(prim_icet_structure,
                                       [cutoff],
                                       position_tolerance=position_tolerance)[0]

    # get fractional positions for neighbor_list
    frac_positions = get_fractional_positions_from_neighbor_list(
        prim_icet_structure, neighbor_list)

    logger.debug(f'Number of fractional positions: {len(frac_positions)}')
    if frac_positions is not None:
        matrix_of_equivalent_positions.build(frac_positions)

    return matrix_of_equivalent_positions, prim_icet_structure, neighbor_list


def _get_lattice_site_matrix_of_equivalent_positions(
        structure: Structure,
        matrix_of_equivalent_positions: MatrixOfEquivalentPositions,
        fractional_position_tolerance: float,
        prune: bool = True) -> np.ndarray:
    """
    Returns a transformed matrix of equivalent positions with lattice sites as
    entries instead of fractional coordinates.

    Parameters
    ----------
    structure
        Primitive structure.
    matrix_of_equivalent_positions
        Matrix of equivalent positions with fractional coordinates format entries.
    fractional_position_tolerance
        Tolerance applied when evaluating distances in fractional coordinates.
    prune
        If ``True`` the matrix of equivalent positions will be pruned.

    Returns
    -------
        Matrix of equivalent positions in row major order with entries in lattice site format.
    """
    eqpos_frac = matrix_of_equivalent_positions.get_equivalent_positions()

    eqpos_lattice_sites = []
    for row in eqpos_frac:
        positions = _fractional_to_cartesian(row, structure.cell)
        lattice_sites = []
        if np.all(structure.pbc):
            lattice_sites = structure.find_lattice_sites_by_positions(
                positions=positions, fractional_position_tolerance=fractional_position_tolerance)
        else:
            raise ValueError('Input structure must have periodic boundary conditions.')
        if lattice_sites is not None:
            eqpos_lattice_sites.append(lattice_sites)
        else:
            logger.warning('Unable to transform any element in a column of the'
                           ' fractional matrix of equivalent positions to lattice site')
    if prune:
        logger.debug('Size of columns of the matrix of equivalent positions before'
                     ' pruning {}'.format(len(eqpos_lattice_sites)))

        eqpos_lattice_sites = _prune_matrix_of_equivalent_positions(eqpos_lattice_sites)

        logger.debug('Size of columns of the matrix of equivalent positions after'
                     ' pruning {}'.format(len(eqpos_lattice_sites)))

    return eqpos_lattice_sites


def _prune_matrix_of_equivalent_positions(matrix_of_equivalent_positions: List[List[LatticeSite]]):
    """
    Prunes the matrix so that the first column only contains unique elements.

    Parameters
    ----------
    matrix_of_equivalent_positions
        Permutation matrix with :class:`LatticeSite` type entries.
    """

    for i in range(len(matrix_of_equivalent_positions)):
        for j in reversed(range(len(matrix_of_equivalent_positions))):
            if j <= i:
                continue
            if matrix_of_equivalent_positions[i][0] == matrix_of_equivalent_positions[j][0]:
                matrix_of_equivalent_positions.pop(j)
                logger.debug('Removing duplicate in matrix of equivalent positions'
                             'i: {} j: {}'.format(i, j))
    return matrix_of_equivalent_positions


def _fractional_to_cartesian(fractional_coordinates: List[List[float]],
                             cell: np.ndarray) -> List[float]:
    """
    Converts cell metrics from fractional to Cartesian coordinates.

    Parameters
    ----------
    fractional_coordinates
        List of fractional coordinates.
    cell
        Cell metric.
    """
    cartesian_coordinates = [np.dot(frac, cell)
                             for frac in fractional_coordinates]
    return cartesian_coordinates
