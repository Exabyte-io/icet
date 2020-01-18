"""
This module provides the OrbitList class.
"""

from typing import List

import numpy as np

from _icet import _OrbitList
from ase import Atoms
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.neighbor_list import get_neighbor_lists
from icet.core.matrix_of_equivalent_positions import \
    _get_lattice_site_matrix_of_equivalent_positions, \
    matrix_of_equivalent_positions_from_structure
from icet.core.structure import Structure
from icet.input_output.logging_tools import logger

logger = logger.getChild('orbit_list')


class OrbitList(_OrbitList):
    """
    The orbit list object handles an internal list of orbits.

    An orbit has a list of equivalent sites with the restriction
    that at least one site is in the cell of the primitive structure.

    Parameters
    ----------
    structure : Atoms
        This structure will be used to construct a primitive
        structure on which all the lattice sites in the orbits
        are based.
    cutoffs : list of float
        the i-th element of this list is the cutoff for orbits with
        order i+2
    symprec : float
        tolerance imposed when analyzing the symmetry using spglib
    position_tolerance : float
        tolerance applied when comparing positions in Cartesian coordinates
    fractional_position_tolerance : float
        tolerance applied when comparing positions in fractional coordinates
    """

    def __init__(self,
                 structure: Atoms,
                 cutoffs: List[float],
                 symprec: float,
                 position_tolerance: float,
                 fractional_position_tolerance: float) -> None:
        max_cutoff = np.max(cutoffs)
        # Set up a permutation matrix
        matrix_of_equivalent_positions, prim_structure, _ \
            = matrix_of_equivalent_positions_from_structure(structure=structure,
                                                            cutoff=max_cutoff,
                                                            position_tolerance=position_tolerance,
                                                            find_primitive=False,
                                                            symprec=symprec)

        logger.info('Done getting matrix_of_equivalent_positions.')

        # Get a list of neighbor-lists
        neighbor_lists = get_neighbor_lists(prim_structure, cutoffs, position_tolerance)

        logger.info('Done getting neighbor lists.')

        # Transform matrix_of_equivalent_positions to be in lattice site format
        pm_lattice_sites \
            = _get_lattice_site_matrix_of_equivalent_positions(prim_structure,
                                                               matrix_of_equivalent_positions,
                                                               fractional_position_tolerance,
                                                               prune=True)

        logger.info('Transformation of matrix of equivalent positions'
                    ' to lattice neighbor format completed.')

        _OrbitList.__init__(
            self, prim_structure, pm_lattice_sites, neighbor_lists, position_tolerance)
        self.check_equivalent_clusters(position_tolerance)
        logger.info('Finished construction of orbit list.')

    @property
    def primitive_structure(self):
        """
        Returns the primitive structure to which the lattice sites in
        the orbits are referenced to.
        """
        return self._primitive_structure.copy()

    def __str__(self):
        """String representation."""
        nice_str = 'Number of orbits: {}'.format(len(self))

        for i, orbit in enumerate(self.orbits):
            cluster_str = self.orbits[i].representative_cluster.__str__()
            nice_str += '\norbit {} - Multiplicity {} - Cluster: {}'.format(
                i, len(orbit), cluster_str)
        return nice_str

    def get_supercell_orbit_list(self,
                                 structure: Atoms,
                                 position_tolerance: float):
        """
        Returns an orbit list for a supercell structure.

        Parameters
        ----------
        structure
            supercell atomic structure
        position_tolerance : float
            tolerance applied when comparing positions in Cartesian coordinates
        """
        log = LocalOrbitListGenerator(self, Structure.from_atoms(structure), position_tolerance)

        supercell_orbit_list = log.generate_full_orbit_list()

        return supercell_orbit_list

    def remove_inactive_orbits(self,
                               allowed_species: List[List[str]]) -> None:
        """ Removes orbits with inactive sites.

        Parameters
        ----------
        allowed_species
            the list of allowed species on each site in the primitive
            structure
        """
        prim_structure = self.get_primitive_structure()
        number_of_allowed_species = [len(sym) for sym in allowed_species]
        prim_structure.set_number_of_allowed_species(number_of_allowed_species)
        self._remove_inactive_orbits(prim_structure)
