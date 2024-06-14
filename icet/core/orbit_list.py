"""
This module provides the OrbitList class.
"""

from typing import List, Dict
from collections import Counter

import numpy as np

from _icet import _OrbitList
from icet.core.orbit import Orbit # noqa
from ase import Atoms
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.neighbor_list import get_neighbor_lists
from icet.core.matrix_of_equivalent_positions import \
    _get_lattice_site_matrix_of_equivalent_positions, \
    matrix_of_equivalent_positions_from_structure
from icet.core.structure import Structure
from icet.tools.geometry import (chemical_symbols_to_numbers,
                                 atomic_number_to_chemical_symbol)
from icet.input_output.logging_tools import logger

logger = logger.getChild('orbit_list')


class OrbitList(_OrbitList):
    """
    The orbit list object handles an internal list of orbits.

    An orbit has a list of equivalent sites with the restriction
    that at least one site is in the cell of the primitive structure.

    Note
    ----
    As a user you will usually not interact directly with objects of this type.

    Parameters
    ----------
    structure
        This structure will be used to construct a primitive
        structure on which all the lattice sites in the orbits
        are based.
    cutoffs
        The `i`-th element of this list is the cutoff for orbits with
        order `i+2`.
    chemical_symbols
        List of chemical symbols, each of which must map to an element
        of the periodic table.

        The outer list must be the same length as the `structure` object
        and :attr:`chemical_symbols[i]` will correspond to the allowed species
        on lattice site ``i``.
    symprec
        Tolerance imposed when analyzing the symmetry using spglib.
    position_tolerance
        Tolerance applied when comparing positions in Cartesian coordinates.
    fractional_position_tolerance
        Tolerance applied when comparing positions in fractional coordinates.
    """

    def __init__(self,
                 structure: Atoms,
                 cutoffs: List[float],
                 chemical_symbols: List[List[str]],
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
        prim_structure.allowed_atomic_numbers = [chemical_symbols_to_numbers(syms)
                                                 for syms in chemical_symbols]

        logger.info('Done getting matrix_of_equivalent_positions.')

        # Get a list of neighbor-lists
        neighbor_lists = get_neighbor_lists(structure=prim_structure, cutoffs=cutoffs,
                                            position_tolerance=position_tolerance)

        logger.info('Done getting neighbor lists.')

        # Transform matrix_of_equivalent_positions to be in lattice site format
        pm_lattice_sites = _get_lattice_site_matrix_of_equivalent_positions(
            structure=prim_structure,
            matrix_of_equivalent_positions=matrix_of_equivalent_positions,
            fractional_position_tolerance=fractional_position_tolerance,
            prune=True)

        logger.info('Transformation of matrix of equivalent positions'
                    ' to lattice neighbor format completed.')

        _OrbitList.__init__(self,
                            structure=prim_structure,
                            matrix_of_equivalent_sites=pm_lattice_sites,
                            neighbor_lists=neighbor_lists,
                            position_tolerance=position_tolerance)
        logger.info('Finished construction of orbit list.')

    @property
    def primitive_structure(self):
        """
        A copy of the primitive structure to which the lattice sites in
        the orbits are referenced to.
        """
        return self.get_structure().to_atoms()

    def __str__(self):
        """ String representation. """
        s = []
        s += ['Number of orbits: {}'.format(len(self))]
        for k, orbit in enumerate(self.orbits):
            s += [f'Orbit {k}:']
            s += [f'\t{s_tmp}' for s_tmp in orbit.__str__().split('\n')][:-1]
        return '\n'.join(s)

    def __getitem__(self, ind: int):
        return self.get_orbit(ind)

    def get_supercell_orbit_list(self,
                                 structure: Atoms,
                                 fractional_position_tolerance: float):
        """
        Returns the orbit list for a supercell structure.

        Parameters
        ----------
        structure
            Atomic structure.
        fractional_position_tolerance
            Tolerance applied when comparing positions in fractional coordinates.
        """
        lolg = LocalOrbitListGenerator(
            self,
            structure=Structure.from_atoms(structure),
            fractional_position_tolerance=fractional_position_tolerance)
        supercell_orbit_list = lolg.generate_full_orbit_list()
        return supercell_orbit_list

    def get_cluster_counts(self,
                           structure: Atoms,
                           fractional_position_tolerance: float,
                           orbit_indices: List[int] = None) -> Dict[int, Counter]:
        """
        Counts all clusters in a structure by finding their local orbit list.

        Parameters
        ----------
        structure
            Structure for which to count clusters. This structure needs to
            be commensurate with the structure this orbit list is based on.
        fractional_position_tolerance
            Tolerance applied when comparing positions in fractional coordinates.
        orbit_indices
            Indices of orbits, for which counts are requested; if ``None`` all
            orbits will be counted.

        Returns
        -------
            Dictionary, the keys of which are orbit indices and the values
            cluster counts. The latter are themselves dicts, with tuples
            of chemical symbols as keys and the number of such clusters
            as values.
        """
        supercell_orbit_list = self.get_supercell_orbit_list(
            structure=structure,
            fractional_position_tolerance=fractional_position_tolerance)

        # Collect counts for all orbit_indices
        if orbit_indices is None:
            orbit_indices = range(len(self))
        structure_icet = Structure.from_atoms(structure)
        cluster_counts_full = {}
        for i in orbit_indices:
            orbit = supercell_orbit_list.get_orbit(i)
            counts = orbit.get_cluster_counts(structure_icet)
            sorted_counts = Counter()
            for atomic_numbers, count in counts.items():
                symbols = atomic_number_to_chemical_symbol(atomic_numbers)
                sorted_symbols = tuple(sorted(symbols))
                sorted_counts[sorted_symbols] += count
            cluster_counts_full[i] = sorted_counts
        return cluster_counts_full
