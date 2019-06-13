"""
This module provides the OrbitList class.
"""

from typing import List

import numpy as np

from _icet import _OrbitList
from ase import Atoms
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.neighbor_list import get_neighbor_lists
from icet.core.permutation_matrix import (_get_lattice_site_permutation_matrix,
                                          permutation_matrix_from_atoms)
from icet.core.structure import Structure
from icet.io.logging import logger

logger = logger.getChild('orbit_list')


class OrbitList(_OrbitList):
    """
    The orbit list object has an internal list of orbits.

    An orbit has a list of equivalent sites with the restriction
    that at least one site is in the cell of the primitive structure.

    Parameters
    ----------
    atoms : ase.Atoms
            This atoms object will be used to construct a primitive
            structure on which all the lattice sites in the orbits
            are based on.
    cutoffs : list of float
              cutoffs[i] is the cutoff for orbits with order i+2.
    """

    def __init__(self, atoms_in, cutoffs):
        if isinstance(atoms_in, Structure):
            atoms_in = atoms_in.to_atoms()
        atoms = atoms_in.copy()
        atoms.wrap()
        max_cutoff = np.max(cutoffs)
        # Set up a permutation matrix
        permutation_matrix, prim_structure, _ \
            = permutation_matrix_from_atoms(atoms, max_cutoff, find_prim=False)

        logger.info('Done getting permutation_matrix.')

        # Get a list of neighbor-lists
        neighbor_lists = get_neighbor_lists(prim_structure, cutoffs)

        logger.info('Done getting neighbor lists.')

        # Transform permutation_matrix to be in lattice site format
        pm_lattice_sites \
            = _get_lattice_site_permutation_matrix(prim_structure,
                                                   permutation_matrix,
                                                   prune=True)

        logger.info('Transformation of permutation matrix to lattice neighbor'
                    'format completed.')

        _OrbitList.__init__(self, prim_structure,
                            pm_lattice_sites, neighbor_lists)
        self.sort()
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
            nice_str += "\norbit {} - Multiplicity {} - Cluster: {}".format(
                i, len(orbit), cluster_str)
        return nice_str

    def get_supercell_orbit_list(self, atoms: Atoms):
        """
        Returns an orbit list for a supercell structure.

        Parameters
        ----------
        atoms
            supercell atomic structure

        Returns
        -------
        An OrbitList object
        """
        structure = Structure.from_atoms(atoms)
        log = LocalOrbitListGenerator(self, structure)

        supercell_orbit_list = log.generate_full_orbit_list()

        return supercell_orbit_list

    def remove_inactive_orbits(self, allowed_species: List[List[str]]) -> None:
        """ Removes orbits with inactive sites

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
