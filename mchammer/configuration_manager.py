import random
from numpy import array
from typing import Dict, List, Tuple
from ase import Atoms
from icet.core.sublattices import Sublattices
from icet.tools.geometry import atomic_number_to_chemical_symbol


class SwapNotPossibleError(Exception):
    pass


class ConfigurationManager(object):
    """
    The ConfigurationManager owns and handles information pertaining to a
    configuration being sampled in a Monte Carlo simulation.

    Parameters
    ----------
    atoms : ASE Atoms
        configuration to be handled
    sublattices : Sublattices
        sublattices class used to define allowed occupations and so on

    Todo
    ----
    * revise docstrings
    """

    def __init__(self, atoms: Atoms, sublattices: Sublattices) -> None:

        self._atoms = atoms.copy()
        self._occupations = self._atoms.numbers
        self._sublattices = sublattices

        self._sites_by_species = self._get_sites_by_species()

    def _get_sites_by_species(self) -> List[Dict[int, List[int]]]:
        """Returns the sites that are occupied for each species.  Each
        dictionary represents one sublattice where the key is the
        species (by atomic number) and the value is the list of sites
        occupied by said species in the respective sublattice.
        """
        sites_by_species = []
        for sl in self._sublattices:
            species_dict = {key: [] for key in sl.atomic_numbers}
            for site in sl.indices:
                species_dict[self._occupations[site]].append(site)
            sites_by_species.append(species_dict)
        return sites_by_species

    @property
    def occupations(self) -> List[int]:
        """ occupation vector of the configuration (copy) """
        return self._occupations.copy()

    @property
    def sublattices(self) -> Sublattices:
        """sublattices of the configuration"""
        return self._sublattices

    @property
    def atoms(self) -> Atoms:
        """ atomic structure associated with configuration (copy) """
        atoms = self._atoms.copy()
        atoms.set_atomic_numbers(self.occupations)
        return atoms

    def is_swap_possible(self, sublattice_index: int) -> bool:
        """ Checks if swap is possible on specific sublattice.

        Parameters
        ----------
        sublattice_index
            index of sublattice to be checked
         """
        sl = self.sublattices[sublattice_index]
        return len(set(self.occupations[sl.indices])) > 1

    def get_swapped_state(self, sublattice_index: int) -> Tuple[List[int], List[int]]:
        """Returns two random sites (first element of tuple) and their
        occupation after a swap (second element of tuple).  The new
        configuration will obey the occupation constraints associated
        with the configuration mananger.

        Parameters
        ----------
        sublattice_index
            sublattice from which to pick sites
        """
        # pick the first site
        try:
            site1 = random.choice(self.sublattices[sublattice_index].indices)
        except IndexError:
            raise SwapNotPossibleError(
                'Sublattice {} is empty.'.format(sublattice_index))

        # pick the second site
        possible_swap_species = \
            set(self._sublattices.get_allowed_numbers_on_site(site1)) - \
            set([self._occupations[site1]])
        possible_swap_sites = []
        for Z in possible_swap_species:
            possible_swap_sites.extend(self._sites_by_species[sublattice_index][Z])

        possible_swap_sites = array(possible_swap_sites)

        try:
            site2 = random.choice(possible_swap_sites)
        except IndexError:
            raise SwapNotPossibleError(
                'Cannot swap on sublattice {} since it is full of {} species .'
                .format(sublattice_index,
                        atomic_number_to_chemical_symbol([self._occupations[site1]])[0]))

        return ([site1, site2], [self._occupations[site2], self._occupations[site1]])

    def get_flip_state(self, sublattice_index: int) -> Tuple[int, int]:
        """
        Returns a site index and a new species for the site.

        Parameters
        ----------
        sublattice_index
            index of sublattice from which to pick a site
        """

        site = random.choice(self._sublattices[sublattice_index].indices)
        species = random.choice(list(
            set(self._sublattices[sublattice_index].atomic_numbers) -
            set([self._occupations[site]])))
        return site, species

    def update_occupations(self, sites: List[int], species: List[int]):
        """
        Updates the occupation vector of the configuration being sampled.
        This will change the state in both the configuration in the calculator
        and the configuration manager.

        Parameters
        ----------
        sites
            indices of sites of the configuration to change
        species
            new occupations by atomic number
        """

        # Update _sites_by_sublattice
        for site, new_Z in zip(sites, species):
            if 0 > new_Z > 118:
                raise ValueError('Invalid new species {} on site {}'.format(new_Z, site))
            if len(self._occupations) >= site < 0:
                raise ValueError('Site {} is not a valid site index'.format(site))
            old_Z = self._occupations[site]
            sublattice_index = self.sublattices.get_sublattice_index(site)

            if new_Z not in self.sublattices[sublattice_index].atomic_numbers:
                raise ValueError('Invalid new species {} on site {}'.format(new_Z, site))

            # Remove site from list of sites for old species
            self._sites_by_species[sublattice_index][old_Z].remove(site)
            # Add site to list of sites for new species
            try:
                self._sites_by_species[sublattice_index][new_Z].append(site)
            except KeyError:
                raise ValueError('Invalid new species {} on site {}'.format(new_Z, site))

        # Update occupation vector itself
        self._occupations[sites] = species
