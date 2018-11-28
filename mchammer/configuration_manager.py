import copy
import random
from numpy import array
from typing import Dict, List, Tuple, Union
from ase import Atoms


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
    strict_constraints : list of list of int
        strictest form of the allowed occupations
    sites_by_sublattice : list of list of int
        sites (inner list) that belong to each sublattice (outer list)
    occupation_constraints : list of list of int
        optional occupation constraint to enfore a more stricter species
        occupation than what is allowed from the Calculator

    Todo
    ----
    * occupation constraint not implemented
    * add check that all sites in the different sublattices all have the same
      occupation constraint.
    * revise docstrings
    * clarify "occupation_constraints" vs "strict_constraints";
      the OccupationConstraints class should help here
    """

    def __init__(self, atoms: Atoms,
                 strict_constraints: Union[List[List[int]], List[int]],
                 sites_by_sublattice: Union[List[List[int]], List[int]],
                 occupation_constraints: List[List[int]] = None) -> None:

        self._atoms = atoms
        self._occupations = atoms.numbers
        self._sites_by_sublattice = sites_by_sublattice

        if occupation_constraints is not None:
            self._check_occupation_constraint(
                strict_constraints, occupation_constraints)
        else:
            occupation_constraints = strict_constraints
        self._occupation_constraints = occupation_constraints
        self._allowed_species = self._set_up_allowed_species()
        self._sites_by_species = self._get_sites_by_species()

    def _set_up_allowed_species(self) -> List[int]:
        """ Returns a list of allowed species. """
        allowed_species = set()
        for occ in self._occupation_constraints:
            for species in occ:
                allowed_species.add(species)
        return list(allowed_species)

    def _get_sites_by_species(self) -> List[Dict[int, List[int]]]:
        """Returns the sites that are occupied for each species.  Each
        dictionary represents one sublattice where the key is the
        species (by atomic number) and the value is the list of sites
        occupied by said species in the respective sublattice.
        """
        sites_by_species = []
        for sublattice in self._sites_by_sublattice:
            species_dict = {key: [] for key in self._allowed_species}
            for site in sublattice:
                species_dict[self._occupations[site]].append(site)
            sites_by_species.append(species_dict)
        return sites_by_species

    def _check_occupation_constraint(self,
                                     strict_constraints: Union[List[List[int]],
                                                               List[int]],
                                     occupation_constraints: List[List[int]]):
        """Checks that the user defined occupation constraints are stricter or
        as strict as the strict constraints.

        Parameters
        ----------
        strict_constraints
            additional (stricter) constraints specified for this
            configuration manager instance
        occupation_constraints
            "default" constraints

        Todo
        ----
        This method should be revised and rewritten once the
        OccupationConstraints class is available. At least some of
        this functionality should probably be moved into said class.
        """

        if not len(strict_constraints) == len(occupation_constraints):
            raise ValueError(
                'strict_occupations and occupation_constraints'
                ' must be equal length')

        for strict_occ, occ in zip(strict_constraints, occupation_constraints):
            if not set(occ).issubset(strict_occ):
                raise ValueError(
                    'User defined occupation_constraints must be '
                    'stricter or as strict as strict_occupations constraints.')

    @property
    def occupations(self) -> List[int]:
        """ occupation vector of the configuration (copy) """
        return self._occupations.copy()

    @property
    def occupation_constraints(self) ->List[List[int]]:
        """occupation constraints associated with configuration manager
        (copy); the outer list runs over sites with each inner list
        representing the species allowed on this site.
        """
        return copy.deepcopy(self._occupation_constraints)

    @property
    def sublattices(self) -> List[List[int]]:
        """ sites belonging to each sublattice of the configuration (copy) """
        return copy.deepcopy(self._sites_by_sublattice)

    @property
    def atoms(self) -> Atoms:
        """ atomic structure associated with configuration (copy) """
        atoms = self._atoms.copy()
        atoms.set_atomic_numbers(self.occupations)
        return atoms

    def get_swapped_state(self, sublattice: int) -> Tuple[List[int],
                                                          List[int]]:
        """Returns two random sites (first element of tuple) and their
        occupation after a swap (second element of tuple).  The new
        configuration will obey the occupation constraints associated
        with the configuration mananger.

        Parameters
        ----------
        sublattice
            sublattice from which to pick sites

        Todo
        ----
        * profile this method as it is called frequently
        * look for speed up opportunities
        * The current implementation assumes all sites in this sublattice to
          have the same allowed occupations.
        """
        # pick the first site
        try:
            site1 = random.choice(self._sites_by_sublattice[sublattice])
        except IndexError:
            raise SwapNotPossibleError('Sublattice {} is empty.'
                                       .format(sublattice))

        # pick the second site
        possible_swap_species = \
            set(self._occupation_constraints[site1]) - \
            set([self._occupations[site1]])
        possible_swap_sites = []
        for Z in possible_swap_species:
            possible_swap_sites.extend(self._sites_by_species[sublattice][Z])

        possible_swap_sites = array(possible_swap_sites)

        try:
            site2 = random.choice(possible_swap_sites)
        except IndexError:
            raise SwapNotPossibleError

        return ([site1, site2],
                [self._occupations[site2], self._occupations[site1]])

    def get_flip_state(self, sublattice: int) -> Tuple[int, int]:
        """
        Returns a site index and a new species for the site.

        Parameters
        ----------
        sublattice
            index of sublattice from which to pick a site
        """

        site = random.choice(self._sites_by_sublattice[sublattice])
        species = random.choice(list(
            set(self._occupation_constraints[site]) -
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
            old_Z = self._occupations[site]
            for isub, sublattice_sites in enumerate(self._sites_by_sublattice):
                if site in sublattice_sites:
                    break
            else:
                raise ValueError(
                    'Site {} is not present in any sublattice.'.format(site))

            # Remove site from list of sites for old species
            self._sites_by_species[isub][old_Z].remove(site)
            # Add site to list of sites for new species
            try:
                self._sites_by_species[isub][new_Z].append(site)
            except KeyError:
                raise ValueError('Invalid new species {} on site {}'
                                 .format(new_Z, site))

        # Update occupation vector itself
        self._occupations[sites] = species
