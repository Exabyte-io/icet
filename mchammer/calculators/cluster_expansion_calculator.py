from typing import List, Union

import numpy as np

from _icet import _ClusterExpansionCalculator
from icet.input_output.logging_tools import logger
from ase import Atoms
from icet import ClusterExpansion
from icet.core.structure import Structure
from icet.core.sublattices import Sublattices
from mchammer.calculators.base_calculator import BaseCalculator


class ClusterExpansionCalculator(BaseCalculator):
    """A :class:`ClusterExpansionCalculator` object enables the efficient
    calculation of properties described by a cluster expansion. It is
    specific for a particular (supercell) structure and commonly
    employed when setting up a Monte Carlo simulation, see
    :ref:`ensembles`.

    Cluster expansions, e.g., of the energy, typically yield property
    values *per site*. When running a Monte Carlo simulation one,
    however, considers changes in the *total* energy of the
    system. The default behavior is therefore to multiply the output
    of the cluster expansion by the number of sites. This behavior can
    be changed via the :attr:`scaling` keyword parameter.

    Parameters
    ----------
    structure
        Structure for which to set up the calculator.
    cluster_expansion
        Cluster expansion from which to build calculator.
    name
        Human-readable identifier for this calculator.
    scaling
        Scaling factor applied to the property value predicted by the
        cluster expansion.
    use_local_energy_calculator
        Evaluate energy changes using only the local environment; this method
        is generally *much* faster. Unless you know what you are doing do *not*
        set this option to ``False``.
    """

    def __init__(self,
                 structure: Atoms, cluster_expansion: ClusterExpansion,
                 name: str = 'Cluster Expansion Calculator',
                 scaling: Union[float, int] = None,
                 use_local_energy_calculator: bool = True) -> None:
        super().__init__(name=name)

        structure_cpy = structure.copy()
        cluster_expansion.prune()

        if cluster_expansion._cluster_space.is_supercell_self_interacting(structure):
            logger.warning('The ClusterExpansionCalculator self-interacts, '
                           'which may lead to erroneous results. To avoid '
                           'self-interaction, use a larger supercell or a '
                           'cluster space with shorter cutoffs.')

        self.use_local_energy_calculator = use_local_energy_calculator
        self.cpp_calc = _ClusterExpansionCalculator(
            cluster_space=cluster_expansion.get_cluster_space_copy(),
            structure=Structure.from_atoms(structure_cpy),
            fractional_position_tolerance=cluster_expansion.fractional_position_tolerance)

        self._cluster_expansion = cluster_expansion
        if scaling is None:
            self._property_scaling = len(structure)
        else:
            self._property_scaling = scaling

        self._sublattices = self.cluster_expansion._cluster_space.get_sublattices(structure)

    @property
    def cluster_expansion(self) -> ClusterExpansion:
        """ Cluster expansion from which calculator was set up. """
        return self._cluster_expansion

    def calculate_total(self, *, occupations: List[int]) -> float:
        """
        Calculates and returns the total property value of the current
        configuration.

        Parameters
        ----------
        occupations
            The entire occupation vector (i.e., list of atomic species).
        """

        cv = self.cpp_calc.get_cluster_vector(occupations)
        return np.dot(cv, self.cluster_expansion.parameters) * self._property_scaling

    def calculate_change(self, *, sites: List[int],
                         current_occupations: List[int],
                         new_site_occupations: List[int]) -> float:
        """
        Calculates and returns the sum of the contributions to the property
        due to the sites specified in :attr:`sites`.

        Parameters
        ----------
        sites
            Indices of sites at which occupations will be changed.
        current_occupations
            Entire occupation vector (atomic numbers) before change.
        new_site_occupations
            Atomic numbers after change at the sites defined by :attr:`sites`.
        """
        occupations = np.array(current_occupations)

        if not self.use_local_energy_calculator:
            e_before = self.calculate_total(occupations=occupations)
            occupations[sites] = np.array(new_site_occupations)
            e_after = self.calculate_total(occupations=occupations)
            return e_after - e_before

        change = 0.0
        try:
            for index, new_occupation in zip(sites, new_site_occupations):
                change += self._calculate_partial_change(occupations=occupations,
                                                         flip_index=index,
                                                         new_occupation=new_occupation)
                occupations[index] = new_occupation  # Safe because we work with a copy
        except Exception as e:
            msg = 'Caught exception {}. Try setting parameter '.format(e)
            msg += 'use_local_energy_calculator to False in init'
            raise RuntimeError(msg)
        return change * self._property_scaling

    def _calculate_partial_change(self, occupations: List[int], flip_index: int,
                                  new_occupation: int):
        """
        Internal method to calculate the local contribution for one site index.

        Parameters
        ----------
        occupations
            Entire occupation vector.
        flip_index
            Lattice index for site where a flipped has occurred.
        new_occupation
            Atomic number of new occupation at site :attr:`flip_index`.
        """
        cv_change = self.cpp_calc.get_cluster_vector_change(occupations=occupations,
                                                            flip_index=flip_index,
                                                            new_occupation=new_occupation)

        return np.dot(cv_change, self.cluster_expansion.parameters)

    @property
    def sublattices(self) -> Sublattices:
        """ Sublattices of the calculator structure. """
        return self._sublattices
