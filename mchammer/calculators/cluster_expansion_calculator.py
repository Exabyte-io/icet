from _icet import _ClusterExpansionCalculator
from ase import Atoms
from icet import ClusterExpansion
from mchammer.calculators.base_calculator import BaseCalculator
from typing import Union, List
from icet import Structure
from icet import ClusterSpace
import numpy as np


class ClusterExpansionCalculator(BaseCalculator):
    """A ClusterExpansionCalculator object enables the efficient
    calculation of properties described by a cluster expansion. It is
    specific for a particular (supercell) structure and commonly
    employed when setting up a Monte Carlo simulation, see
    :ref:`ensembles`.

    Cluster expansions, e.g., of the energy, typically yield property
    values *per site*. When running a Monte Carlo simulation one,
    however, considers changes in the *total* energy of the
    system. The default behavior is therefore to multiply the output
    of the cluster expansion by the number of sites. This behavior can
    be changed via the ``scaling`` keyword parameter.

    Parameters
    ----------
    atoms
        structure for which to set up the calculator
    cluster_expansion : ClusterExpansion
        cluster expansion from which to build calculator
    name
        human-readable identifier for this calculator
    scaling
        scaling factor applied to the property value predicted by the
        cluster expansion

    Todo
    ----
    * add OccupationConstraints once available

    """

    def __init__(self, atoms: Atoms, cluster_expansion: ClusterExpansion,
                 name: str='Cluster Expansion Calculator',
                 scaling: Union[float, int]=None) -> None:
        super().__init__(atoms=atoms, name=name)

        atoms_cpy = atoms.copy()
        self.cpp_calc = _ClusterExpansionCalculator(
            cluster_expansion.cluster_space, Structure.from_atoms(atoms_cpy))
        self._cluster_expansion = cluster_expansion
        self._local_cluster_space = ClusterSpace(
            self.cluster_expansion.cluster_space._atoms.copy(),
            self.cluster_expansion.cluster_space._cutoffs,
            self.cluster_expansion.cluster_space._chemical_symbols,
            self.cluster_expansion.cluster_space._mi)
        self._cluster_expansion = cluster_expansion
        if scaling is None:
            self._property_scaling = len(atoms)
        else:
            self._property_scaling = scaling

    @property
    def cluster_expansion(self) -> ClusterExpansion:
        """ cluster expansion from which calculator was constructed """
        return self._cluster_expansion

    def calculate_total(self, *, occupations: List[int]) -> float:
        """
        Calculates and returns the total property value of the current
        configuration.

        Parameters
        ----------
        occupations
            the entire occupation vector (i.e. list of atomic species)
        """
        self.atoms.set_atomic_numbers(occupations)
        return self.cluster_expansion.predict(self.atoms) * \
            self._property_scaling

    def calculate_local_contribution(self, *, local_indices: List[int],
                                     occupations: List[int]) -> float:
        """
        Calculates and returns the sum of the contributions to the property
        due to the sites specified in `local_indices`

        Parameters
        ----------
        local_indices
            sites over which to sum up the local contribution
        occupations
            entire occupation vector
        """

        self.atoms.set_atomic_numbers(occupations)
        local_contribution = 0
        exclude_indices = []  # type: List[int]

        for index in local_indices:
            local_contribution += self._calculate_local_contribution(
                index, exclude_indices=exclude_indices)
            exclude_indices.append(index)

        return local_contribution * self._property_scaling

    def _calculate_local_contribution(self, index: int,
                                      exclude_indices: List[int] = []):
        """
        Internal method to calculate the local contribution for one
        index.

        Parameters
        ----------
        index : int
            lattice index
        exclude_indices
            previously calculated indices, these indices will
            be ignored in order to avoid double counting bonds

        """
        local_cv = self.cpp_calc.get_local_cluster_vector(
            self.atoms.get_atomic_numbers(), index, exclude_indices)
        return np.dot(local_cv, self.cluster_expansion.parameters)

    @property
    def occupation_constraints(self) -> List[List[int]]:
        """ map from site to allowed species """
        species = list(
            self.cluster_expansion.cluster_space.species_map.keys())
        return [species] * len(self.atoms)
