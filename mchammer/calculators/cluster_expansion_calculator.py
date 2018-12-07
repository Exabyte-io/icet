from _icet import _ClusterExpansionCalculator
from ase import Atoms
from icet import ClusterExpansion
from mchammer.calculators.base_calculator import BaseCalculator
from typing import Union, List
from icet import Structure
from icet.tools.geometry import find_lattice_site_by_position
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
    use_local_energy_calculator
        evaluate energy changes using only the local environment; this method
        is generally *much* faster; unless you know what you are doing do *not*
        set this option to `False`
    """

    def __init__(self, atoms: Atoms, cluster_expansion: ClusterExpansion,
                 name: str = 'Cluster Expansion Calculator',
                 scaling: Union[float, int] = None,
                 use_local_energy_calculator: bool = True) -> None:
        super().__init__(atoms=atoms, name=name)

        atoms_cpy = atoms.copy()
        cluster_expansion.prune()

        self.use_local_energy_calculator = use_local_energy_calculator
        if self.use_local_energy_calculator:
            self.cpp_calc = _ClusterExpansionCalculator(
                cluster_expansion.cluster_space,
                Structure.from_atoms(atoms_cpy))

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
        if not self.use_local_energy_calculator:
            return self.calculate_total(occupations=occupations)

        self.atoms.set_atomic_numbers(occupations)

        local_contribution = 0
        exclude_indices = []  # type: List[int]

        for index in local_indices:
            try:
                local_contribution += self._calculate_local_contribution(
                    index, exclude_indices=exclude_indices)
            except Exception as e:
                msg = "caugh exception {}. Try setting flag ".format(e)
                msg += "`use_local_energy_calculator to False` in init"
                raise RuntimeError(msg)

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
        allowed_species_prim = \
            self.cluster_expansion.cluster_space.chemical_symbols
        indices_in_prim = [find_lattice_site_by_position(
            self.cluster_expansion.cluster_space.primitive_structure,
            position=pos).index for pos in self.atoms.positions]
        allowed_species = [allowed_species_prim[i] for i in indices_in_prim]
        return allowed_species
