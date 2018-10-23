"""
Definition of the variance-constrained semi-grand canonical ensemble class.
"""

import numpy as np

from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.units import kB
from typing import Dict

from .. import DataContainer
from .base_ensemble import BaseEnsemble
from ..calculators.base_calculator import BaseCalculator


class VCSGCEnsemble(BaseEnsemble):
    """Variance-constrained semi-grand canonical (VCSGC) ensemble.

    Instances of this class allow one to simulate systems in the VCSGC
    ensemble (:math:`N\phi\kappa VT`), i.e. at constant temperature
    (:math:`T`), total number of sites (:math:`N=\sum_i N_i`),
    and two additional parameters :math:`\phi` and :math:`\kappa`, which
    constrain the concentration and variance of the concentration,
    respectively. The derivative of the canonical free energy can be
    expressed in observables of the ensemble,

    .. math::

        \\frac{1}{N} \\frac{\\partial F}{\\partial c} = - \phi - 2 N \kappa
        \\langle c \\rangle.

    Unlike the SGC ensemble, the VCSGC ensemble allows for sampling across
    multi-phase regions, meaning that the free energy can be recovered by
    direct integration even when such regions are present.

    The VCSGC ensemble currently supports systems with no more than two
    different species.

    When using this ensemble, please cite
    Sadigh, B. and Erhart, P., Phys. Rev. B **86**, 134204 (2012)
    [SadErh12]_.

    Attributes
    -----------
    temperature : float
        temperature :math:`T` in appropriate units [commonly Kelvin]
    boltzmann_constant : float
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e. units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K]
    kappa : float
        parameter that constrains the fluctuations of the concentration
        (referred to as :math:`\bar{\kappa}` in [SadErh12]_)
    """

    def __init__(self, atoms: Atoms=None, calculator: BaseCalculator=None,
                 name: str='Variance-constrained semi-grand' +
                           ' canonical ensemble',
                 data_container: DataContainer=None, random_seed: int=None,
                 data_container_write_period: float=np.inf,
                 ensemble_data_write_interval: int=None,
                 trajectory_write_interval: int=None,
                 boltzmann_constant: float=kB, *, temperature: float,
                 phis: Dict[str, float],
                 kappa: float):

        super().__init__(
            atoms=atoms, calculator=calculator, name=name,
            data_container=data_container,
            random_seed=random_seed,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

        self.temperature = temperature
        self.boltzmann_constant = boltzmann_constant

        self._phis = None
        self.phis = phis
        self.kappa = kappa

        if len(self.configuration._allowed_species) > 2:
            raise NotImplementedError('VCSGCEnsemble does not yet support '
                                      'cluster spaces with more than two '
                                      'species.')

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        self.total_trials += 1

        # choose flip
        sublattice_index = self.get_random_sublattice_index()
        index, new_species = \
            self.configuration.get_flip_state(sublattice_index)
        old_species = self.configuration.occupations[index]

        # Calculate difference in VCSGC thermodynamic potential.
        # Note that this assumes that only one atom was flipped.
        N = len(self.atoms)
        occupations = self.configuration._occupations.tolist()
        potential_diff = 1.0  # dN
        potential_diff -= occupations.count(old_species)
        potential_diff -= 0.5 * N * self.phis[old_species]
        potential_diff -= occupations.count(new_species)
        potential_diff += 0.5 * N * self._phis[new_species]
        potential_diff *= self.kappa
        potential_diff /= N

        potential_diff += self._get_property_change([index], [new_species])

        if self._acceptance_condition(potential_diff):
            self.accepted_trials += 1
            self.update_occupations([index], [new_species])

    def _acceptance_condition(self, potential_diff: float) -> bool:
        """
        Evaluates Metropolis acceptance criterion.

        Parameters
        ----------
        potential_diff
            the change in the thermodynamic potential associated
            with the trial step
        """
        if potential_diff < 0:
            return True
        else:
            return np.exp(-potential_diff / (
                self.boltzmann_constant * self.temperature)) > \
                self._next_random_number()

    @property
    def phis(self) -> Dict[int, float]:
        """phis :math:`\\phi_i`, one for each
        element but their sum must be :math:`-2.0`
        (referred to as :math:`\bar{\phi}` in [SadErh12]_)"""
        return self._phis

    @phis.setter
    def phis(self, phis):
        if not isinstance(phis, dict):
            raise TypeError('phis must be dict, not {}'.format(type(phis)))
        if abs(sum(phis.values()) + 2) > 1e-6:
            raise ValueError('The sum of all phis must equal -2')

        self._phis = {}

        for key, phi in phis.items():
            if isinstance(key, str):
                atomic_number = atomic_numbers[key]
                self._phis[atomic_number] = phi
            elif isinstance(key, int):
                self._phis[key] = phi
        if set(self.configuration._allowed_species) != set(self._phis.keys()):
            raise ValueError('phis were not set for all species')

    def _get_ensemble_data(self) -> Dict:
        """
        Returns a dict with the default data of
        the ensemble.

        Here temperature and species counts
        are added to the default data.

        Returns
        -------
        dict : ensemble data key pairs

        """
        data = super()._get_ensemble_data()

        # concentration parameters (phis)
        for atnum, phi in self.phis.items():
            data['phi_{}'.format(chemical_symbols[atnum])] = phi

        # variance parameter (kappa)
        data['kappa'] = self.kappa

        # free energy derivative
        atnum_1 = min(self.phis.keys())
        concentration = self.configuration._occupations.tolist().count(
            atnum_1) / len(self.atoms)
        data['free_energy_derivative'] = \
            - 2 * self.kappa * concentration - self.kappa * self.phis[atnum_1]

        # temperature
        data['temperature'] = self.temperature

        # species counts
        atoms = self.configuration.atoms
        unique, counts = np.unique(atoms.numbers, return_counts=True)
        # TODO: avoid accessing a protected member of a client class
        for atnum in self.configuration._allowed_species:
            data['{}_count'.format(chemical_symbols[atnum])] = 0
        for atnum, count in zip(unique, counts):
            data['{}_count'.format(chemical_symbols[atnum])] = count

        return data
