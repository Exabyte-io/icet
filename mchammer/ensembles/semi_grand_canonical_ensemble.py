"""
Definition of the semi-grand canonical ensemble class.
"""

import numpy as np

from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.units import kB
from collections import OrderedDict
from typing import Dict, Union

from .. import DataContainer
from .base_ensemble import BaseEnsemble
from ..calculators.base_calculator import BaseCalculator


class SemiGrandCanonicalEnsemble(BaseEnsemble):
    """Semi-grand canonical (SGC) ensemble.

    Instances of this class allow one to simulate systems in the SGC
    ensemble (:math:`N\Delta\mu_i VT`), i.e. at constant temperature
    (:math:`T`), total number of sites (:math:`N=\sum_i N_i`),
    relative chemical potentials (:math:`\Delta\mu_i=\mu_i - \mu_1`,
    where :math:`i` denotes the species), and volume (:math:`V`).

    The probability density of the SGC ensemble for a
    :math:`m`-component system is

    .. math::

        \\rho_{\\text{SGC}} = \exp\\Big[ - \\big( E
        + \sum_{i>1}^m \Delta\mu_i N_i \\big) / k_B T \\Big]

    with the *relative* chemical potentials :math:`\Delta\mu_i = \mu_i -
    \mu_1` and species counts :math:`N_i`. Unlike the
    :class:`CanonicalEnsemble<mchammer.ensembles.CanonicalEnsemble>`, the
    number of the respecive species (or, equivalently, the concentrations) are
    allowed to vary in the SGC ensemble. A trial step thus consists of
    randomly picking an atom and changing its identity with a probability

    .. math::
        P = \min \{ 1, \, \exp [ - \\Delta E / k_B T  ] \},

    where :math:`\\Delta E` is the change in potential energy caused by the
    swap.

    The canonical ensemble provides an ideal framework for studying the
    properties of a system at a specific concentrations. Properties such as
    potential energy and chemical ordering at a specific temperature can
    conveniently be studied by simulating at that temperature. The canonical
    ensemble is also a convenient tool for "optimizing" a system, i.e.,
    finding its lowest energy chemical ordering. In practice, this is usually
    achieved by simulated annealing, i.e. the system is equilibrated at a high
    temperature, after which the temperature is continuously lowered until the
    acceptance probability is almost zero. In a well-behaved system, the
    chemical ordering at that point corresponds to a low-energy structure,
    possibly the global minimum at that particular concentration.

    Attributes
    -----------
    temperature : float
        temperature :math:`T` in appropriate units [commonly Kelvin]
    boltzmann_constant : float
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e. units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K]

    """

    def __init__(self, atoms: Atoms=None, calculator: BaseCalculator=None,
                 name: str='Semi-grand canonical ensemble',
                 data_container: DataContainer=None, random_seed: int=None,
                 data_container_write_period: float=np.inf,
                 ensemble_data_write_interval: int=None,
                 trajectory_write_interval: int=None,
                 boltzmann_constant: float=kB, *, temperature: float,
                 chemical_potentials: Dict[str, float]):

        super().__init__(
            atoms=atoms, calculator=calculator, name=name,
            data_container=data_container,
            random_seed=random_seed,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

        self.temperature = temperature
        self.boltzmann_constant = boltzmann_constant

        self._chemical_potentials = None
        self.chemical_potentials = chemical_potentials

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        self.total_trials += 1

        # energy change
        sublattice_index = self.get_random_sublattice_index()
        index, species = \
            self.configuration.get_flip_state(sublattice_index)
        potential_diff = self._get_property_change([index], [species])

        # change in chemical potential
        old_species = self.configuration.occupations[index]
        chemical_potential_diff = \
            self.chemical_potentials[old_species] - \
            self.chemical_potentials[species]
        potential_diff += chemical_potential_diff

        if self._acceptance_condition(potential_diff):
            self.accepted_trials += 1
            self.update_occupations([index], [species])

    def _acceptance_condition(self, potential_diff: float) -> bool:
        """
        Evaluates Metropolis acceptance criterion.

        Parameters
        ----------
        potential_diff
            change in the thermodynamic potential associated
            with the trial step
        """
        if potential_diff < 0:
            return True
        else:
            return np.exp(-potential_diff/(
                self.boltzmann_constant * self.temperature)) > \
                self._next_random_number()

    @property
    def chemical_potentials(self) -> Dict[int, float]:
        """ chemical potentials :math:`\\mu_i` """
        return self._chemical_potentials

    @chemical_potentials.setter
    def chemical_potentials(self,
                            chemical_potentials: Dict[Union[int, str], float]):
        if not isinstance(chemical_potentials, dict):
            raise TypeError('chemical_potentials has the wrong type')

        cps = OrderedDict([(key, val) if isinstance(key, int)
                           else (atomic_numbers[key], val)
                           for key, val in chemical_potentials.items()])

        if self._chemical_potentials is None:
            # TODO: add check with respect to configuration_manager
            self._chemical_potentials = cps
        else:
            for num in cps:
                if num not in self._chemical_potentials:
                    raise ValueError(
                        'Unknown species {} in chemical_potentials'.
                        format(num))
            self._chemical_potentials.update(cps)

    def _get_ensemble_data(self) -> Dict:
        """Returns the data associated with the ensemble. For the SGC
        ensemble this specifically includes the temperature and the
        species counts.
        """
        # generic data
        data = super()._get_ensemble_data()

        # temperature
        data['temperature'] = self.temperature

        # chemical potentials
        for atnum, chempot in self.chemical_potentials.items():
            data['mu_{}'.format(chemical_symbols[atnum])] = chempot

        # species counts
        atoms = self.configuration.atoms
        unique, counts = np.unique(atoms.numbers, return_counts=True)
        # TODO: avoid accessing a protected member of a client class
        for atnum in self.configuration._allowed_species:
            data['{}_count'.format(chemical_symbols[atnum])] = 0
        for atnum, count in zip(unique, counts):
            data['{}_count'.format(chemical_symbols[atnum])] = count

        return data
