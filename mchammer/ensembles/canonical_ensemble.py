"""Definition of the canonical ensemble class."""

from typing import Dict
import numpy as np

from ase import Atoms
from ase.units import kB

from .. import DataContainer
from .base_ensemble import BaseEnsemble
from ..calculators.base_calculator import BaseCalculator


class CanonicalEnsemble(BaseEnsemble):
    """Canonical Ensemble.

    Instances of this class allow one to simulate systems in the
    canonical ensemble (:math:`N_iVT`), i.e. at constant temperature
    (:math:`T`), number of atoms of each species (:math:`N_i`), and volume
    (:math:`V`).

    The probability density of the canonical ensemble is the well-known Boltzmann factor,

    .. math::

        \\rho_{\\text{C}} = \exp [ - E / k_B T ].

    Since the concentrations or, equivalently, the number of atoms of each
    species, is held fixed in the canonical ensemble, a trial step must
    conserve the concentrations. This is accomplished by randomly picking two
    unlike atoms and swapping their identities. The swap is accepted with
    probability

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
                 name: str='Canonical ensemble',
                 data_container: DataContainer=None, random_seed: int=None,
                 data_container_write_period: float=np.inf,
                 ensemble_data_write_interval: int=None,
                 trajectory_write_interval: int=None,
                 boltzmann_constant: float=kB, *, temperature: float):

        super().__init__(
            atoms=atoms, calculator=calculator, name=name,
            data_container=data_container,
            random_seed=random_seed,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

        self.temperature = temperature
        self.boltzmann_constant = boltzmann_constant

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        self.total_trials += 1

        sublattice_index = self.get_random_sublattice_index()
        sites, species = \
            self.configuration.get_swapped_state(sublattice_index)

        potential_diff = self._get_property_change(sites, species)

        if self._acceptance_condition(potential_diff):
            self.accepted_trials += 1
            self.update_occupations(sites, species)

    def _acceptance_condition(self, potential_diff: float) -> bool:
        """Evaluates Metropolis acceptance criterion.

        Parameters
        ----------
        potential_diff
            change in the thermodynamic potential associated
            with the trial step
        """
        if potential_diff < 0:
            return True
        else:
            return np.exp(-potential_diff / (
                self.boltzmann_constant * self.temperature)) > \
                self._next_random_number()

    def _get_ensemble_data(self) -> Dict:
        """Returns the data associated with the ensemble. For the SGC
        ensemble this specifically includes the temperature and the
        species counts.
        """
        data = super()._get_ensemble_data()
        data['temperature'] = self.temperature
        return data
