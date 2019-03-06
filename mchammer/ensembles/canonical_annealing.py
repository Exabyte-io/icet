"""Definition of the canonical annealing class."""

import numpy as np

from ase import Atoms
from ase.units import kB
from typing import Dict

from .. import DataContainer
from .base_ensemble import BaseEnsemble
from ..calculators.base_calculator import BaseCalculator


class CanonicalAnnealing(BaseEnsemble):

    def __init__(self, atoms: Atoms, calculator: BaseCalculator,
                 T_start: float, T_stop: float, n_steps: int,
                 function_tag: str = 'linear',
                 user_tag: str = None,
                 boltzmann_constant: float = kB,
                 data_container: DataContainer = None, random_seed: int = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None) -> None:

        self._ensemble_parameters = dict(n_steps=n_steps, function_tag=function_tag)

        super().__init__(
            atoms=atoms, calculator=calculator, user_tag=user_tag,
            data_container=data_container,
            random_seed=random_seed,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

        self._boltzmann_constant = boltzmann_constant
        self._temperature = T_start
        self._T_start = T_start
        self._T_stop = T_stop
        self._n_steps = n_steps
        self._function_tag = function_tag
        self._function = available_functions[function_tag]

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def T_start(self) -> float:
        return self._T_start

    @property
    def T_stop(self) -> float:
        return self._T_stop

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def function_tag(self) -> str:
        return self._function_tag

    @property
    def boltzmann_constant(self) -> float:
        """ Boltzmann constant :math:`k_B` (see parameters section above) """
        return self._boltzmann_constant

    def run(self):
        """ Runs the ensemble """
        super().run(self.n_steps)

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        self._temperature = self._function(self.total_trials, self.T_start, self.T_stop, self.n_steps)
        self._total_trials += 1

        sublattice_index = self.get_random_sublattice_index()
        sites, species = self.configuration.get_swapped_state(sublattice_index)
        potential_diff = self._get_property_change(sites, species)

        if self._acceptance_condition(potential_diff):
            self._accepted_trials += 1
            self.update_occupations(sites, species)

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
        elif abs(self.temperature) < 1e-6:  # temperature is numerically zero
            return False
        else:
            p = np.exp(-potential_diff / (self.boltzmann_constant * self.temperature))
            return p > self._next_random_number()

    def _get_ensemble_data(self) -> Dict:
        """Returns the data associated with the ensemble. For the CanonicalAnnealing
        this specifically includes the temperature..
        """
        # generic data
        data = super()._get_ensemble_data()
        data['temperature'] = self.temperature
        return data


def function_linear(step, T_start, T_stop, n_steps):
    return T_start + (T_stop-T_start) * step / (n_steps - 1)


def function_quadratic(step, T_start, T_stop, n_steps):
    t = (step - n_steps + 1) / (n_steps - 1)
    scale = t**2
    return T_start * scale + T_stop


def function_cos(step, T_start, T_stop, n_steps):
    scale = (1 + np.cos(np.pi * step/(n_steps-1))) / 2
    return T_start * scale + T_stop


available_functions = dict(linear=function_linear,
                           quadratic=function_quadratic,
                           cos=function_cos)
