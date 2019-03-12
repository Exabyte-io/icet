"""Definition of the canonical annealing class."""

import numpy as np

from ase import Atoms
from ase.units import kB
from typing import Dict

from .. import DataContainer
from .base_ensemble import BaseEnsemble
from ..calculators.base_calculator import BaseCalculator


class CanonicalAnnealing(BaseEnsemble):
    """Instances of this class allow one to carry out simulated
    annealing in the canonical ensemble.
    See canonical ensemble for more information about this ensemble.

    This ensemble can be very useful when for example trying to find
    ground-states or generating low energy configurations.

    The cooling scheme to be used can be chosen from the pre-defined
    cooling functions, or a user-defined function can be used which
    must then have the following interface

    def cooling_function(step, T_start, T_stop, n_steps):
        T = ...
        return T

    where step refers to current mctrial step.

    Parameters
    ----------
    atoms : :class:`ase:Atoms`
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    calculator : :class:`BaseCalculator`
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
    T_start : float
        temperature from which the annealing is started
    T_stop : float
        final temperature for annealing
    n_steps : int
        number of steps to take in the annealing simulation
    cooling_function : str/function
        to use the predefined cooling functions give a string
        `linear` or `exponential`, otherwise provide a function
    boltzmann_constant : float
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e. units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K]
    user_tag : str
        human-readable tag for ensemble [default: None]
    data_container : str
        name of file the data container associated with the ensemble
        will be written to; if the file exists it will be read, the
        data container will be appended, and the file will be
        updated/overwritten
    random_seed : int
        seed for the random number generator used in the Monte Carlo
        simulation
    ensemble_data_write_interval : int
        interval at which data is written to the data container; this
        includes for example the current value of the calculator
        (i.e. usually the energy) as well as ensembles specific fields
        such as temperature or the number of atoms of different species
    data_container_write_period : float
        period in units of seconds at which the data container is
        written to file; writing periodically to file provides both
        a way to examine the progress of the simulation and to back up
        the data [default: np.inf]
    trajectory_write_interval : int
        interval at which the current occupation vector of the atomic
        configuration is written to the data container.
    """

    def __init__(self, atoms: Atoms, calculator: BaseCalculator,
                 T_start: float, T_stop: float, n_steps: int,
                 cooling_function: str = 'exponential',
                 user_tag: str = None,
                 boltzmann_constant: float = kB,
                 data_container: DataContainer = None, random_seed: int = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None) -> None:

        self._ensemble_parameters = dict(n_steps=n_steps)

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

        if isinstance(cooling_function, str):
            available = sorted(available_cooling_functions.keys())
            if cooling_function not in available:
                raise ValueError('Select from the available cooling_functions {}'.format(available))
            self._cooling_function = available_cooling_functions[cooling_function]
        elif callable(cooling_function):
            self._cooling_function = cooling_function
        else:
            raise TypeError('cooling_function must be either str or a function')

    @property
    def temperature(self) -> float:
        """ Current temperature """
        return self._temperature

    @property
    def T_start(self) -> float:
        """ Starting temperature """
        return self._T_start

    @property
    def T_stop(self) -> float:
        """ Starting temperature """
        return self._T_stop

    @property
    def n_steps(self) -> int:
        """ Number of steps to carry out """
        return self._n_steps

    @property
    def boltzmann_constant(self) -> float:
        """ Boltzmann constant :math:`k_B` (see parameters section above) """
        return self._boltzmann_constant

    def run(self):
        """ Runs the annealing. """
        if self.total_trials >= self.n_steps:
            raise Exception('Annealing has already finished')
        super().run(self.n_steps - self.total_trials)

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        self._temperature = self._cooling_function(
            self.total_trials, self.T_start, self.T_stop, self.n_steps)
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
        """Returns the data associated with the ensemble. For the
        CanonicalAnnealing this specifically includes the temperature.
        """
        data = super()._get_ensemble_data()
        data['temperature'] = self.temperature
        return data


def _cooling_linear(step, T_start, T_stop, n_steps):
    return T_start + (T_stop-T_start) * step / (n_steps - 1)


def _cooling_exponential(step, T_start, T_stop, n_steps):
    return T_start - (T_start - T_stop) * np.log(step+1) / np.log(n_steps)


available_cooling_functions = dict(linear=_cooling_linear,
                                   exponential=_cooling_exponential)
