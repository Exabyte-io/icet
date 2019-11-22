"""Definition of the canonical annealing class."""

import numpy as np

from ase import Atoms
from ase.units import kB
from typing import Dict, List

from .. import DataContainer
from ..calculators.base_calculator import BaseCalculator
from .thermodynamic_base_ensemble import ThermodynamicBaseEnsemble


class CanonicalAnnealing(ThermodynamicBaseEnsemble):
    """Instances of this class allow one to carry out simulated annealing
    in the canonical ensemble, i.e. the temperature is varied in
    pre-defined fashion while the composition is kept fixed.  See
    :class:`mchammer.ensembles.CanonicalEnsemble` for more information
    about the standard canonical ensemble.

    The canonical annealing ensemble can be useful, for example, for
    finding ground states or generating low energy configurations.

    The temperature control scheme is selected via the
    ``cooling_function`` keyword argument, while the initial and final
    temperature are set via the ``T_start`` and ``T_stop`` arguments.
    Several pre-defined temperature control schemes are available
    including `linear` and `exponential`. In the latter case the
    temperature varies logarithmatically as a function of the MC step,
    emulating the exponential temperature dependence of the atomic
    exchange rate encountered in many materials.  It is also possible
    to provide a user defined cooling function via the keyword
    argument.  This function must comply with the following function
    header::

        def cooling_function(step, T_start, T_stop, n_steps):
            T = ...  # compute temperature
            return T

    Here ``step`` refers to the current MC trial step.

    Parameters
    ----------
    structure : :class:`Atoms <ase.Atoms>`
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    calculator : :class:`BaseCalculator <mchammer.calculators.ClusterExpansionCalculator>`
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
    T_start : float
        temperature from which the annealing is started
    T_stop : float
        final temperature for annealing
    n_steps : int
        number of steps to take in the annealing simulation
    cooling_function : str/function
        to use the predefined cooling functions provide a string
        `linear` or `exponential`, otherwise provide a function
    boltzmann_constant : float
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e. units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K]
    user_tag : str
        human-readable tag for ensemble [default: None]
    random_seed : int
        seed for the random number generator used in the Monte Carlo
        simulation
    data_container : str
        name of file the data container associated with the ensemble
        will be written to; if the file exists it will be read, the
        data container will be appended, and the file will be
        updated/overwritten
    data_container_write_period : float
        period in units of seconds at which the data container is
        written to file; writing periodically to file provides both
        a way to examine the progress of the simulation and to back up
        the data [default: np.inf]
    ensemble_data_write_interval : int
        interval at which data is written to the data container; this
        includes for example the current value of the calculator
        (i.e. usually the energy) as well as ensembles specific fields
        such as temperature or the number of atoms of different species
    trajectory_write_interval : int
        interval at which the current occupation vector of the atomic
        configuration is written to the data container.
    sublattice_probabilities : List[float]
        probability for picking a sublattice when doing a random swap.
        This should be as long as the number of sublattices and should
        sum up to 1.

    """

    def __init__(self,
                 structure: Atoms,
                 calculator: BaseCalculator,
                 T_start: float,
                 T_stop: float,
                 n_steps: int,
                 cooling_function: str = 'exponential',
                 user_tag: str = None,
                 boltzmann_constant: float = kB,
                 random_seed: int = None,
                 data_container: str = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None,
                 sublattice_probabilities: List[float] = None) -> None:

        self._ensemble_parameters = dict(n_steps=n_steps)

        # add species count to ensemble parameters
        for sl in calculator.sublattices:
            for symbol in sl.chemical_symbols:
                key = 'n_atoms_{}'.format(symbol)
                count = structure.get_chemical_symbols().count(symbol)
                self._ensemble_parameters[key] = count

        super().__init__(
            structure=structure, calculator=calculator, user_tag=user_tag,
            random_seed=random_seed,
            data_container=data_container,
            data_container_class=DataContainer,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval,
            boltzmann_constant=boltzmann_constant)

        self._temperature = T_start
        self._T_start = T_start
        self._T_stop = T_stop
        self._n_steps = n_steps

        self._ground_state_candidate = self.configuration.structure
        self._ground_state_candidate_potential = self.calculator.calculate_total(
            occupations=self.configuration.occupations)

        # setup cooling function
        if isinstance(cooling_function, str):
            available = sorted(available_cooling_functions.keys())
            if cooling_function not in available:
                raise ValueError(
                    'Select from the available cooling_functions {}'.format(available))
            self._cooling_function = available_cooling_functions[cooling_function]
        elif callable(cooling_function):
            self._cooling_function = cooling_function
        else:
            raise TypeError('cooling_function must be either str or a function')

        if sublattice_probabilities is None:
            self._swap_sublattice_probabilities = self._get_swap_sublattice_probabilities()
        else:
            self._swap_sublattice_probabilities = sublattice_probabilities

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
    def estimated_ground_state(self):
        """ Structure with lowest observed potential during run """
        return self._ground_state_candidate.copy()

    @property
    def estimated_ground_state_potential(self):
        """ Lowest observed potential during run """
        return self._ground_state_candidate_potential

    def run(self):
        """ Runs the annealing. """
        if self.step >= self.n_steps:
            raise Exception('Annealing has already finished')
        super().run(self.n_steps - self.step)

    def _do_trial_step(self) -> int:
        """ Carries out one Monte Carlo trial step. """
        self._temperature = self._cooling_function(
            self.step, self.T_start, self.T_stop, self.n_steps)
        sublattice_index = self.get_random_sublattice_index(self._swap_sublattice_probabilities)
        return self.do_canonical_swap(sublattice_index=sublattice_index)

    def _get_ensemble_data(self) -> Dict:
        """Returns the data associated with the ensemble. For the
        CanonicalAnnealing this specifically includes the temperature.
        """
        data = super()._get_ensemble_data()
        data['temperature'] = self.temperature
        if data['potential'] < self._ground_state_candidate_potential:
            self._ground_state_candidate_potential = data['potential']
            self._ground_state_candidate = self.configuration.structure
        return data


def _cooling_linear(step, T_start, T_stop, n_steps):
    return T_start + (T_stop-T_start) * step / (n_steps - 1)


def _cooling_exponential(step, T_start, T_stop, n_steps):
    return T_start - (T_start - T_stop) * np.log(step+1) / np.log(n_steps)


available_cooling_functions = dict(linear=_cooling_linear, exponential=_cooling_exponential)
