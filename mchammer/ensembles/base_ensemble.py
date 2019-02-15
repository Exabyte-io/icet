import os
import random
from abc import ABC, abstractmethod
from math import gcd
from time import time
from typing import Dict, List, Union, BinaryIO, TextIO

import numpy as np

from ase import Atoms
from ase.data import chemical_symbols
from collections import OrderedDict

from ..calculators.base_calculator import BaseCalculator
from ..configuration_manager import ConfigurationManager
from ..data_container import DataContainer
from ..observers.base_observer import BaseObserver


class BaseEnsemble(ABC):
    """Base ensemble class.

    Parameters
    ----------
    calculator : :class:`BaseCalculator`
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
    atoms : :class:`ase:Atoms`
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    user_tag : str
        human-readable tag for ensemble [default: None]
    data_container : str
        name of file the data container associated with the ensemble
        will be written to; if the file exists it will be read, the
        data container will be appended, and the file will be
        updated/overwritten
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
    random_seed : int
        seed for the random number generator used in the Monte Carlo
        simulation
    """

    def __init__(self, atoms: Atoms, calculator: BaseCalculator,
                 user_tag: str = None, data_container: DataContainer = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None,
                 random_seed: int = None) -> None:

        # initialize basic variables
        self._accepted_trials = 0
        self._total_trials = 0
        self._observers = {}
        self._step = 0

        # calculator and configuration
        self._calculator = calculator
        self._user_tag = user_tag
        strict_constraints_symbol = self.calculator.occupation_constraints
        symbols = list({tuple(sym)
                        for sym in strict_constraints_symbol if len(sym) > 1})
        sublattices = [[] for _ in symbols]
        for i, constraint in enumerate(strict_constraints_symbol):
            for j, sym in enumerate(symbols):
                if len(sym) < 2:
                    continue
                if sorted(constraint) == sorted(sym):
                    sublattices[j].append(i)
        self._sublattices = sublattices
        strict_constraints = []
        for symbols in strict_constraints_symbol:
            numbers = []
            for symbol in symbols:
                numbers.append(chemical_symbols.index(symbol))
            strict_constraints.append(numbers)
        self.configuration = ConfigurationManager(
            atoms, strict_constraints, sublattices)

        # random number generator
        if random_seed is None:
            self._random_seed = random.randint(0, 1e16)
        else:
            self._random_seed = random_seed
        random.seed(a=self._random_seed)

        # add ensemble parameters and metadata
        self._ensemble_parameters['n_atoms'] = len(self.atoms)
        metadata = OrderedDict(ensemble_name=self.__class__.__name__,
                               user_tag=user_tag,
                               seed=self.random_seed)

        # data container
        self._data_container_write_period = data_container_write_period

        self._data_container_filename = data_container

        if data_container is not None and os.path.isfile(data_container):
            self._data_container = DataContainer.read(data_container)

            dc_ensemble_parameters = self.data_container.ensemble_parameters
            if not dicts_equal(self.ensemble_parameters,
                               dc_ensemble_parameters):
                raise ValueError('Ensemble parameters do not match with those'
                                 ' stored in DataContainer file: {}'.format(
                                     set(dc_ensemble_parameters.items()) -
                                     set(self.ensemble_parameters.items())))
            self._restart_ensemble()
        else:
            if data_container is not None:
                # check if path to file exists
                filedir = os.path.dirname(data_container)
                if filedir and not os.path.isdir(filedir):
                    raise FileNotFoundError('Path to data container file does'
                                            ' not exist: {}'.format(filedir))
            self._data_container = \
                DataContainer(atoms=atoms,
                              ensemble_parameters=self.ensemble_parameters,
                              metadata=metadata)

        # interval for writing data and further preparation of data container
        default_interval = max(1, 10 * round(len(atoms) / 10))

        if ensemble_data_write_interval is None:
            self._ensemble_data_write_interval = default_interval
        else:
            self._ensemble_data_write_interval = ensemble_data_write_interval

        # Handle trajectory writing
        if trajectory_write_interval is None:
            self._trajectory_write_interval = default_interval
        else:
            self._trajectory_write_interval = trajectory_write_interval

        self._find_observer_interval()

    @property
    def atoms(self) -> Atoms:
        """ current configuration (copy) """
        return self.configuration.atoms.copy()

    @property
    def total_trials(self) -> int:
        """ number of Monte Carlo trial steps """
        return self._total_trials

    @property
    def accepted_trials(self) -> int:
        """ number of accepted trial steps """
        return self._accepted_trials

    @property
    def data_container(self) -> DataContainer:
        """ data container associated with ensemble """
        return self._data_container

    @property
    def observers(self) -> Dict[str, BaseObserver]:
        """ observers """
        return self._observers

    @property
    def acceptance_ratio(self) -> float:
        """ acceptance ratio """
        if self.total_trials > 0:
            return self.accepted_trials / self.total_trials
        return 0

    @property
    def calculator(self) -> BaseCalculator:
        """ calculator attached to the ensemble """
        return self._calculator

    @property
    def data_container_write_period(self) -> float:
        " data container write period "
        return self._data_container_write_period

    @data_container_write_period.setter
    def data_container_write_period(self, data_container_write_period):
        self._data_container_write_period = data_container_write_period

    def run(self, number_of_trial_steps: int, reset_step: bool = False):
        """
        Samples the ensemble for the given number of trial steps.

        Parameters
        ----------
        number_of_trial_steps
            number of MC trial steps to run in total
        reset_step
            if True the MC trial step counter and the data container will
            be reset to zero and empty, respectively.
        """

        last_write_time = time()
        if reset_step:
            initial_step = 0
            final_step = number_of_trial_steps
            self.reset_data_container()
        else:
            initial_step = self._step
            final_step = self._step + number_of_trial_steps
            # run Monte Carlo simulation such that we start at an
            # interval which lands on the observer interval
            if not initial_step == 0:
                first_run_interval = self.observer_interval -\
                    (initial_step -
                     (initial_step // self.observer_interval) *
                        self.observer_interval)
                first_run_interval = min(
                    first_run_interval, number_of_trial_steps)
                self._run(first_run_interval)
                initial_step += first_run_interval
                self._step += first_run_interval

        step = initial_step
        while step < final_step:
            uninterrupted_steps = min(
                self.observer_interval, final_step - step)
            if self._step % self.observer_interval == 0:
                self._observe(self._step)
            if self._data_container_filename is not None and \
                    time() - last_write_time > \
                    self.data_container_write_period:
                self.write_data_container(self._data_container_filename)
                last_write_time = time()

            self._run(uninterrupted_steps)
            step += uninterrupted_steps
            self._step += uninterrupted_steps

        # If we end on an observation interval we also observe
        if self._step % self.observer_interval == 0:
            self._observe(self._step)

        if self._data_container_filename is not None:
            self.write_data_container(self._data_container_filename)

    def _run(self, number_of_trial_steps: int):
        """Runs MC simulation for a number of trial steps without
        interruption.

        Parameters
        ----------
        number_of_trial_steps
            number of trial steps to run without stopping
        """
        for _ in range(number_of_trial_steps):
            self._do_trial_step()

    def _observe(self, step: int):
        """Submits current configuration to observers and appends
        observations to data container.

        Parameters
        ----------
        step
            the current trial step
        """
        row_dict = {}

        # Ensemble specific data
        if step % self._ensemble_data_write_interval == 0:
            ensemble_data = self._get_ensemble_data()
            for key, value in ensemble_data.items():
                row_dict[key] = value

        # Trajectory data
        if step % self._trajectory_write_interval == 0:
            row_dict['occupations'] = self.configuration.occupations.tolist()

        # Observer data
        for observer in self.observers.values():
            if step % observer.interval == 0:
                if observer.return_type is dict:
                    for key, value in observer.get_observable(
                            self.calculator.atoms).items():
                        row_dict[key] = value
                else:
                    row_dict[observer.tag] = observer.get_observable(
                        self.calculator.atoms)

        if len(row_dict) > 0:
            self._data_container.append(mctrial=step, record=row_dict)

    @abstractmethod
    def _do_trial_step(self):
        pass

    @property
    def user_tag(self) -> str:
        """ tag used for labeling the ensemble """
        return self._user_tag

    @property
    def random_seed(self) -> int:
        """ seed used to initialize random number generator """
        return self._random_seed

    def _next_random_number(self) -> float:
        """ Returns the next random number from the PRNG. """
        return random.random()

    @property
    def observer_interval(self) -> int:
        """minimum number of steps to run Monte Carlo simulation without
        interruption for observation
        """
        return self._observer_interval

    def _find_observer_interval(self) -> int:
        """
        Finds the greatest common denominator from the observation intervals.
        """
        intervals = [obs.interval for obs in self.observers.values()]

        if self._ensemble_data_write_interval is not np.inf:
            intervals.append(self._ensemble_data_write_interval)
        if self._trajectory_write_interval is not np.inf:
            intervals.append(self._trajectory_write_interval)
        if intervals:
            self._observer_interval = self._get_gcd(intervals)

    def _get_gcd(self, values: List[int]) -> int:
        """
        Finds the greatest common denominator (GCD) from a list.
        """
        if len(values) == 1:
            return values[0]

        if len(values) > 2:
            gcd_right = gcd(values[-1], values[-2])
            values.pop()
            values.pop()
            values.append(gcd_right)
            return self._get_gcd(values)
        else:
            return gcd(values[0], values[1])

    def attach_observer(self, observer: BaseObserver, tag=None):
        """
        Attaches an observer to the ensemble.

        Parameters
        ----------
        observer
            observer instance to attach
        tag
            name used in data container
        """
        if not isinstance(observer, BaseObserver):
            raise TypeError('observer has the wrong type: {}'
                            .format(type(observer)))

        if tag is not None:
            observer.tag = tag
            self.observers[tag] = observer
        else:
            self.observers[observer.tag] = observer

        self._find_observer_interval()

    def reset_data_container(self):
        """ Resets the data container and the trial step counter. """
        self._step = 0
        self._total_trials = 0
        self._accepted_trials = 0

        self._data_container.reset()

    def update_occupations(self, sites: List[int], species: List[int]):
        """Updates the occupation vector of the configuration being sampled.
        This will change the state of the configuration in both the
        calculator and the configuration manager.

        Parameters
        ----------
        sites
            indices of sites of the configuration to change
        species
            new occupations (species) by atomic number

        Raises
        ------
        ValueError
            if input lists are not of the same length
        """

        if len(sites) != len(species):
            raise ValueError('sites and species must have the same length.')
        self.configuration.update_occupations(sites, species)

    def _get_property_change(self,
                             sites: List[int], species: List[int]) -> float:
        """Computes and returns the property change due to a change of the
        configuration.

        _N.B.:_ This method leaves to configuration itself unchanged.

        Parameters
        ----------
        sites
            indices of sites to change
        species
            new occupations (species) by atomic number
        """
        current_species = self.configuration.occupations[sites]
        current_property = self.calculator.calculate_local_contribution(
            local_indices=sites,
            occupations=self.configuration.occupations)

        self.update_occupations(sites=sites, species=species)
        new_property = self.calculator.calculate_local_contribution(
            local_indices=sites,
            occupations=self.configuration.occupations)
        property_change = new_property - current_property

        # Restore initial configuration
        self.update_occupations(sites, current_species)
        return property_change

    def _get_ensemble_data(self) -> dict:
        """ Returns the current calculator property. """
        return {
            'potential': self.calculator.calculate_total(
                occupations=self.configuration.occupations),
            'acceptance_ratio': self.acceptance_ratio}

    def get_random_sublattice_index(self) -> int:
        """Returns a random sublattice index based on the weights of the
        sublattice.

        Todo
        ----
        * fix this method
        * add unit test
        """
        total_active_sites = sum([len(sub) for sub in self._sublattices])
        probability_distribution = [
            len(sub) / total_active_sites for sub in self._sublattices]
        pick = np.random.choice(
            range(0, len(self._sublattices)), p=probability_distribution)
        return pick

    def _restart_ensemble(self):
        """Restarts ensemble using the last state saved in DataContainer file.
        """

        # Restart step
        self._step = self.data_container.last_state['last_step']

        # Update configuration
        occupations = self.data_container.last_state['occupations']
        sites = list(range(len(self.configuration.atoms)))
        self.update_occupations(sites, occupations)

        # Restart number of total and accepted trial steps
        self._total_trials = self._step
        self._accepted_trials = \
            self.data_container.last_state['accepted_trials']

        # Restart state of random number generator
        random.setstate(self.data_container.last_state['random_state'])

    def write_data_container(self, outfile: Union[str, BinaryIO, TextIO]):
        """Updates last state of the Monte Carlo simulation and
        writes DataContainer to file.

        Parameters
        ----------
        outfile
            file to which to write
        """
        self._data_container._update_last_state(
            last_step=self._step,
            occupations=self.configuration.occupations.tolist(),
            accepted_trials=self._accepted_trials,
            random_state=random.getstate())

        self.data_container._write(outfile)

    @property
    def ensemble_parameters(self) -> dict:
        """Returns parameters associated with the ensemble."""
        return self._ensemble_parameters.copy()


def dicts_equal(dict1: Dict, dict2: Dict, atol: float = 1e-12) -> bool:
    """ Check if two dicts are equal, if float or integers
    are in the dicts then atol is used for comparing them. """
    if len(dict1) != len(dict2):
        return False
    for key in dict1.keys():
        if key not in dict2:
            return False
        if isinstance(dict1[key], (int, float)) and \
           isinstance(dict2[key], (int, float)):
            if not np.isclose(dict1[key], dict2[key], rtol=0.0, atol=atol):
                return False
        else:
            if dict1[key] != dict2[key]:
                return False
    return True
