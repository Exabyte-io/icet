import os
import random
from abc import ABC, abstractmethod
from math import gcd
from time import time
from typing import List, Dict
import numpy as np

from ase import Atoms

from ..data_container import DataContainer
from ..calculators.base_calculator import BaseCalculator
from ..configuration_manager import ConfigurationManager
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
    name : str
        human-readable ensemble name [default: `BaseEnsemble`]
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

    Attributes
    ----------
    accepted_trials : int
        number of accepted trial steps
    total_trials : int
        number of total trial steps
    data_container_write_period : int
        period in units of seconds at which the data container is
        written to file
    """

    def __init__(self, calculator=None, atoms=None, name='BaseEnsemble',
                 data_container=None, data_container_write_period=np.inf,
                 ensemble_data_write_interval=None,
                 trajectory_write_interval=None,
                 random_seed=None):

        if calculator is None:
            raise TypeError('Missing required keyword argument: calculator')
        if atoms is None:
            raise TypeError('Missing required keyword argument: atoms')

        # initialize basic variables
        self.accepted_trials = 0
        self.total_trials = 0
        self._observers = {}
        self._step = 0

        # calculator and configuration
        self._calculator = calculator
        self._name = name
        strict_constraints = self.calculator.occupation_constraints
        sublattices = [[i for i in range(len(self.calculator.atoms))]]
        self.configuration = ConfigurationManager(
            atoms, strict_constraints, sublattices)

        # random number generator
        if random_seed is None:
            self._random_seed = random.randint(0, 1e16)
        else:
            self._random_seed = random_seed
        random.seed(a=self._random_seed)

        # data container
        self.data_container_write_period = data_container_write_period
        self._data_container_filename = data_container
        if data_container is not None and os.path.isfile(data_container):
            self._data_container = DataContainer.read(data_container)
            self._restart_ensemble()
        else:
            self._data_container = \
                DataContainer(atoms=atoms, ensemble_name=name,
                              random_seed=self._random_seed)

        # interval for writing data and further preparation of data container
        default_interval = max(1, 10*round(len(atoms)/10))

        if ensemble_data_write_interval is None:
            self._ensemble_data_write_interval = default_interval
        else:
            self._ensemble_data_write_interval = ensemble_data_write_interval

        # Handle trajectory writing
        if trajectory_write_interval is None:
            self._trajectory_write_interval = default_interval
        else:
            self._trajectory_write_interval = trajectory_write_interval
        self._data_container.add_observable('occupations')

        self._find_observer_interval()

    @property
    def atoms(self) -> Atoms:
        """ current configuration (copy) """
        return self.configuration.atoms.copy()

    @property
    def step(self) -> int:
        """ current Monte Carlo trial step """
        return self._step

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

    def run(self, number_of_trial_steps: int, reset_step: bool=False):
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
                    time()-last_write_time > self.data_container_write_period:
                self._write_data_container()
                last_write_time = time()

            self._run(uninterrupted_steps)
            step += uninterrupted_steps
            self._step += uninterrupted_steps

        # If we end on an observation interval we also observe
        if self._step % self.observer_interval == 0:
            self._observe(self._step)

        if self._data_container_filename is not None:
            self._write_data_container()

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
                if key not in self._data_container.observables:
                    self._data_container.add_observable(key)
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
    def name(self) -> str:
        """ ensemble name """
        return self._name

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

        if observer.return_type is dict:
            for key in observer.get_keys():
                self._data_container.add_observable(key)
        else:
            self._data_container.add_observable(observer.tag)

        self._find_observer_interval()

    def reset_data_container(self):
        """ Resets the data container and the trial step counter. """
        self._step = 0
        self.total_trials = 0
        self.accepted_trials = 0

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
        return 0

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
        self.total_trials = self._step
        self.accepted_trials = \
            self.data_container.last_state['accepted_trials']

        # Restart state of random number generator
        random.setstate(self.data_container.last_state['random_state'])

    def _write_data_container(self):
        """Updates last state of the Monte Carlo simulation and
        writes DataContainer to file."""

        self._data_container._update_last_state(
            last_step=self._step,
            occupations=self.configuration.occupations.tolist(),
            accepted_trials=self.accepted_trials,
            random_state=random.getstate())

        self.data_container.write(self._data_container_filename)
