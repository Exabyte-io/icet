import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from math import gcd
from time import time
from typing import BinaryIO, Dict, List, TextIO, Union

import numpy as np

from ase import Atoms
from icet.core.sublattices import Sublattices

from ..calculators.base_calculator import BaseCalculator
from ..configuration_manager import ConfigurationManager
from ..data_container import DataContainer
from ..observers.base_observer import BaseObserver


class BaseEnsemble(ABC):
    """Base ensemble class.

    Parameters
    ----------
    structure : :class:`Atoms <ase.Atoms>`
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    calculator : :class:`BaseCalculator <mchammer.calculators.ClusterExpansionCalculator>`
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
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
        the data.
    trajectory_write_interval : int
        interval at which the current occupation vector of the atomic
        configuration is written to the data container.
    random_seed : int
        seed for the random number generator used in the Monte Carlo
        simulation
    """

    def __init__(self,
                 structure: Atoms,
                 calculator: BaseCalculator,
                 user_tag: str = None,
                 data_container: DataContainer = None,
                 data_container_write_period: float = 600,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None,
                 random_seed: int = None) -> None:

        # initialize basic variables
        self._accepted_trials = 0
        self._observers = {}
        self._step = 0

        # calculator and configuration
        self._calculator = calculator
        self._user_tag = user_tag
        sublattices = self.calculator.sublattices

        sublattices.assert_occupation_is_allowed(structure.get_chemical_symbols())

        # item for sublist in l for item in sublist
        symbols_flat = [s for sl in sublattices.active_sublattices for s in sl.chemical_symbols]
        if len(symbols_flat) != len(set(symbols_flat)):
            bad_symbols = set([s for s in symbols_flat if symbols_flat.count(s) > 1])
            raise ValueError('Symbols {} found on multiple active sublattices'.format(bad_symbols))

        self.configuration = ConfigurationManager(structure, sublattices)

        # random number generator
        if random_seed is None:
            self._random_seed = random.randint(0, 1e16)
        else:
            self._random_seed = random_seed
        random.seed(a=self._random_seed)

        # add ensemble parameters and metadata
        self._ensemble_parameters['n_atoms'] = len(self.structure)
        metadata = OrderedDict(ensemble_name=self.__class__.__name__,
                               user_tag=user_tag, seed=self.random_seed)

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
            self._data_container = DataContainer(structure=structure,
                                                 ensemble_parameters=self.ensemble_parameters,
                                                 metadata=metadata)

        # interval for writing data and further preparation of data container
        self._default_interval = len(structure)

        if ensemble_data_write_interval is None:
            self._ensemble_data_write_interval = self._default_interval
        else:
            self._ensemble_data_write_interval = ensemble_data_write_interval

        # Handle trajectory writing
        if trajectory_write_interval is None:
            self._trajectory_write_interval = self._default_interval
        else:
            self._trajectory_write_interval = trajectory_write_interval

        self._find_observer_interval()

    @property
    def structure(self) -> Atoms:
        """ current configuration (copy) """
        return self.configuration.structure.copy()

    @property
    def data_container(self) -> DataContainer:
        """ data container associated with ensemble """
        return self._data_container

    @property
    def observers(self) -> Dict[str, BaseObserver]:
        """ observers """
        return self._observers

    @property
    def calculator(self) -> BaseCalculator:
        """ calculator attached to the ensemble """
        return self._calculator

    @property
    def step(self) -> int:
        """ current configuration (copy) """
        return self._step

    def run(self, number_of_trial_steps: int):
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

        initial_step = self.step
        final_step = self.step + number_of_trial_steps
        # run Monte Carlo simulation such that we start at an
        # interval which lands on the observer interval
        if not initial_step == 0:
            first_run_interval = self.observer_interval -\
                (initial_step -
                 (initial_step // self.observer_interval) *
                    self.observer_interval)
            first_run_interval = min(first_run_interval, number_of_trial_steps)
            self._run(first_run_interval)
            initial_step += first_run_interval

        step = initial_step
        while step < final_step:
            uninterrupted_steps = min(self.observer_interval, final_step - step)
            if self.step % self.observer_interval == 0:
                self._observe(self.step)
            if self._data_container_filename is not None and \
                    time() - last_write_time > self._data_container_write_period:
                self.write_data_container(self._data_container_filename)
                last_write_time = time()

            self._run(uninterrupted_steps)
            step += uninterrupted_steps

        # If we end on an observation interval we also observe
        if self.step % self.observer_interval == 0:
            self._observe(self.step)

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
            accepted = self._do_trial_step()
            self._step += 1
            self._accepted_trials += accepted

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

            # reset accepted trial count
            self._accepted_trials = 0

        # Trajectory data
        if step % self._trajectory_write_interval == 0:
            row_dict['occupations'] = self.configuration.occupations.tolist()

        # Observer data
        for observer in self.observers.values():
            if step % observer.interval == 0:
                if observer.return_type is dict:
                    for key, value in observer.get_observable(self.calculator.structure).items():
                        row_dict[key] = value
                else:
                    row_dict[observer.tag] = observer.get_observable(self.calculator.structure)

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
            raise TypeError('observer has the wrong type: {}'.format(type(observer)))

        if observer.interval is None:
            observer.interval = self._default_interval

        if tag is not None:
            observer.tag = tag
            self.observers[tag] = observer
        else:
            self.observers[observer.tag] = observer

        self._find_observer_interval()

    def update_occupations(self, sites: List[int], species: List[int]):
        """Updates the occupation vector of the configuration being
        sampled. This will change the state of the configuration in
        both the calculator and the configuration manager.

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

    def _get_property_change(self, sites: List[int], species: List[int]) -> float:
        """Computes and returns the property change due to a change of
        the configuration.

        _N.B.:_ This method leaves the configuration itself unchanged.

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
        potential = self.calculator.calculate_total(occupations=self.configuration.occupations)
        return {'potential': potential,
                'acceptance_ratio': self._accepted_trials / self._ensemble_data_write_interval}

    def get_random_sublattice_index(self, probability_distribution) -> int:
        """Returns a random sublattice index based on the weights of the
        sublattice.

        Parameters
        ----------
        probability_distribution
            probability distributions for the sublattices
        """

        if len(probability_distribution) != len(self.sublattices):
            raise ValueError('probability_distribution should have the same size as sublattices')
        pick = np.random.choice(len(self.sublattices), p=probability_distribution)
        return pick

    def _restart_ensemble(self):
        """Restarts ensemble using the last state saved in DataContainer file.
        """

        # Restart step
        self._step = self.data_container.last_state['last_step']

        # Update configuration
        occupations = self.data_container.last_state['occupations']
        active_sites = []
        for sl in self.sublattices.active_sublattices:
            active_sites.extend(sl.indices)
        active_occupations = [occupations[s] for s in active_sites]
        self.update_occupations(active_sites, active_occupations)

        # Restart number of total and accepted trial steps
        self._accepted_trials = self.data_container.last_state['accepted_trials']

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
            last_step=self.step,
            occupations=self.configuration.occupations.tolist(),
            accepted_trials=self._accepted_trials,
            random_state=random.getstate())

        self.data_container.write(outfile)

    @property
    def ensemble_parameters(self) -> dict:
        """Returns parameters associated with the ensemble."""
        return self._ensemble_parameters.copy()

    @property
    def sublattices(self) -> Sublattices:
        """sublattices for the configuration being sampled"""
        return self.configuration.sublattices


def dicts_equal(dict1: Dict, dict2: Dict, atol: float = 1e-12) -> bool:
    """Returns True (False) if two dicts are equal (not equal), if
    float or integers are in the dicts then atol is used for comparing them."""
    if len(dict1) != len(dict2):
        return False
    for key in dict1.keys():
        if key not in dict2:
            return False
        if isinstance(dict1[key], (int, float)) and isinstance(dict2[key], (int, float)):
            if not np.isclose(dict1[key], dict2[key], rtol=0.0, atol=atol):
                return False
        else:
            if dict1[key] != dict2[key]:
                return False
    return True
