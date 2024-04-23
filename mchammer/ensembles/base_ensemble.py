"""Definition of the abstract base ensemble class."""

import os
import random
import warnings

from abc import ABC, abstractmethod
from collections import OrderedDict
from math import gcd
from time import time
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from ase import Atoms
from icet.core.sublattices import Sublattices

from ..calculators.base_calculator import BaseCalculator
from ..configuration_manager import ConfigurationManager
from ..data_containers.base_data_container import BaseDataContainer
from ..observers.base_observer import BaseObserver


class BaseEnsemble(ABC):
    """Base ensemble class.

    Parameters
    ----------
    structure
        Atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector.
    calculator
        Calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion.
    user_tag
        Human-readable tag for ensemble. Default: ``None``.
    random_seed
        Seed for the random number generator used in the Monte Carlo simulation.
    dc_filename
        Name of file the data container associated with the ensemble
        will be written to. If the file exists it will be read, the
        data container will be appended, and the file will be
        updated/overwritten.
    data_container_class
        Used to initialize custom (ensemble specific) data container objects.
        By default the class uses the generic :class:`BaseDataContainer` class.
    data_container_write_period
        Period in units of seconds at which the data container is
        written to file. Writing periodically to file provides both
        a way to examine the progress of the simulation and to back up
        the data.
    ensemble_data_write_interval
        Interval at which data is written to the data container. This
        includes for example the current value of the calculator
        (i.e., usually the energy) as well as ensembles specific fields
        such as temperature or the number of atoms of different species.
        Default: Number of sites in the :attr:`structure`.
    trajectory_write_interval
        Interval at which the current occupation vector of the atomic
        configuration is written to the data container.
        Default: Number of sites in the :attr:`structure`.
    """

    def __init__(self,
                 structure: Atoms,
                 calculator: BaseCalculator,
                 user_tag: str = None,
                 random_seed: int = None,
                 dc_filename: str = None,
                 data_container: str = None,
                 data_container_class: Type[BaseDataContainer] = BaseDataContainer,
                 data_container_write_period: float = 600,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None) -> None:

        # initialize basic variables
        self._accepted_trials = 0
        self._observers: Dict[str, BaseObserver] = {}
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
            self._random_seed = random.randint(0, int(1e16))
        else:
            self._random_seed = random_seed
        random.seed(a=self._random_seed)

        # add ensemble parameters and metadata
        if not self._ensemble_parameters:
            self._ensemble_parameters: Dict[str, Any] = {}
        self._ensemble_parameters['n_atoms'] = len(self.structure)
        metadata = OrderedDict(ensemble_name=self.__class__.__name__,
                               user_tag=user_tag, seed=self.random_seed)

        # data container
        self._data_container_write_period = data_container_write_period
        if data_container is not None:
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn('data_container is deprecated, use dc_filename', DeprecationWarning)
            self._data_container_filename = data_container
        else:
            self._data_container_filename = dc_filename

        if dc_filename is not None and os.path.isfile(dc_filename):
            self._data_container: BaseDataContainer = data_container_class.read(dc_filename)

            dc_ensemble_parameters = self.data_container.ensemble_parameters
            if not dicts_equal(self.ensemble_parameters,
                               dc_ensemble_parameters):
                raise ValueError('Ensemble parameters do not match those'
                                 ' stored in data container file: {}'.format(
                                     set(dc_ensemble_parameters.items()) -
                                     set(self.ensemble_parameters.items())))
            self._restart_ensemble()
        else:
            if dc_filename is not None:
                # check if path to file exists
                filedir = os.path.dirname(dc_filename)
                if filedir and not os.path.isdir(filedir):
                    raise FileNotFoundError('Path to data container file does'
                                            ' not exist: {}'.format(filedir))
            self._data_container = data_container_class(
                structure=structure,
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
        """ Current configuration (copy). """
        return self.configuration.structure

    @property
    def data_container(self) -> BaseDataContainer:
        """ Data container associated with ensemble. """
        return self._data_container

    @property
    def observers(self) -> Dict[str, BaseObserver]:
        """ Observers. """
        return self._observers

    @property
    def calculator(self) -> BaseCalculator:
        """ Calculator attached to the ensemble. """
        return self._calculator

    @property
    def step(self) -> int:
        """ Current trial step counter. """
        return self._step

    def run(self, number_of_trial_steps: int):
        """
        Samples the ensemble for the given number of trial steps.

        Parameters
        ----------
        number_of_trial_steps
            Number of MC trial steps to run in total.
        reset_step
            If ``True`` the MC trial step counter and the data container will
            be reset to zero and empty, respectively.

        Raises
        ------
        TypeError
            If :attr:`number_of_trial_steps` is not an ``int``.
        """

        if not isinstance(number_of_trial_steps, int):
            raise TypeError('number_of_trial_steps must be an integer ({})'
                            .format(number_of_trial_steps))

        last_write_time = time()

        initial_step = self.step
        final_step = self.step + number_of_trial_steps
        # run Monte Carlo simulation such that we start at an
        # interval which lands on the observer interval
        if initial_step != 0:
            first_run_interval = self.observer_interval -\
                (initial_step -
                 (initial_step // self.observer_interval) *
                 self.observer_interval)
            first_run_interval = min(first_run_interval, number_of_trial_steps)
            self._run(first_run_interval)
            initial_step += first_run_interval

        step = initial_step
        while step < final_step and not self._terminate_sampling():
            uninterrupted_steps = min(self.observer_interval, final_step - step)
            if self.step % self.observer_interval == 0:
                self._observe(self.step)
            if self._data_container_filename is not None and \
                    time() - last_write_time > self._data_container_write_period:
                self.write_data_container(self._data_container_filename)
                last_write_time = time()

            self._run(uninterrupted_steps)
            step += uninterrupted_steps

        # if we end on an observation interval we also observe
        if self.step % self.observer_interval == 0:
            self._observe(self.step)

        # allow ensemble a chance to go clean
        self._finalize()

        if self._data_container_filename is not None:
            self.write_data_container(self._data_container_filename)

    def _run(self, number_of_trial_steps: int):
        """Runs MC simulation for a number of trial steps without interruption.

        Parameters
        ----------
        number_of_trial_steps
            Number of trial steps to run without stopping.
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
            Current trial step.
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
            assert isinstance(observer.interval, int), 'interval is not an int'
            if step % observer.interval == 0:
                if observer.return_type is dict:
                    for key, val in observer.get_observable(self.configuration.structure).items():
                        row_dict[key] = val
                else:
                    row_dict[observer.tag] = observer.get_observable(self.configuration.structure)

        if len(row_dict) > 0:
            self._data_container.append(mctrial=step, record=row_dict)

    @abstractmethod
    def _do_trial_step(self):
        pass

    @property
    def user_tag(self) -> Optional[str]:
        """ Tag used for labeling the ensemble. """
        return self._user_tag

    @property
    def random_seed(self) -> int:
        """ Seed used to initialize random number generator. """
        return self._random_seed

    def _next_random_number(self) -> float:
        """ Returns the next random number from the PRNG. """
        return random.random()

    @property
    def observer_interval(self) -> int:
        """ Minimum number of steps to run Monte Carlo simulation without
        interruption for observation.
        """
        return self._observer_interval

    def _find_observer_interval(self) -> None:
        """
        Finds the greatest common denominator from the observation intervals.
        """
        intervals = [obs.interval for obs in self.observers.values()]

        if self._ensemble_data_write_interval is not np.inf:
            intervals.append(self._ensemble_data_write_interval)
        if self._trajectory_write_interval is not np.inf:
            intervals.append(self._trajectory_write_interval)
        if intervals:
            assert all([isinstance(k, int) for k in intervals]), 'intervals must be ints'
            self._observer_interval = self._get_gcd(intervals)

    def _get_gcd(self, values: List[int]) -> int:
        """ Finds the greatest common denominator (GCD) from a list of integers. """
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

    def attach_observer(self, observer: BaseObserver, tag: str = None):
        """
        Attaches an observer to the ensemble.

        If the observer does not have an observation interval,
        then it will be set to the default_interval.

        Parameters
        ----------
        observer
            Observer instance to attach.
        tag
            Name used in data container.
        """
        if not isinstance(observer, BaseObserver):
            raise TypeError('observer has the wrong type: {type(observer))}')

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
            Indices of sites of the configuration to change.
        species
            New occupations (species) by atomic number.

        Raises
        ------
        ValueError
            If input lists are not of the same length.
        """

        if len(sites) != len(species):
            raise ValueError('sites and species must have the same length.')
        self.configuration.update_occupations(sites, species)
        self.calculator.accept_change()

    def _get_property_change(self, sites: List[int], species: List[int]) -> float:
        """Computes and returns the property change due to a change of
        the configuration.

        Note
        ----
        This method leaves the configuration itself unchanged.

        Parameters
        ----------
        sites
            Indices of sites to change.
        species
            New occupations (species) by atomic number.
        """
        return self.calculator.calculate_change(sites=sites,
                                                current_occupations=self.configuration.occupations,
                                                new_site_occupations=species)

    def _get_ensemble_data(self) -> dict:
        """ Returns the current calculator property. """
        potential = self.calculator.calculate_total(occupations=self.configuration.occupations)
        return {'potential': potential,
                'acceptance_ratio': self._accepted_trials / self._ensemble_data_write_interval}

    def get_random_sublattice_index(self, probability_distribution) -> int:
        """ Returns a random sublattice index based on the weights of the
        sublattice.

        Parameters
        ----------
        probability_distribution
            Probability distributions for the sublattices.
        """
        n_sublattices = len(self.sublattices)
        if len(probability_distribution) != n_sublattices:
            raise ValueError('probability_distribution should have the same size as sublattices')
        if n_sublattices == 1:  # Special but very common case, we save some time
            return 0
        pick = random.choices(list(range(n_sublattices)),
                              weights=probability_distribution)[0]
        return pick

    def _restart_ensemble(self):
        """Restarts ensemble using the last state saved in data container file.
        """

        # Restart step
        self._step = self.data_container._last_state['last_step']

        # Update configuration
        occupations = self.data_container._last_state['occupations']
        active_sites = []
        for sl in self.sublattices.active_sublattices:
            active_sites.extend(sl.indices)
        active_occupations = [occupations[s] for s in active_sites]
        self.update_occupations(active_sites, active_occupations)

        # Restart number of total and accepted trial steps
        self._accepted_trials = self.data_container._last_state['accepted_trials']

        # Restart state of random number generator
        random.setstate(self.data_container._last_state['random_state'])

    def write_data_container(self, outfile: Union[str, bytes]):
        """Updates last state of the Monte Carlo simulation and
        writes data container to file.

        Parameters
        ----------
        outfile
            File to which to write.
        """
        self._data_container._update_last_state(
            last_step=self.step,
            occupations=self.configuration.occupations.tolist(),
            accepted_trials=self._accepted_trials,
            random_state=random.getstate())

        self.data_container.write(outfile)

    @property
    def ensemble_parameters(self) -> dict:
        """ Parameters associated with the ensemble. """
        return self._ensemble_parameters.copy()

    @property
    def sublattices(self) -> Sublattices:
        """ Sublattices for the configuration being sampled. """
        return self.configuration.sublattices

    def _terminate_sampling(self) -> bool:
        """This method is called from the run method to determine whether the MC
        sampling loop should be terminated for a reason other than having exhausted
        the number of iterations. The method can be overriden by child classes in
        order to provide an alternative exit mechanism.
        """
        return False

    def _finalize(self) -> None:
        """This method is called from the run method after the conclusion of
        the MC cycles but before the data container is written. This
        method can be used by child classes to carry out clean-up
        tasks, including e.g., adding "left-over" data to the data
        container.
        """
        pass

    def __str__(self) -> str:
        """ String representation. """
        width = 60
        name = self.__class__.__name__
        s = [' {} '.format(name).center(width, '=')]

        fmt = '{:15} : {}'
        for k, v in self.ensemble_parameters.items():
            s += [fmt.format(k, v)]

        s += [fmt.format('step', self.step)]
        s += [fmt.format('calculator', self._calculator.__class__.__name__)]
        return '\n'.join(s)


def dicts_equal(dict1: Dict, dict2: Dict, atol: float = 1e-12) -> bool:
    """Returns True (False) if two dicts are equal (not equal), if
    float or integers are in the dicts then atol is used for comparing them.
    """
    if len(dict1) != len(dict2):
        return False
    for key in dict1.keys():
        if key not in dict2:
            return False
        if isinstance(dict1[key], (int, float)) and isinstance(dict2[key], (int, float)):
            if not np.isclose(dict1[key], dict2[key], rtol=0.0, atol=atol) and \
                    not np.isnan(dict1[key]) and not np.isnan(dict2[key]):
                return False
        else:
            if dict1[key] != dict2[key]:
                return False
    return True
