from abc import ABC, abstractmethod
from mchammer.data_container import DataContainer
from mchammer.observers.base_observer import BaseObserver
import random
from math import gcd


class BaseEnsemble(ABC):
    """
    Base ensemble abstract class.

    Attributes
    ----------
    name : str
        name of the ensemble.
    accepted_trials : int
        number of accepted trial steps.
    total_trials : int
        number of total trial steps.
    """

    def __init__(self, calculator, name='BaseEnsemble',
                 data_container=None, random_seed=None):

        self._calculator = calculator
        self._name = name

        self.accepted_trials = 0
        self.total_trials = 0
        self._observers = {}

        if random_seed is None:
            self._random_seed = random.randint(0, 1e16)
        else:
            self._random_seed = random_seed
        random.seed(a=self.random_seed)

        if data_container is None:
            self._data_container = DataContainer(atoms=None,
                                                 ensemble_name=name,
                                                 random_seed=random_seed)
        else:
            raise NotImplementedError

    @property
    def structure(self):
        """
        ASE Atoms object :
        The current state of the  structure being sampled in the ensemble.
        """
        return self.calculator.atoms.copy()

    @property
    def data_container(self):
        """
        mchammer DataContainer.
        """
        return self._data_container

    @property
    def observers(self):
        """
        dict of mchammer observers.
        """
        return self._observers

    @property
    def acceptance_ratio(self):
        """
        float : the acceptance ratio,
        i.e. accepted_trials / total_trials.
        """
        return self.accepted_trials / self.total_trials

    @property
    def calculator(self):
        """
        mchammer Calculator.
        """
        return self._calculator

    def run(self, number_of_MC_steps):
        """
        Sample the ensemble for `number_of_MC_steps` steps.

        Parameters:
        -----------

        number_of_MC_steps: int
            number of steps to run in total
        """
        pass

    @abstractmethod
    def do_trial_move(self):
        pass

    @property
    def name(self):
        """
        str : Name of BaseEnsemble.
        """
        return self._name

    @property
    def random_seed(self):
        """
        int : Random seed used in random number generator.
        """
        return self._random_seed

    def next_random_number(self):
        """
        Return the next random number from the RNG.
        """
        return random.random()

    @property
    def minimum_observation_interval(self):
        """
        int : minimum number of steps to run
        Monte Carlo simulation without observation interruptions.
        """
        return self._minimum_observation_interval

    @abstractmethod
    def _run(self, number_of_trial_moves):
        """
        Private method for running the MCMC simulation
        without interruption.

        number_of_trial_moves : int
           number of trial moves to run without stopping.
        """
        pass

    def _find_minimum_observation_interval(self):
        """
        Find the greatest common denominator from the observation intervals.
        """

        intervals = [obs.interval for obs in self.observers]
        self._minimum_observation_interval = self._get_gcd(intervals)

    def _get_gcd(self, interval_list):
        """
        Find the gcd from the list of intervals.

        interval_list : list of int
        """

        if len(interval_list) > 2:
            gcd_right = gcd(interval_list[-1], interval_list[-2])
            interval_list.pop()
            interval_list.pop()
            interval_list.append(gcd_right)
            return self._get_gcd(interval_list)
        else:
            return gcd(interval_list[0], interval_list[1])

    def attach_observer(self, observer, tag=None):
        """
        Attaches an observer to the ensemble.

        Parameters:
        ----------
        observer : mchammer Observer object
        """
        assert isinstance(observer, BaseObserver), \
            'observer argument invalid; must be child of BaseObserver'

        if tag is not None:
            self.observers[tag] = observer
        else:
            self.observers[observer.tag] = observer

        self._find_minimum_observation_interval()
