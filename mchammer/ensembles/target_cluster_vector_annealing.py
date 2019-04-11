from ase import Atoms
from mchammer.calculators.target_vector_calculator import TargetVectorCalculator
from .canonical_ensemble import CanonicalEnsemble
from .canonical_annealing import _cooling_exponential
import numpy as np
from typing import List
import random
from icet.io.logging import logger

logger = logger.getChild('target_cluster_vector_annealing')


class TargetClusterVectorAnnealing():
    """
    Instances of this class allow one to carry out simulated annealing
    towards a target cluster vector. Because it is impossible
    to know *a priori* which supercell shape accomodates the best
    match, this ensemble allows the annealing to be done for multiple
    :class:`ase.Atoms` objects at the same time.

    Parameters
    ----------
    atoms
        atomic configurations to be used in the Monte Carlo simulation;
        also defines the initial occupation vectors
    calculators
        calculators corresponding to each :class:`Atoms <ase.Atoms>`
        object
    T_start
        artificial temperature at which annealing is started
    T_stop : float
        artificial temperature at which annealing is stopped
    random_seed : int
        seed for random number generator used in the Monte Carlo
        simulation
    """

    def __init__(self, atoms: List[Atoms],
                 calculators: List[TargetVectorCalculator],
                 T_start: float = 5.0, T_stop: float = 0.001,
                 random_seed: int = None) -> None:

        if type(atoms) == Atoms:
            raise ValueError(
                'A list of ASE Atoms (supercells) must be provided')
        if len(atoms) != len(calculators):
            raise ValueError('There must be as many supercells as there '
                             'are calculators ({} != {})'.format(len(atoms),
                                                                 len(calculators)))

        logger.info('Initializing target cluster vector annealing '
                    'with {} supercells'.format(len(atoms)))

        # random number generator
        if random_seed is None:
            self._random_seed = random.randint(0, 1e16)
        else:
            self._random_seed = random_seed
        random.seed(a=self._random_seed)

        # Initialize an ensemble for each supercell
        sub_ensembles = []
        for ens_id, (supercell, calculator) in enumerate(zip(atoms, calculators)):
            sub_ensembles.append(CanonicalEnsemble(atoms=supercell,
                                                   calculator=calculator,
                                                   random_seed=random.randint(
                                                       0, 1e16),
                                                   user_tag='ensemble_{}'.format(
                                                       ens_id),
                                                   temperature=T_start,
                                                   data_container=None))
        self._sub_ensembles = sub_ensembles
        self._current_score = self._sub_ensembles[0].calculator.calculate_total(
            self._sub_ensembles[0].configuration.occupations)
        self._best_score = self._current_score
        self._best_atoms = atoms[0]
        self._temperature = T_start
        self._T_start = T_start
        self._T_stop = T_stop
        self._total_trials = 0
        self._accepted_trials = 0
        self._n_steps = 42

    def generate_structure(self, number_of_trial_steps: int = None) -> Atoms:
        """
        Run a structure annealing simulation.

        Parameters
        ----------
        number_of_trial_steps
            Total number of trial steps to perform. If None,
            run (on average) 3000 steps per supercell
        """
        if number_of_trial_steps is None:
            self._n_steps = 3000 * len(self._sub_ensembles)
        else:
            self._n_steps = number_of_trial_steps

        self._temperature = self._T_start
        self._total_trials = 0
        self._accepted_trials = 0
        while self.total_trials < self.n_steps:
            if self._total_trials % 1000 == 0:
                logger.info('MC step {}/{} ({} accepted trials, '
                            'temperature {:.3f}), '
                            'best score: {:.3f}'.format(self.total_trials,
                                                        self.n_steps,
                                                        self.accepted_trials,
                                                        self.temperature,
                                                        self.best_score))
            self._do_trial_step()
        return self.best_atoms

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        self._temperature = _cooling_exponential(
            self.total_trials, self.T_start, self.T_stop, self.n_steps)
        self._total_trials += 1

        # Choose a supercell
        ensemble = random.choice(self._sub_ensembles)

        # Choose two sites and swap
        sublattice_index = ensemble.get_random_sublattice_index()
        sites, species = ensemble.configuration.get_swapped_state(
            sublattice_index)

        # Update occupations so that the cluster vector (and its score)
        # can be calculated
        ensemble.configuration.update_occupations(sites, species)
        new_score = ensemble.calculator.calculate_total(
            ensemble.configuration.occupations)

        if self._acceptance_condition(new_score - self.current_score):
            self._current_score = new_score
            self._accepted_trials += 1

            # Since we are looking for the best structures we want to
            # keep track of the best one we have found as yet (the
            # current one may have a worse score)
            if self._current_score < self._best_score:
                self._best_atoms = ensemble.atoms
                self._best_score = self._current_score
        else:
            ensemble.configuration.update_occupations(
                sites, list(reversed(species)))

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
            p = np.exp(-potential_diff / self.temperature)
            return p > random.random()

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
        """ Stop temperature """
        return self._T_stop

    @property
    def n_steps(self) -> int:
        """ Number of steps to carry out """
        return self._n_steps

    @property
    def total_trials(self) -> int:
        """ Number of steps carried out so far """
        return self._total_trials

    @property
    def accepted_trials(self) -> int:
        """ Number of accepted trials carried out so far """
        return self._accepted_trials

    @property
    def current_score(self) -> float:
        """ Current target vector score """
        return self._current_score

    @property
    def best_score(self) -> float:
        """ Best target vector score found so far """
        return self._best_score

    @property
    def best_atoms(self) -> float:
        """ Structure most closely matching target vector so far """
        return self._best_atoms
