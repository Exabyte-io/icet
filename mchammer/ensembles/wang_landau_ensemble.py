"""Definition of the Wang-Landau multi-canonical ensemble class."""

from collections import OrderedDict
from typing import List

import numpy as np

from ase import Atoms
from pandas import DataFrame

from .. import DataContainer
from ..calculators.base_calculator import BaseCalculator
from .thermodynamic_base_ensemble import BaseEnsemble


class WangLandauEnsemble(BaseEnsemble):
    """
    Todo
    ----
    * check restart functionality
    * write docstring starting from CanonicalEnsemble
    * add more optimal f updates (see https://doi.org/10.1063/1.2803061, https://arxiv.org/abs/cond-mat/0702414)
    * check Berg, Comput. Phys. Commun. 153, 397 (2003)
    * check also Berg-Neuhaus
    * check Wang–Landau multibondic cluster approach to simulations of second-order transitions (https://doi.org/10.1016/j.phpro.2010.09.040)
    * check https://arxiv.org/pdf/1303.1767.pdf
    """

    def __init__(self,
                 structure: Atoms,
                 calculator: BaseCalculator,
                 energy_spacing: float,
                 fill_factor_update_interval: int = 1e4,
                 fill_factor_limit: float = 1e-7,
                 flatness_threshold: float = 0.8,
                 last_visit_limit: int = 1e4,
                 user_tag: str = None,
                 data_container: DataContainer = None,
                 random_seed: int = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None,
                 sublattice_probabilities: List[float] = None) -> None:

        # ensemble parameters
        self._energy_spacing = energy_spacing
        self._fill_factor_update_interval = fill_factor_update_interval
        self._fill_factor_limit = fill_factor_limit
        self._flatness_threshold = flatness_threshold
        self._last_visit_limit = last_visit_limit
        self._ensemble_parameters = {}
        self._ensemble_parameters['energy_spacing'] = energy_spacing
        self._ensemble_parameters['fill_factor_update_interval'] = self._fill_factor_update_interval
        self._ensemble_parameters['fill_factor_limit'] = self._fill_factor_limit
        self._ensemble_parameters['flatness_threshold'] = self._flatness_threshold
        self._ensemble_parameters['last_visit_limit'] = self._last_visit_limit

        # add species count to ensemble parameters
        symbols = set([symbol for sub in calculator.sublattices
                       for symbol in sub.chemical_symbols])
        for symbol in symbols:
            key = 'n_atoms_{}'.format(symbol)
            count = structure.get_chemical_symbols().count(symbol)
            self._ensemble_parameters[key] = count

        # the constructor of the parent classes must be called *after*
        # the ensemble_parameters dict has been populated
        super().__init__(
            structure=structure,
            calculator=calculator,
            user_tag=user_tag,
            data_container=data_container,
            random_seed=random_seed,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

        # handle probabilities for swaps on different sublattices
        if sublattice_probabilities is None:
            self._swap_sublattice_probabilities = self._get_swap_sublattice_probabilities()
        else:
            self._swap_sublattice_probabilities = sublattice_probabilities

        # intialize Wang-Landau algorithm
        if not hasattr(self, '_converged'):
            self._converged = False
        if not hasattr(self, '_fill_factor'):
            self._fill_factor = 1
        if not hasattr(self, '_histogram'):
            self._histogram = {}
        if not hasattr(self, '_last_visit'):
            self._last_visit = {}
        if not hasattr(self, '_entropy'):
            self._entropy = {}
        self._potential = self.calculator.calculate_total(
            occupations=self.configuration.occupations)

    def _restart_ensemble(self):
        """ Restarts ensemble using the last state saved in DataContainer file.
        """
        super()._restart_ensemble()
        last_saved = self.data_container.data.iloc[-1]
        self._converged = last_saved.converged
        self._fill_factor = last_saved.fill_factor
        self._histogram = last_saved.histogram.copy()
        self._last_visit = last_saved.last_visit.copy()
        self._entropy = last_saved.entropy.copy()

    def do_canonical_swap(self, sublattice_index: int, allowed_species: List[int] = None) -> int:
        """ Carries out one Monte Carlo trial step. This method
        has been taken from CanonicalEnsemble.

        Parameters
        ---------
        sublattice_index
            the sublattice the swap will act on
        allowed_species
            list of atomic numbers for allowed species

        Returns
        -------
        Returns 1 or 0 depending on if trial move was accepted or rejected
        """
        sites, species = self.configuration.get_swapped_state(sublattice_index, allowed_species)
        potential_diff = self._get_property_change(sites, species)

        if self._acceptance_condition(potential_diff):
            self.update_occupations(sites, species)
            return 1
        return 0

    def _get_swap_sublattice_probabilities(self) -> List[float]:
        """ Returns sublattice probabilities suitable for swaps. This method
        has been taken from CanonicalEnsemble. """
        sublattice_probabilities = []
        for i, sl in enumerate(self.sublattices):
            if self.configuration.is_swap_possible(i):
                sublattice_probabilities.append(len(sl.indices))
            else:
                sublattice_probabilities.append(0)
        norm = sum(sublattice_probabilities)
        if norm == 0:
            raise ValueError('No canonical swaps are possible on any of the active sublattices.')
        sublattice_probabilities = [p / norm for p in sublattice_probabilities]
        return sublattice_probabilities

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. This method
        has been taken from CanonicalEnsemble. """
        sublattice_index = self.get_random_sublattice_index(self._swap_sublattice_probabilities)
        return self.do_canonical_swap(sublattice_index=sublattice_index)

    def _acceptance_condition(self, potential_diff: float) -> bool:
        """
        Evaluates Metropolis acceptance criterion.

        Parameters
        ----------
        potential_diff
            change in the thermodynamic potential associated
            with the trial step
        """

        # acceptance/rejection step
        ix_cur = int(np.around(self._potential / self._energy_spacing))
        ix_new = int(np.around((self._potential + potential_diff) / self._energy_spacing))
        S_cur = self._entropy.get(ix_cur, 0)
        S_new = self._entropy.get(ix_new, 0)
        delta = np.exp(S_cur - S_new)
        if delta > 1 or delta > self._next_random_number():
            accept = True
            self._potential += potential_diff
            ix_cur = ix_new
        else:
            accept = False

        # update histogram and entropy
        self._entropy[ix_cur] = self._entropy.get(ix_cur, 0) + self._fill_factor
        self._histogram[ix_cur] = self._histogram.get(ix_cur, 0) + 1
        self._last_visit[ix_cur] = self._step

        # check flatness of histogram
        if self._step > 0 and self._step % self._fill_factor_update_interval == 0:
            # only include histogram bins that have been visited within a reasonable time
            histogram_masked = np.ma.masked_where(
                np.array(list(self._last_visit.values())) > self._step - self._last_visit_limit,
                np.array(list(self._histogram.values())))
            limit = self._flatness_threshold * np.average(histogram_masked)
            if np.all(histogram_masked > limit):
                self._fill_factor /= 2
                self._histogram = {}
                self._last_visit = {}
                if self._fill_factor <= self._fill_factor_limit:
                    self._converged = True

        return accept

    def _get_ensemble_data(self) -> dict:
        """ Returns the default ensemble data of the parent class as well as
        the data related to the Wang-Landau algorithm."""
        data = super()._get_ensemble_data()
        self._potential = data['potential']
        data['converged'] = self._converged
        data['fill_factor'] = self._fill_factor
        data['histogram'] = OrderedDict(sorted(self._histogram.items()))
        data['entropy'] = OrderedDict(sorted(self._entropy.items()))
        data['last_visit'] = OrderedDict(sorted(self._last_visit.items()))
        return data

    def _terminate_sampling(self):
        """ Returns whether the Wang-Landau algorithm has converged. This is
        used in the run method implemented in the BaseEnsemble to evaluate
        whether the sampling loop should be terminated. """
        return self._converged


def get_wang_landau_data(dc: DataContainer, tag: str = 'entropy', step: int = -1,
                         normalize: bool = True) -> DataFrame:
    """ Returns density of states, entropy or histogram from a Wang-Landau simulation.

    Parameters
    ----------
    dc
        data container from a Wang-Landau simulation
    tag
        type of data returned; can be either 'density', 'entropy' or 'histogram'
    step
        Monte Carlo trial step for which to return data
    normalize
        if True the data will be normalized
    """
    if tag not in ['density', 'histogram', 'entropy']:
        raise ValueError('tag ({}) must be either "density", "histogram" or "entropy"'.format(tag))
    energy_spacing = dc.ensemble_parameters['energy_spacing']
    if tag == 'density':
        data = OrderedDict(sorted(dc.data['entropy'].iloc[step].items()))
    else:
        data = OrderedDict(sorted(dc.data[tag].iloc[step].items()))
    energies = energy_spacing * np.array(list(data.keys()))
    data = np.array(list(data.values()))
    if tag == 'density':
        data = np.exp(-data)
    if normalize:
        data = data / np.sum(data)
    return DataFrame(data={'energy': energies, 'data': data})
