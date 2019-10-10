"""Definition of the Wang-Landau multi-canonical ensemble class."""

from collections import OrderedDict
from typing import Dict, List

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
                 fill_factor_update_interval: int = 1e3,
                 fill_factor_limit: float = 1e-8,
                 flatness_threshold: float = 0.9,
                 last_visit_limit: int = None,
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
        # The following parameters are _intentionally excluded_ from
        # the ensemble_parameters dict as it would prevent users from
        # changing their values between restarts. The latter is advantageous
        # as these runs can require restarts as well as parameter adjustments
        # to achieve convergence.
        #  * fill_factor_update_interval
        #  * fill_factor_limit
        #  * flatness_threshold
        #  * last_visit_limit

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

        # initialize Wang-Landau algorithm; in the case of a restart
        # these quantities are read from the data container file; the
        # if-conditions prevent these values from being overwritten
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
        self._fraction_flat_histogram = None
        self._potential = self.calculator.calculate_total(
            occupations=self.configuration.occupations)

    def _restart_ensemble(self):
        """Restarts ensemble using the last state saved in DataContainer
        file.  Note that this method does _not_ use the last_state
        property of the data container but rather uses the last data
        written the data frame.
        """
        super()._restart_ensemble()
        df = self.data_container.data
        index = df[df['entropy'].notna()].index[-1]
        last_saved = df.iloc[index]
        self._converged = last_saved.converged
        self._fill_factor = last_saved.fill_factor
        self._histogram = last_saved.histogram
        self._last_visit = last_saved.last_visit
        self._entropy = last_saved.entropy

    def do_canonical_swap(self, sublattice_index: int, allowed_species: List[int] = None) -> int:
        """ Carries out one Monte Carlo trial step. This method
        has been copied without modification from CanonicalEnsemble.

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
        """Returns sublattice probabilities suitable for swaps. This method
        has been copied without modification from CanonicalEnsemble.
        """
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
        """Carries out one Monte Carlo trial step. This method has been
        copied without modification from CanonicalEnsemble.
        """
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
        if delta >= 1 or delta >= self._next_random_number():
            accept = True
            self._potential += potential_diff
            ix_cur = ix_new
        else:
            accept = False

        # update histogram and entropy
        self._entropy[ix_cur] = self._entropy.get(ix_cur, 0) + self._fill_factor
        self._histogram[ix_cur] = self._histogram.get(ix_cur, 0) + 1
        self._last_visit[ix_cur] = self.step

        # check flatness of histogram
        if self.step % self._fill_factor_update_interval == 0 and self.step > 0:

            # only include histogram bins that have been visited within a reasonable time
            if self._last_visit_limit is not None:
                histogram = np.ma.masked_where(
                    np.array(list(self._last_visit.values())) < self.step - self._last_visit_limit,
                    np.array(list(self._histogram.values())))
            else:
                histogram = np.array(list(self._histogram.values()))
            limit = self._flatness_threshold * np.average(histogram)
            self._fraction_flat_histogram = np.sum(histogram > limit) / len(self._histogram)

            if np.all(histogram > limit):

                # check whether the Wang-Landau algorithm has converged
                if self._fill_factor <= self._fill_factor_limit:
                    self._converged = True
                else:
                    # add current histogram, entropy, and related data to data container
                    self._add_histograms_to_data_container()
                    # update fill factor and reset histogram
                    self._fill_factor /= 2
                    self._histogram = dict.fromkeys(self._histogram, 0)
                    self._last_visit = dict.fromkeys(self._last_visit, 0)
                    self._fraction_flat_histogram = None

        return accept

    def _add_histograms_to_data_container(self):
        """This method adds information regarding the current state of the
        convergence of the Wang-Landau algorithm to the data
        container.
        """
        row_dict = {}
        row_dict['potential'] = self._potential
        row_dict['fraction_flat_histogram'] = self._fraction_flat_histogram
        row_dict['converged'] = self._converged
        row_dict['fill_factor'] = self._fill_factor
        row_dict['histogram'] = OrderedDict(sorted(self._histogram.items()))
        row_dict['entropy'] = OrderedDict(sorted(self._entropy.items()))
        row_dict['last_visit'] = OrderedDict(sorted(self._last_visit.items()))
        self._data_container.append(mctrial=self.step, record=row_dict)

    def _finalize(self) -> None:
        """This method is called from the run method after the conclusion of
        the MC cycles but before the data container is written. Here
        it is used to add the final state of the Wang-Landau algorithm
        to the data container in order to enable direct restarts.
        """
        self._add_histograms_to_data_container()

    def _terminate_sampling(self) -> bool:
        """Returns True if the Wang-Landau algorithm has converged. This is
        used in the run method implemented of BaseEnsemble to
        evaluate whether the sampling loop should be terminated.
        """
        return self._converged

    def _get_ensemble_data(self) -> Dict:
        """Returns the data associated with the ensemble. For the Wang-Landau
        this specifically includes the fraction of bins that satisfy
        the flatness condition.
        """
        data = super()._get_ensemble_data()
        if hasattr(self, '_fraction_flat_histogram'):
            data['fraction_flat_histogram'] = self._fraction_flat_histogram
        return data


def get_wang_landau_data(dc: DataContainer, tag: str = 'entropy', step: int = -1,
                         normalize: bool = True) -> DataFrame:
    """Returns density of states, entropy or histogram from a Wang-Landau
    simulation.

    Todo
    ----
    * it says step but it is actually the index

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
        data = OrderedDict(sorted(dc.data.iloc[step]['entropy'].items()))
    else:
        data = OrderedDict(sorted(dc.data.iloc[step][tag].items()))
    energies = energy_spacing * np.array(list(data.keys()))
    data = np.array(list(data.values()))
    if tag == 'density':
        data = np.exp(-data)
    if normalize:
        data = data / np.sum(data)
    return DataFrame(data={'energy': energies, 'data': data})
