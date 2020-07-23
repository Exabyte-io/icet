"""Definition of the Wang-Landau algorithm class."""

import random

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ase import Atoms

from .. import WangLandauDataContainer
from ..calculators.base_calculator import BaseCalculator
from .thermodynamic_base_ensemble import BaseEnsemble
from icet.input_output.logging_tools import logger

logger = logger.getChild('wang_landau_ensemble')


class WangLandauEnsemble(BaseEnsemble):
    """Instances of this class allow one to sample a system using the
    Wang-Landau (WL) algorithm, see Phys. Rev. Lett. **86**, 2050
    (2001) [WanLan01a]_. The WL algorithm enables one to acquire the
    density of states (DOS) as a function of energy, from which one
    can readily calculate many thermodynamic observables as a function
    of temperature. To this end, the WL algorithm accumulates both the
    microcanonical entropy :math:`S(E)` and a histogram :math:`H(E)`
    on an energy grid with a predefined spacing (``energy_spacing``).

    The algorithm is initialized as follows.

     #. Generate an initial configuration.
     #. Initialize counters for the microcanonical entropy
        :math:`S(E)` and the histogram :math:`H(E)` to zero.
     #. Set the fill factor :math:`f=1`.

    It then proceeds as follows.

    #. Propose a new configuration (see ``trial_move``).
    #. Accept or reject the new configuration with probability

       .. math::

          P = \\min \\{ 1, \\, \\exp [ S(E_\\mathrm{new}) - S(E_\\mathrm{cur}) ] \\},

       where :math:`E_\\mathrm{cur}` and :math:`E_\\mathrm{new}` are the
       energies of the current and new configurations, respectively.
    #. Update the microcanonical entropy :math:`S(E)\\leftarrow S(E) + f`
       and histogram :math:`H(E) \\leftarrow H(E) + 1` where
       :math:`E` is the energy of the system at the end of the move.
    #. Check the flatness of the histogram :math:`H(E)`. If
       :math:`H(E) > \\chi \\langle H(E)\\rangle\\,\\forall E` reset the histogram
       :math:`H(E) = 0` and reduce the fill factor :math:`f \\leftarrow f / 2`.
       The parameter :math:`\\chi` is set via ``flatness_limit``.
    #. If :math:`f` is smaller than ``fill_factor_limit`` terminate
       the loop, otherwise return to 1.

    The microcanonical entropy :math:`S(E)` and the histogram along
    with related information are written to the data container every
    time :math:`f` is updated. Using the density :math:`\\rho(E) = \\exp S(E)`
    one can then readily compute various thermodynamic quantities,
    including, e.g., the average energy:

    .. math::

       \\left<E\\right> = \\frac{\\sum_E E \\rho(E) \\exp(-E / k_B T)}{
       \\sum_E \\rho(E) \\exp(-E / k_B T)}

    Parameters
    ----------
    structure : :class:`Atoms <ase.Atoms>`
        atomic configuration to be used in the Wang-Landau simulation;
        also defines the initial occupation vector
    calculator : :class:`BaseCalculator <mchammer.calculators.ClusterExpansionCalculator>`
        calculator to be used for calculating potential changes
    trial_move : str
        One can choose between two different trial moves for
        generating new configurations. In a 'swap' move two sites are
        selected and their occupations are swapped; in a 'flip' move
        one site is selected and its occupation is flipped to a
        different species. While 'swap' moves conserve the
        concentrations of the species in the system, 'flip' moves
        allow one in principle to sample the full composition space.
    energy_spacing : float
        defines the bin size of the energy grid on which the microcanonical
        entropy :math:`S(E)`, and thus the density :math:`\\exp S(E)`, is
        evaluated; the spacing should be small enough to capture the features
        of the density of states; too small values will, however, render the
        convergence very tedious if not impossible
    energy_limit_left : float
        defines the lower limit of the energy range within which the
        microcanonical entropy :math:`S(E)` will be sampled. By default
        (`None`) no limit is imposed. Setting limits can be useful if only a
        part of the density of states is required.
    energy_limit_right : float
        defines the upper limit of the energy range within which the
        microcanonical entropy :math:`S(E)` will be sampled. By default
        (`None`) no limit is imposed. Setting limits can be useful if only a
        part of the density of states is required.
    fill_factor_limit : float
        If the fill_factor :math:`f` falls below this value, the
        WL sampling loop is terminated.
    flatness_check_interval : int
        For computational efficiency the flatness condition is only
        evaluated every ``flatness_check_interval``-th trial step. By
        default (``None``) ``flatness_check_interval`` is set to 1000
        times the number of sites in ``structure``, i.e. 1000 Monte
        Carlo sweeps.
    flatness_limit : float
        The histogram :math:`H(E)` is deemed sufficiently flat if
        :math:`H(E) > \\chi \\left<H(E)\\right>\\,\\forall
        E`. ``flatness_limit`` sets the parameter :math:`\\chi`.
    window_search_penalty : float
        If `energy_limit_left` and/or `energy_limit_right` have been
        provided, a modified acceptance probability,
        :math:`P=\\min\\{1,\\,\\exp[C_\\mathrm{WSP}(d_\\mathrm{new}-
        d_\\mathrm{cur})]\\}`, will be used until a configuration is
        found within the interval of interest. This parameter,
        specifically, corresponds to :math:`C_\\mathrm{WSP}`, which
        controls how strongly moves that lead to an increase in the
        distance, i.e. difference in energy divided by the energy
        spacing, to the energy window (:math:`d_\\mathrm{new}>
        d_\\mathrm{cur}`) should be penalized. A higher value leads
        to a lower acceptance probability for such moves.
    user_tag : str
        human-readable tag for ensemble [default: None]
    dc_filename : str
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
        the data [default: 600 s]
    trajectory_write_interval : int
        interval at which the current occupation vector of the atomic
        configuration is written to the data container.
    sublattice_probabilities : List[float]
        probability for picking a sublattice when doing a random swap.
        The list must contain as many elements as there are sublattices
        and it needs to sum up to 1.

    Example
    -------
    The following snippet illustrates how to carry out a Wang-Landau
    simulation. For the purpose of demonstration, the parameters of
    the cluster expansion are set to obtain a two-dimensional square
    Ising model, one of the systems studied in the original work by
    Wang and Landau::

        >>> from ase import Atoms
        >>> from icet import ClusterExpansion, ClusterSpace
        >>> from mchammer.calculators import ClusterExpansionCalculator
        >>> from mchammer.ensembles import WangLandauEnsemble

        >>> # prepare cluster expansion
        >>> prim = Atoms('Au', positions=[[0, 0, 0]], cell=[1, 1, 10], pbc=True)
        >>> cs = ClusterSpace(prim, cutoffs=[1.1], chemical_symbols=['Ag', 'Au'])
        >>> ce = ClusterExpansion(cs, [0, 0, 2])

        >>> # prepare initial configuration
        >>> structure = prim.repeat((4, 4, 1))
        >>> for k in range(8):
        ...     structure[k].symbol = 'Ag'

        >>> # set up and run Wang-Landau simulation
        >>> calculator = ClusterExpansionCalculator(structure, ce)
        >>> mc = WangLandauEnsemble(structure=structure,
        ...                         calculator=calculator,
        ...                         energy_spacing=1,
        ...                         dc_filename='ising_2d_run.dc')
        >>> mc.run(number_of_trial_steps=len(structure)*1000)  # in practice one requires more steps

    """

    def __init__(self,
                 structure: Atoms,
                 calculator: BaseCalculator,
                 energy_spacing: float,
                 energy_limit_left: float = None,
                 energy_limit_right: float = None,
                 trial_move: str = 'swap',
                 fill_factor_limit: float = 1e-6,
                 flatness_check_interval: int = None,
                 flatness_limit: float = 0.8,
                 window_search_penalty: float = 2.0,
                 user_tag: str = None,
                 dc_filename: str = None,
                 data_container: str = None,
                 random_seed: int = None,
                 data_container_write_period: float = 600,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None,
                 sublattice_probabilities: List[float] = None) -> None:

        # set trial move
        if trial_move == 'swap':
            self.do_move = self._do_swap
            self._get_sublattice_probabilities = self._get_swap_sublattice_probabilities
        elif trial_move == 'flip':
            self.do_move = self._do_flip
            self._get_sublattice_probabilities = self._get_flip_sublattice_probabilities
        else:
            raise ValueError('Invalid value for trial_move: {}.'
                             ' Must be either "swap" or "flip".'.format(trial_move))

        # set default values that are system dependent
        if flatness_check_interval is None:
            flatness_check_interval = len(structure) * 1000

        # parameters pertaining to construction of entropy and histogram
        self._energy_spacing = energy_spacing
        self._fill_factor_limit = fill_factor_limit
        self._flatness_check_interval = flatness_check_interval
        self._flatness_limit = flatness_limit

        # energy window
        self._window_search_penalty = window_search_penalty
        self._bin_left = self._get_bin_index(energy_limit_left)
        self._bin_right = self._get_bin_index(energy_limit_right)
        if self._bin_left is not None and \
                self._bin_right is not None and self._bin_left >= self._bin_right:
            raise ValueError('Invalid energy window: left boundary ({}, {}) must be'
                             ' smaller than right boundary ({}, {})'
                             .format(energy_limit_left, self._bin_left,
                                     energy_limit_right, self._bin_right))

        # ensemble parameters
        self._ensemble_parameters = {}
        self._ensemble_parameters['energy_spacing'] = energy_spacing
        self._ensemble_parameters['trial_move'] = trial_move
        self._ensemble_parameters['energy_limit_left'] = energy_limit_left
        self._ensemble_parameters['energy_limit_right'] = energy_limit_right
        # The following parameters are _intentionally excluded_ from
        # the ensemble_parameters dict as it would prevent users from
        # changing their values between restarts. The latter is advantageous
        # as these runs can require restarts and possibly parameter adjustments
        # to achieve convergence.
        #  * fill_factor_limit
        #  * flatness_check_interval
        #  * flatness_limit
        #  * entropy_write_frequency
        #  * window_search_penalty

        # add species count to ensemble parameters
        symbols = set([symbol for sub in calculator.sublattices
                       for symbol in sub.chemical_symbols])
        for symbol in symbols:
            key = 'n_atoms_{}'.format(symbol)
            count = structure.get_chemical_symbols().count(symbol)
            self._ensemble_parameters[key] = count

        # set the convergence, which may be updated in case of a restart
        self._converged = None  # type: Optional[bool]

        # the constructor of the parent classes must be called *after*
        # the ensemble_parameters dict has been populated
        super().__init__(
            structure=structure,
            calculator=calculator,
            user_tag=user_tag,
            random_seed=random_seed,
            dc_filename=dc_filename,
            data_container=data_container,
            data_container_class=WangLandauDataContainer,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

        # handle probabilities for swaps on different sublattices
        if sublattice_probabilities is None:
            self._sublattice_probabilities = self._get_sublattice_probabilities()
        else:
            self._sublattice_probabilities = sublattice_probabilities

        # initialize Wang-Landau algorithm; in the case of a restart
        # these quantities are read from the data container file; the
        # if-conditions prevent these values from being overwritten
        self._potential = self.calculator.calculate_total(
            occupations=self.configuration.occupations)
        self._reached_energy_window = self._inside_energy_window(
            self._get_bin_index(self._potential))
        if not hasattr(self, '_fill_factor'):
            self._fill_factor = 1.0
        if not hasattr(self, '_fill_factor_history'):
            if self._reached_energy_window:
                self._fill_factor_history = {self.step: self._fill_factor}
            else:
                self._fill_factor_history = {}
        if not hasattr(self, '_entropy_history'):
            self._entropy_history = {}
        if not hasattr(self, '_histogram'):
            self._histogram = {}  # type: Dict[int, int]
        if not hasattr(self, '_entropy'):
            self._entropy = {}  # type: Dict[int, float]

    @property
    def fill_factor(self) -> float:
        """ current value of fill factor """
        return self._fill_factor

    @property
    def fill_factor_history(self) -> Dict[int, float]:
        """evolution of the fill factor in the Wang-Landau algorithm (key=MC
        trial step, value=fill factor)
        """
        return self._fill_factor_history

    @property
    def converged(self) -> Optional[bool]:
        """ True if convergence has been achieved """
        return self._converged

    @property
    def flatness_limit(self) -> float:
        """The histogram :math:`H(E)` is deemed sufficiently flat if
        :math:`H(E) > \\chi \\left<H(E)\\right>\\,\\forall
        E` where ``flatness_limit`` sets the parameter :math:`\\chi`.
        """
        return self._flatness_limit

    @flatness_limit.setter
    def flatness_limit(self, new_value) -> None:
        self._flatness_limit = new_value
        self._converged = None

    @property
    def fill_factor_limit(self) -> float:
        """ If the fill_factor :math:`f` falls below this value, the
        Wang-Landau sampling loop is terminated. """
        return self._fill_factor_limit

    @fill_factor_limit.setter
    def fill_factor_limit(self, new_value) -> None:
        self._fill_factor_limit = new_value
        self._converged = None

    @property
    def flatness_check_interval(self) -> int:
        """ number of MC trial steps between checking the flatness
        condition """
        return self._flatness_check_interval

    @flatness_check_interval.setter
    def flatness_check_interval(self, new_value: int) -> None:
        self._flatness_check_interval = new_value

    def run(self, number_of_trial_steps: int):
        """
        Samples the ensemble for the given number of trial steps.

        Parameters
        ----------
        number_of_trial_steps
            maximum number of MC trial steps to run in total (the
            run will terminate earlier if `fill_factor_limit` is reached)
        reset_step
            if True the MC trial step counter and the data container will
            be reset to zero and empty, respectively.

        Raises
        ------
        TypeError
            if `number_of_trial_steps` is not an int
        """
        if self.converged:
            logger.warning('Convergence has already been reached.')
        else:
            super().run(number_of_trial_steps)

    def _terminate_sampling(self) -> bool:
        """Returns True if the Wang-Landau algorithm has converged. This is
        used in the run method implemented of BaseEnsemble to
        evaluate whether the sampling loop should be terminated.
        """
        # N.B.: self._converged can be None
        if self._converged is not None:
            return self._converged
        else:
            return False

    def _restart_ensemble(self):
        """Restarts ensemble using the last state saved in the data container
        file. Note that this method does _not_ use the last_state property of
        the data container but rather uses the last data written the data frame.
        """
        super()._restart_ensemble()
        self._fill_factor = self.data_container._last_state['fill_factor']
        self._fill_factor_history = self.data_container._last_state['fill_factor_history']
        self._entropy_history = self.data_container._last_state['entropy_history']
        self._histogram = self.data_container._last_state['histogram']
        self._entropy = self.data_container._last_state['entropy']
        histogram = np.array(list(self._histogram.values()))
        limit = self._flatness_limit * np.average(histogram)
        self._converged = (self._fill_factor <= self._fill_factor_limit
                           ) & np.all(histogram >= limit)

    def write_data_container(self, outfile: Union[str, bytes]):
        """Updates last state of the Wang-Landau simulation and
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
            random_state=random.getstate(),
            fill_factor=self._fill_factor,
            fill_factor_history=self._fill_factor_history,
            entropy_history=self._entropy_history,
            histogram=OrderedDict(sorted(self._histogram.items())),
            entropy=OrderedDict(sorted(self._entropy.items())))
        self.data_container.write(outfile)

    def _acceptance_condition(self, potential_diff: float) -> bool:
        """Evaluates Metropolis acceptance criterion.

        Parameters
        ----------
        potential_diff
            change in the thermodynamic potential associated
            with the trial step
        """

        # acceptance/rejection step
        bin_old = self._get_bin_index(self._potential)
        bin_new = self._get_bin_index(self._potential + potential_diff)
        bin_cur = bin_old
        if self._allow_move(bin_cur, bin_new):
            S_cur = self._entropy.get(bin_cur, 0)
            S_new = self._entropy.get(bin_new, 0)
            delta = np.exp(S_cur - S_new)
            if delta >= 1 or delta >= self._next_random_number():
                accept = True
                self._potential += potential_diff
                bin_cur = bin_new
            else:
                accept = False
        else:
            accept = False

        if not self._reached_energy_window:
            # check whether the target energy window has been reached
            self._reached_energy_window = self._inside_energy_window(bin_cur)
            # if the target window has been reached remove unused bins
            # from histogram and entropy counters
            if self._reached_energy_window:
                self._fill_factor_history[self.step] = self._fill_factor
                # flush data from data container except for initial step
                self._data_container._data_list = [self._data_container._data_list[0]]
                self._entropy = {k: self._entropy[k]
                                 for k in self._entropy if self._inside_energy_window(k)}
                self._histogram = {k: self._histogram[k]
                                   for k in self._histogram if self._inside_energy_window(k)}
            else:
                # then reconsider accept/reject based on whether we
                # approached the window or not
                dist_new = np.inf
                dist_old = np.inf
                if self._bin_left is not None:
                    dist_new = min(dist_new, abs(bin_new - self._bin_left))
                    dist_old = min(dist_old, abs(bin_old - self._bin_left))
                if self._bin_right is not None:
                    dist_new = min(dist_new, abs(bin_new - self._bin_right))
                    dist_old = min(dist_old, abs(bin_old - self._bin_right))
                assert dist_new < np.inf and dist_old < np.inf
                exp_dist = np.exp((dist_old - dist_new) * self._window_search_penalty)
                if exp_dist >= 1 or exp_dist >= self._next_random_number():
                    # should be accepted
                    if not accept:
                        # reset potential
                        self._potential += potential_diff
                    bin_cur = bin_new
                    accept = True
                else:
                    # should be rejected
                    if accept:
                        # reset potential
                        self._potential -= potential_diff
                    bin_cur = bin_old
                    accept = False

        # update histograms and entropy counters
        self._update_entropy(bin_cur)

        return accept

    def _update_entropy(self, bin_cur: int) -> None:
        """Updates counters for histogram and entropy, checks histogram
        flatness, and updates fill factor if indicated.
        """

        # update histogram and entropy
        self._entropy[bin_cur] = self._entropy.get(bin_cur, 0) + self._fill_factor
        self._histogram[bin_cur] = self._histogram.get(bin_cur, 0) + 1

        # check flatness of histogram
        if self.step % self._flatness_check_interval == 0 and \
                self.step > 0 and self._reached_energy_window:

            # shift entropy counter in order to avoid overflow
            entropy_ref = np.min(list(self._entropy.values()))
            for k in self._entropy:
                self._entropy[k] -= entropy_ref

            # check whether the Wang-Landau algorithm has converged
            histogram = np.array(list(self._histogram.values()))
            limit = self._flatness_limit * np.average(histogram)
            is_flat = np.all(histogram >= limit)
            self._converged = (self._fill_factor <= self._fill_factor_limit) & is_flat
            if is_flat and not self._converged:
                # update fill factor
                self._fill_factor /= 2
                self._fill_factor_history[self.step] = self._fill_factor
                # update entropy history
                self._entropy_history[self.step] = OrderedDict(
                    sorted(self._entropy.items()))
                # reset histogram
                self._histogram = dict.fromkeys(self._histogram, 0)

    def _get_bin_index(self, energy: float) -> Optional[int]:
        """ Returns bin index for histogram and entropy dictionaries. """
        if energy is None or np.isnan(energy):
            return None
        return int(np.around(energy / self._energy_spacing))

    def _allow_move(self, bin_cur: Optional[int], bin_new: int) -> bool:
        """Returns True if the current move is to be included in the
        accumulation of histogram and entropy. This logic has been
        moved into a separate function in order to enhance
        readability.
        """
        if self._bin_left is None and self._bin_right is None:
            # no limits on energy window
            return True
        if self._bin_left is not None:
            if bin_cur < self._bin_left:
                # not yet in window (left limit)
                return True
            if bin_new < self._bin_left:
                # imposing left limit
                return False
        if self._bin_right is not None:
            if bin_cur > self._bin_right:
                # not yet in window (right limit)
                return True
            if bin_new > self._bin_right:
                # imposing right limit
                return False
        return True

    def _inside_energy_window(self, bin_k: int) -> bool:
        """Returns True if bin_k is inside the energy window specified for
        this simulation.
        """
        if self._bin_left is not None and bin_k < self._bin_left:
            return False
        if self._bin_right is not None and bin_k > self._bin_right:
            return False
        return True

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        sublattice_index = self.get_random_sublattice_index(self._sublattice_probabilities)
        return self.do_move(sublattice_index=sublattice_index)

    def _do_swap(self, sublattice_index: int, allowed_species: List[int] = None) -> int:
        """Carries out a Monte Carlo trial that involves swapping the species
        on two sites. This method has been copied from
        ThermodynamicBaseEnsemble.

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

    def _do_flip(self, sublattice_index: int, allowed_species: List[int] = None) -> int:
        """Carries out one Monte Carlo trial step that involves flipping the
        species on one site. This method has been adapted from
        ThermodynamicBaseEnsemble.

        Parameters
        ---------
        sublattice_index
            the sublattice the flip will act on
        allowed_species
            list of atomic numbers for allowed species

        Returns
        -------
        Returns 1 or 0 depending on if trial move was accepted or rejected

        """
        index, species = self.configuration.get_flip_state(sublattice_index, allowed_species)
        potential_diff = self._get_property_change([index], [species])
        if self._acceptance_condition(potential_diff):
            self.update_occupations([index], [species])
            return 1
        return 0

    def _get_swap_sublattice_probabilities(self) -> List[float]:
        """Returns sublattice probabilities suitable for swaps. This method
        has been copied without modification from ThermodynamicBaseEnsemble.
        """
        sublattice_probabilities = []  # type: List[Any]
        for i, sl in enumerate(self.sublattices):
            if self.configuration.is_swap_possible(i):
                sublattice_probabilities.append(len(sl.indices))
            else:
                sublattice_probabilities.append(0)
        norm = sum(sublattice_probabilities)
        if norm == 0:
            raise ValueError('No swaps are possible on any of the active sublattices.')
        sublattice_probabilities = [p / norm for p in sublattice_probabilities]
        return sublattice_probabilities

    def _get_flip_sublattice_probabilities(self) -> List[float]:
        """Returns the default sublattice probability which is based on the
        sizes of a sublattice. This method has been copied without
        modification from ThermodynamicBaseEnsemble.
        """
        sublattice_probabilities = []  # type: List[Any]
        for _, sl in enumerate(self.sublattices):
            if len(sl.chemical_symbols) > 1:
                sublattice_probabilities.append(len(sl.indices))
            else:
                sublattice_probabilities.append(0)
        norm = sum(sublattice_probabilities)
        sublattice_probabilities = [p / norm for p in sublattice_probabilities]
        return sublattice_probabilities


def get_bins_for_parallel_simulations(n_bins: int,
                                      energy_spacing: float,
                                      minimum_energy: float,
                                      maximum_energy: float,
                                      overlap: int = 4,
                                      bin_size_exponent: float = 1.0) -> List[Tuple[float, float]]:
    """Generates a list of energy bins (lower and upper bound) suitable for
    parallel Wang-Landau simulations. For the latter, the energy range is
    split up into a several bins (``n_bins``). Each bin is then sampled in a
    separate Wang-Landau simulation. Once the density of states in the
    individual bins has been converged the total density of states can be
    constructed by patching the segments back together. To this end, one
    requires some over overlap between the segments (``overlap``).

    The function returns a list of tuples. Each tuple provides the lower
    (``energy_limit_left``) and upper (``energy_limit_right``) bound of one
    bin, which are then to be used to set ``energy_limit_left`` and
    ``energy_limit_right`` when initializing a :class:`WangLandauEnsemble`
    instance.

    N.B.: The left-most/right-most bin has no lower/upper bound (set to
    ``None``).

    Parameters
    ----------
    n_bins
        number of bins
    energy_spacing
        defines the bin size of the energy grid used by the Wang-Landau
        simulation, see :class:`WangLandauEnsemble` for details
    minimum_energy
        an estimate for the lowest energy to be encountered in this system
    maximum_energy
        an estimate for the highest energy to be encountered in this system
    overlap
        amount of overlap between bins in units of ``energy_spacing``
    bin_size_exponent
        *Expert option*: This parameter allows one to generate a non-uniform
        distribution of bin sizes. If ``bin_size_exponent`` is smaller than
        one bins at the lower and upper end of the energy range (specified via
        ``minimum_energy`` and ``maximum_energy``) will be shrunk relative to
        the bins in the middle of the energy range. In principle this can be
        used one to achieve a more even distribution of computational load
        between the individual Wang-Landau simulations.
    """

    limits = np.linspace(-1, 1, n_bins + 1)
    limits = np.sign(limits) * np.abs(limits) ** bin_size_exponent
    limits *= 0.5 * (maximum_energy - minimum_energy)
    limits += 0.5 * (maximum_energy + minimum_energy)
    limits[0], limits[-1] = None, None

    bounds = []
    for k, (energy_limit_left, energy_limit_right) in enumerate(zip(limits[:-1], limits[1:])):
        if energy_limit_left is not None and energy_limit_right is not None and \
              (energy_limit_right - energy_limit_left) / energy_spacing < 2 * overlap:
            raise ValueError('Energy window too small. min/max: {}/{}'
                             .format(energy_limit_right, energy_limit_left) +
                             ' Try decreasing n_bins ({}) and/or overlap ({}).'
                             .format(n_bins, overlap))
        if energy_limit_left is not None:
            energy_limit_left -= overlap * energy_spacing
        if energy_limit_right is not None:
            energy_limit_right += overlap * energy_spacing
        bounds.append((energy_limit_left, energy_limit_right))

    return bounds
