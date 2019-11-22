"""Definition of the Wang-Landau algorithm class."""

import random

from collections import OrderedDict
from typing import BinaryIO, Dict, List, TextIO, Union

import numpy as np

from ase import Atoms
from pandas import DataFrame

from .. import WangLandauDataContainer
from ..calculators.base_calculator import BaseCalculator
from .thermodynamic_base_ensemble import BaseEnsemble


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
    user_tag : str
        human-readable tag for ensemble [default: None]
    data_container : str
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
        the data [default: np.inf]
    trajectory_write_interval : int
        interval at which the current occupation vector of the atomic
        configuration is written to the data container.
    sublattice_probabilities : List[float]
        probability for picking a sublattice when doing a random swap.
        The list must contain as many elements as there are sublattices
        and it needs to sum up to 1.

    Examples
    --------
    The following snippet illustrates how to carry out a Wang-Landau
    simulation. For the purpose of demonstration, the parameters of
    the cluster expansion are set to obtain a two-dimensional square
    Ising model, one of the systems studied in the original work by
    Wang and Landau::

        from ase import Atoms
        from icet import ClusterExpansion, ClusterSpace
        from mchammer.calculators import ClusterExpansionCalculator
        from mchammer.ensembles import WangLandauEnsemble

        # prepare cluster expansion
        prim = Atoms('Au', positions=[[0, 0, 0]], cell=[1, 1, 10], pbc=True)
        cs = ClusterSpace(prim, cutoffs=[1.1], chemical_symbols=['Ag', 'Au'])
        ce = ClusterExpansion(cs, [0, 0, 2])

        # prepare initial configuration
        structure = prim.repeat((4, 4, 1))
        for k in range(8):
            structure[k].symbol = 'Ag'

        # set up and run Wang-Landau simulation
        calculator = ClusterExpansionCalculator(structure, ce)
        mc = WangLandauEnsemble(structure=structure,
                                calculator=calculator,
                                energy_spacing=1,
                                data_container='ising_2d_run.dc')
        mc.run(number_of_trial_steps=len(structure)*10000)  # carry out 10,000 MC sweeps

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
                 user_tag: str = None,
                 data_container: str = None,
                 random_seed: int = None,
                 data_container_write_period: float = np.inf,
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
            flatness_check_interval = len(structure) * 1e3

        # parameters pertaining to construction of entropy and histogram
        self._energy_spacing = energy_spacing
        self._fill_factor_limit = fill_factor_limit
        self._flatness_check_interval = flatness_check_interval
        self._flatness_limit = flatness_limit

        # energy window
        self._bin_left = self._get_bin_index(energy_limit_left)
        self._bin_right = self._get_bin_index(energy_limit_right)
        if self._bin_left is not None and \
                self._bin_right is not None and self._bin_left >= self._bin_right:
            raise ValueError('Invalid energy window: left boundary ({}, {}) must be'
                             ' smaller than right boundary ({}, {})'
                             .format(energy_limit_left, self._bin_left,
                                     energy_limit_right, self._bin_right))
        self._reached_energy_window = self._bin_left is None and self._bin_right is None

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
            random_seed=random_seed,
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
        self._converged = None
        self._potential = self.calculator.calculate_total(
            occupations=self.configuration.occupations)
        if not hasattr(self, '_fill_factor'):
            self._fill_factor = 1
        if not hasattr(self, '_fill_factor_history'):
            if self._reached_energy_window:
                self._fill_factor_history = {self.step: self._fill_factor}
            else:
                self._fill_factor_history = {}
        if not hasattr(self, '_histogram'):
            self._histogram = {}
        if not hasattr(self, '_entropy'):
            self._entropy = {}

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
    def converged(self) -> bool:
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

    def get_entropy(self) -> DataFrame:
        """ entropy accumulated during Wang-Landau simulation """
        df = DataFrame(data={'energy': self._energy_spacing * np.array(list(self._entropy.keys())),
                             'entropy': np.array(list(self._entropy.values()))},
                       index=list(self._entropy.keys()))
        df['entropy'] -= np.min(df['entropy'])
        return df

    def get_histogram(self) -> DataFrame:
        """ histogram accumulated since last update of fill factor """
        df = DataFrame(data={'energy': self._energy_spacing *
                             np.array(list(self._histogram.keys())),
                             'histogram': np.array(list(self._histogram.values()))},
                       index=list(self._histogram.keys()))
        df['histogram'] -= np.min(df['histogram'])
        return df

    def _terminate_sampling(self) -> bool:
        """Returns True if the Wang-Landau algorithm has converged. This is
        used in the run method implemented of BaseEnsemble to
        evaluate whether the sampling loop should be terminated.
        """
        return self._converged is True  # N.B.: self._converged can be None

    def _restart_ensemble(self):
        """Restarts ensemble using the last state saved in the data container
        file. Note that this method does _not_ use the last_state property of
        the data container but rather uses the last data written the data frame.
        """
        super()._restart_ensemble()
        self._fill_factor = self.data_container.last_state['fill_factor']
        self._fill_factor_history = self.data_container.last_state['fill_factor_history']
        self._histogram = self.data_container.last_state['histogram']
        self._entropy = self.data_container.last_state['entropy']

    def write_data_container(self, outfile: Union[str, BinaryIO, TextIO]):
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
        bin_cur = self._get_bin_index(self._potential)
        bin_new = self._get_bin_index(self._potential + potential_diff)
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

            histogram = np.array(list(self._histogram.values()))
            limit = self._flatness_limit * np.average(histogram)
            if np.all(histogram >= limit):

                # check whether the Wang-Landau algorithm has converged
                self._converged = (self._fill_factor <= self._fill_factor_limit)
                if not self._converged:
                    # update fill factor and reset histogram
                    self._fill_factor /= 2
                    self._fill_factor_history[self.step] = self._fill_factor
                    self._histogram = dict.fromkeys(self._histogram, 0)

    def _get_bin_index(self, energy: float) -> int:
        """ Returns bin index for histogram and entropy dictionaries. """
        if energy is None or np.isnan(energy):
            return None
        return int(np.around(energy / self._energy_spacing))

    def _allow_move(self, bin_cur: int, bin_new: int) -> bool:
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
        sublattice_probabilities = []
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
        sublattice_probabilities = []
        for _, sl in enumerate(self.sublattices):
            if len(sl.chemical_symbols) > 1:
                sublattice_probabilities.append(len(sl.indices))
            else:
                sublattice_probabilities.append(0)
        norm = sum(sublattice_probabilities)
        sublattice_probabilities = [p / norm for p in sublattice_probabilities]
        return sublattice_probabilities
