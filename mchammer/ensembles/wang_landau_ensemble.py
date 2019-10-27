"""Definition of the Wang-Landau algorithm class."""

from collections import Counter, OrderedDict
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union

import numpy as np

from ase import Atoms
from ase.units import kB
from pandas import DataFrame, concat as pd_concat

from .. import DataContainer
from ..calculators.base_calculator import BaseCalculator
from ..observers.base_observer import BaseObserver
from .thermodynamic_base_ensemble import BaseEnsemble


class WangLandauEnsemble(BaseEnsemble):
    """Instances of this class allow one to sample a system using the
    Wang-Landau (WL) algorithm, see Phys. Rev. Lett. **86**, 2050
    (2001) [WanLan01]_. The WL algorithm enables one to acquire the
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
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    calculator : :class:`BaseCalculator <mchammer.calculators.ClusterExpansionCalculator>`
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
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
        convergence very tedious if not possible
    energy_limit_left : float
    energy_limit_right : float
        defines the lower and upper limit of the energy range within which the
        microcanonical entropy :math:`S(E)` will be sampled. By default
        (`None`) no limit is imposed. Setting limits can be useful if only a
        part of the density of states is required. Usually these parameters
        are, however, not set directly but set internally if the energy axis
        is sampled in bins (see :func:`run_binned_wang_landau_simulation`).
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
    simulation.  For the purpose of demonstration, the parameters of
    the cluster expansion are set to obtain a two-dimensional square
    Ising model, one of the systems studied in the original work by
    Wang and Landau::

        from ase.build import bulk
        from icet import ClusterExpansion, ClusterSpace
        from mchammer.calculators import ClusterExpansionCalculator
        from mchammer.ensembles import WangLandauEnsemble

        # prepare cluster expansion
        prim = bulk('Au', crystalstructure='sc', a=1.0)
        prim.set_cell([1, 1, 10])
        cs = ClusterSpace(prim, cutoffs=[1.1], chemical_symbols=['Ag', 'Au'])
        ce = ClusterExpansion(cs, [0, 0, 1])

        # prepare initial configuration
        structure = prim.repeat((4, 4, 1))
        for k in range(8):
            structure[k].symbol = 'Ag'

        # set up and run MC simulation
        calculator = ClusterExpansionCalculator(structure, ce)
        mc = WangLandauEnsemble(structure=structure,
                                calculator=calculator,
                                energy_spacing=1,
                                data_container='ising_2d_run.dc')
        mc.run(number_of_trial_steps=len(structure)*10000)  # carry out 10,000,000 MC sweeps

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
            raise ValueError('invalid value for trial_move: {}.'
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
            raise ValueError('invalid energy window: left boundary ({}, {}) must be'
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
        # as these runs can require restarts as well as parameter adjustments
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
            data_container=data_container,
            random_seed=random_seed,
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
        self._converged = False
        if not hasattr(self, '_fill_factor'):
            self._fill_factor = 1
        if not hasattr(self, '_histogram'):
            self._histogram = {}
        if not hasattr(self, '_entropy'):
            self._entropy = {}
        self._fraction_flat_histogram = None
        self._potential = self.calculator.calculate_total(
            occupations=self.configuration.occupations)

    def _restart_ensemble(self):
        """Restarts ensemble using the last state saved in the data container
        file. Note that this method does _not_ use the last_state property of
        the data container but rather uses the last data written the data frame.
        """
        super()._restart_ensemble()
        df = self.data_container.data
        index = df[df['entropy'].notna()].index[-1]
        last_saved = df.iloc[index]
        self._converged = last_saved.converged
        self._fill_factor = last_saved.fill_factor
        self._histogram = last_saved.histogram
        self._entropy = last_saved.entropy

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
                self._entropy = {k: self._entropy[k]
                                 for k in self._entropy if self._inside_energy_window(k)}
                self._histogram = {k: self._histogram[k]
                                   for k in self._histogram if self._inside_energy_window(k)}

        # update histograms and entropy counters
        self._accumulate_entropy(bin_cur)

        return accept

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

    def _accumulate_entropy(self, bin_cur: int) -> None:
        """Updates counters for histogram and entropy, checks histogram
        flatness, and updates fill factor if indicated.
        """

        # update histogram and entropy
        self._entropy[bin_cur] = self._entropy.get(bin_cur, 0) + self._fill_factor
        self._histogram[bin_cur] = self._histogram.get(bin_cur, 0) + 1

        # check flatness of histogram
        if self.step % self._flatness_check_interval == 0 and \
                self.step > 0 and self._reached_energy_window:

            histogram = np.array(list(self._histogram.values()))
            limit = self._flatness_limit * np.average(histogram)
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
                    self._fraction_flat_histogram = None
                    # shift entropy counter in order to avoid overflow
                    entropy_ref = np.min(list(self._entropy.values()))
                    for k in self._entropy:
                        self._entropy[k] -= entropy_ref

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        sublattice_index = self.get_random_sublattice_index(self._sublattice_probabilities)
        return self.do_move(sublattice_index=sublattice_index)

    def _do_swap(self, sublattice_index: int, allowed_species: List[int] = None) -> int:
        """Carries out a Monte Carlo trial that involves swapping the species
        on two sites. This method has been adapted from
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
            raise ValueError('No canonical swaps are possible on any of the active sublattices.')
        sublattice_probabilities = [p / norm for p in sublattice_probabilities]
        return sublattice_probabilities

    def _get_flip_sublattice_probabilities(self) -> List[float]:
        """Returns the default sublattice probability which is based on the
        sizes of a sublattice. This method has been copied without
        modification from ThermodynamicBaseEnsemble.
        """
        sublattice_probabilities = []
        for i, sl in enumerate(self.sublattices):
            if len(sl.chemical_symbols) > 1:
                sublattice_probabilities.append(len(sl.indices))
            else:
                sublattice_probabilities.append(0)
        norm = sum(sublattice_probabilities)
        sublattice_probabilities = [p / norm for p in sublattice_probabilities]
        return sublattice_probabilities

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
        self._data_container.append(mctrial=self.step, record=row_dict)

    def _finalize(self) -> None:
        """This method is called from the run method after the conclusion of
        the MC cycle but before the data container is written. Here
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


def __get_entropy_from_data_container(dc: DataContainer,
                                      iteration: int = -1) -> DataFrame:
    """Returns the (relative) entropy from a _single_ data container
    produced during a Wang-Landau simulation.

    Parameters
    ----------
    dc
        data container, from which to extract the density of states
    iteration
        iteration of Wang-Landau algorithm, from which to use the
        microcanonical entropy; by default the last iteration is used

    Raises
    ------
    ValueError
        if ``dc`` does not contain required data from Wang-Landau
        simulation
    """

    # preparations
    if 'entropy' not in dc.data.columns:
        raise ValueError('Data container must contain "entropy" column.')

    # collect entropy
    energy_spacing = dc.ensemble_parameters['energy_spacing']
    last_index = dc.data[dc.data['entropy'].notna()].index[iteration]
    data = OrderedDict(sorted(dc.data.iloc[last_index]['entropy'].items()))
    df = DataFrame(data={'energy': energy_spacing * np.array(list(data.keys())),
                         'entropy': np.array(list(data.values()))},
                   index=list(data.keys()))
    # shift entropy for numerical stability
    df.entropy -= np.min(df.entropy)

    return df


def get_density_wang_landau(dcs: Union[DataContainer, dict],
                            iteration: int = -1) -> Tuple[DataFrame, dict]:
    """Returns a DataFrame with the total density of states from a Wang-Landau
    simulation. The second element of the tuple is a dictionary that contains the
    standard deviation between the entropy of neighboring data containers in the
    overlap region. These errors should be small compared to the variation of the
    entropy across each energy bin.

    The function can handle both a single data container
    and a list thereof. In the latter case the data containers must cover a
    contiguous energy range and must at least partially overlap.

    Parameters
    ----------
    dcs
        data container(s), from which to extract the density of states
    iteration
        iteration of Wang-Landau algorithm, from which to use the
        microcanonical entropy; by default the last iteration is used

    Raises
    ------
    ValueError
        if multiple data containers are provided and there are inconsistencies
        with regard to basic simulation parameters such as system size or
        energy spacing
    ValueError
        if multiple data containers are provided and there is at least
        one energy region without overlap
    """

    # preparations
    if isinstance(dcs, DataContainer):
        # fetch raw entropy data from data container
        df = __get_entropy_from_data_container(dcs, iteration)
        errors = None

    elif isinstance(dcs, dict) and isinstance(dcs[next(iter(dcs))], DataContainer):
        # minimal consistency checks
        tags = list(dcs.keys())
        tagref = tags[0]
        dcref = dcs[tagref]
        for tag in tags:
            dc = dcs[tag]
            if len(dc.structure) != len(dcref.structure):
                raise ValueError('Number of atoms differs between data containers ({}: {}, {}: {})'
                                 .format(tagref, dcref.ensemble_parameters['n_atoms'],
                                         tag, dc.ensemble_parameters['n_atoms']))
            for param in ['energy_spacing', 'trial_move']:
                if dc.ensemble_parameters[param] != dcref.ensemble_parameters[param]:
                    raise ValueError('{} differs between data containers ({}: {}, {}: {})'
                                     .format(param,
                                             tagref, dcref.ensemble_parameters['n_atoms'],
                                             tag, dc.ensemble_parameters['n_atoms']))

        # fetch raw entropy data from data containers
        entropies = {}
        for tag, dc in dcs.items():
            entropies[tag] = __get_entropy_from_data_container(dc, iteration)

        # sort entropies by energy
        entropies = OrderedDict(sorted(entropies.items(), key=lambda row: row[1].energy.iloc[0]))

        # line up entropy data
        errors = {}
        tags = list(entropies.keys())
        for tag1, tag2 in zip(tags[:-1], tags[1:]):
            df1 = entropies[tag1]
            df2 = entropies[tag2]
            left_lim = np.min(df2.energy)
            right_lim = np.max(df1.energy)
            if left_lim >= right_lim:
                raise ValueError('No overlap in the energy range {}...{}.\n'
                                 .format(right_lim, left_lim) +
                                 ' The closest data containers have tags "{}" and "{}".'
                                 .format(tag1, tag2))
            df1_ = df1[(df1.energy >= left_lim) & (df1.energy <= right_lim)]
            df2_ = df2[(df2.energy >= left_lim) & (df2.energy <= right_lim)]
            offset = np.average(df2_.entropy - df1_.entropy)
            errors['{}-{}'.format(tag1, tag2)] = np.std(df2_.entropy - df1_.entropy)
            entropies[tag2].entropy = entropies[tag2].entropy - offset

        # compile entropy over the entire energy range
        data = {}
        indices = {}
        counts = Counter()
        for df in entropies.values():
            for index, en, ent in zip(df.index, df.energy, df.entropy):
                data[en] = data.get(en, 0) + ent
                counts[en] += 1
                indices[en] = index
        for en in data:
            data[en] = data[en] / counts[en]

        # center entropy to prevent possible numerical issues
        entmin = np.min(list(data.values()))
        df = DataFrame(data={'energy': np.array(list(data.keys())),
                             'entropy': np.array(np.array(list(data.values()))) - entmin},
                       index=list(indices.values()))
    else:
        raise TypeError('dcs ({}) must be either a DataContainer or a list of DataContainer objects'
                        .format(type(dcs)))

    # density of states
    df['density'] = np.exp(df.entropy) / np.sum(np.exp(df.entropy))

    return df, errors


def get_averages_wang_landau(dcs: Union[DataContainer, dict],
                             temperatures: List[float],
                             observables: List[str] = None,
                             boltzmann_constant: float = kB,
                             iteration: int = -1) -> DataFrame:
    """Returns the average and the standard deviation of the energy for the
    temperatures specified. If the `observables` keyword argument is specified
    the function will also return the mean and standard deviation of the
    specified observables.

    Parameters
    ----------
    dcs
        data container(s), from which to extract density of states
        as well as observables
    temperatures
        temperatures, at which to compute the averages
    observables
        observables, for which to compute averages; the observables
        must refer to fields in the data container
    boltzmann_constant
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e. units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K]
    iteration
        iteration of Wang-Landau algorithm, from which to use the
        microcanonical entropy; by default the last iteration is used

    Raises
    ------
    ValueError
        if the data container(s) do(es) not contain entropy data
        from Wang-Landau simulation
    ValueError
        if data container(s) do(es) not contain requested observable
    """

    def check_observables(dc: DataContainer, observables: List[str]) -> None:
        """ Helper function that checks that observables are available in data frame. """
        if observables is None:
            return
        for obs in observables:
            if obs not in dc.data.columns:
                raise ValueError('Observable ({}) not in data container.\n'
                                 'Available observables: {}'.format(observables, dc.data.columns))

    # preparation of observables
    columns_to_keep = ['potential', 'density']
    if observables is not None:
        columns_to_keep.extend(observables)

    # fetch entropy and density of states from data container(s)
    df_density, _ = get_density_wang_landau(dcs, iteration)

    # check that observables are available in data container
    # and prepare comprehensive data frame with relevant information
    if isinstance(dcs, DataContainer):
        check_observables(dcs, observables)
        df_combined = dcs.data.filter(columns_to_keep)
        dcref = dcs
    elif isinstance(dcs, dict):
        for dc in dcs.values():
            check_observables(dc, observables)
        df_combined = pd_concat([dc.data for dc in dcs.values()],
                                ignore_index=True).filter(columns_to_keep)
        dcref = dc

    # compute density for each row in data container if observable averages
    # are to be computed
    if observables is not None:
        energy_spacing = dcref.ensemble_parameters['energy_spacing']
        # NOTE: we rely on the indices of the df_density DataFrame to
        # correspond to the energy scale! This is expected to be handled in
        # the get_density_wang_landau function.
        data_density = list(df_density.density[
            np.array(np.round(df_combined.potential / energy_spacing), dtype=int)])

    enref = np.min(df_density.energy)
    averages = []
    n_atoms = dcref.ensemble_parameters['n_atoms']
    for temperature in temperatures:

        # mean and standard deviation of energy
        boltz = np.exp(- (df_density.energy - enref) / temperature / boltzmann_constant)
        sumint = np.sum(df_density.density * boltz)
        en_mean = np.sum(df_density.energy / n_atoms * df_density.density * boltz) / sumint
        en_std = np.sum((df_density.energy / n_atoms) ** 2 * df_density.density * boltz) / sumint
        en_std = np.sqrt(en_std - en_mean ** 2)
        record = {'temperature': temperature,
                  'potential_mean': en_mean,
                  'potential_std': en_std}

        # mean and standard deviation of other observables
        if observables is not None:
            boltz = np.exp(- (df_combined.potential - enref) / temperature / boltzmann_constant)
            sumint = np.sum(data_density * boltz)
            for obs in observables:
                obs_mean = np.sum(data_density * boltz * df_combined[obs]) / sumint
                obs_std = np.sum(data_density * boltz * df_combined[obs] ** 2) / sumint
                obs_std = np.sqrt(obs_std - obs_mean ** 2)
                record['{}_mean'.format(obs)] = obs_mean
                record['{}_std'.format(obs)] = obs_std

        averages.append(record)

    return DataFrame.from_dict(averages)


def run_binned_wang_landau_simulation(structure: Atoms,
                                      calculator: BaseCalculator,
                                      energy_spacing: float,
                                      n_patches: int,
                                      minimum_energy: float,
                                      maximum_energy: float,
                                      n_steps: int,
                                      n_processes: int,
                                      patch_selection: List[int] = None,
                                      bin_size_exponent: float = 0.5,
                                      data_container_template: str = '',
                                      overlap: float = 4,
                                      trial_move: str = 'swap',
                                      fill_factor_limit: float = 1e-6,
                                      flatness_check_interval: int = None,
                                      flatness_limit: float = 0.8,
                                      random_seed: int = None,
                                      data_container_write_period: float = np.inf,
                                      ensemble_data_write_interval: int = None,
                                      trajectory_write_interval: int = None,
                                      sublattice_probabilities: List[float] = None,
                                      observers: List[BaseObserver] = None) -> None:
    """Runs a series of Wang-Landau simulations that each cover a
    different energy range. Splitting the sampling of the energy range
    into multiple segments has two crucial advantages:

    1. Since the variation of the density across a segment is much
       less extreme than over the entire range, simulations converge
       faster.
    2. Since the different segments can be run independently, they can
       be trivially parallelized.

    After the individual simulations have been completed the entire density
    must be reconstructed by patching the energy segments together. This is
    automatically handled by the :func:`get_density_wang_landau` and
    :func:`get_averages_wang_landau` functions.

    This function internally calls the :class:`WangLandauEnsemble` and
    many parameters of this function are simply forwarded to the
    class. For documentation concerning these parameters please
    consult the :class:`WangLandauEnsemble` class. Here, only the
    mandatory parameters and function specific arguments are
    described.

    Parameters
    ----------
    structure
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    calculator
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
    energy_spacing
        defines the bin size of the energy grid on which the microcanonical
        entropy :math:`S(E)`, and thus the density :math:`\\exp S(E)`, is
        evaluated; the spacing should be small enough to capture the features
        of the density of states; too small values will, however, render the
        convergence very tedious if not possible
    n_patches
        number of segments into which the energy axis is divided
    patch_selection
        list of bin indices to be run (or restarted). By default all bins are
        run. This argument allows one to run only selected bins. This is
        useful when restarting only some of the runs, e.g., since they did not
        converge during the initially alloted number of MC steps or because
        the convergence conditions should be tightened.
    bin_size_exponent
        controls the distribution of bin sizes along the energy axis. The
        outermost regions of the entropy usually exhibit the largest
        variations and therefore often require the longest simulations. By
        decreasing their bin size the variation across the bin is reduced and
        convergence should be faster. The `bin_size_exponent` parameter can
        thus be used to balance the computational load between different
        processes. `bin_size_exponent=1` yields a constant bin size; for
        `bin_size_exponent<1` the bins closer to the borders of the energy
        range are smaller than those in the center. Values larger than one are
        usually not recommended.
    minimum_energy
        estimate for the lower bound of the energy range that will be
        encountered
    maximum_energy
        estimate for the upper bound of the energy range that will be
        encountered
    n_steps
        number of MC trial steps to run in total
    data_container_template
        template for the file the data container will be written to;
        internally the bin index as well as the file ending ".dc" will
        be appended; if the file exists it will be read, the data
        container will be appended, and the file will be
        updated/overwritten
    n_processes
        number of processes that can be run in parallel; typically
        this is the number of available cores
    overlap
        number of bins, by which two neighboring segments should
        overlap; this value should not be too small; larger values
        imply larger overlap between segments and less efficient load
        distribution
    observers
        list of observers to be attached to the MC simulation

    Raises
    ------
    ValueError
        if n_patches is too small
    TypeError
        if patch_selection is not a list
    ValueError
        if the energy window (`maximum_energy - minimum_energy`) is too small
        to accommodate the specified number of patches
    """

    if n_patches < 2:
        raise ValueError('n_patches ({}) must be at least 2'.format(n_patches))
    if patch_selection is not None and not isinstance(patch_selection, list):
        raise TypeError('patch_selection ({}) must be given in the form of a list'
                        .format(patch_selection))

    # set bin boundaries
    limits = np.linspace(-1, 1, n_patches + 1)
    limits = np.sign(limits) * np.abs(limits) ** bin_size_exponent
    limits *= maximum_energy - minimum_energy
    limits += 0.5 * (maximum_energy + minimum_energy)
    limits[0], limits[-1] = None, None

    # Below, the calculator is made available within the scope of the entire
    # module in order for the multiprocessing module to work. If the
    # calculator is handed over to __run_simulation as an argument, an error
    # is raised that states that the calculator class cannot be pickled.
    global __calculator
    __calculator = calculator

    # set up MC simulations
    args = []
    for k, (energy_limit_left, energy_limit_right) in enumerate(zip(limits[:-1], limits[1:])):
        if energy_limit_left is not None and energy_limit_right is not None:
            if (maximum_energy - minimum_energy) / energy_spacing < 2 * overlap:
                raise ValueError('Energy window too small. min/max: {}/{}'
                                 .format(minimum_energy, maximum_energy) +
                                 ' Try decreasing n_patches ({}) and/or overlap ({}).'
                                 .format(n_patches, overlap))
        if energy_limit_left is not None:
            energy_limit_left -= overlap * energy_spacing
        if energy_limit_right is not None:
            energy_limit_right += overlap * energy_spacing
        data_container = '{}k{}.dc'.format(data_container_template, k)
        args.append({'structure': structure,
                     'n_steps': n_steps,
                     'energy_spacing': energy_spacing,
                     'energy_limit_left': energy_limit_left,
                     'energy_limit_right': energy_limit_right,
                     'trial_move': trial_move,
                     'fill_factor_limit': fill_factor_limit,
                     'flatness_limit': flatness_limit,
                     'flatness_check_interval': flatness_check_interval,
                     'data_container': data_container,
                     'random_seed': random_seed,
                     'data_container_write_period': data_container_write_period,
                     'ensemble_data_write_interval': ensemble_data_write_interval,
                     'trajectory_write_interval': trajectory_write_interval,
                     'sublattice_probabilities': sublattice_probabilities,
                     'observers': observers})

    # run MC simulations
    pool = Pool(processes=n_processes)
    if args is None:
        pool.map(__run_simulation, args)
    else:
        if np.any(np.array(patch_selection) < 0) or np.any(np.array(patch_selection) > len(args)):
            raise ValueError('Invalid patch index in patch_selection ({});'
                             ' allowed values: 0 ... {}.'
                             .format(patch_selection, len(args)))
        pool.map(__run_simulation, [a for k, a in enumerate(args) if k in patch_selection])


def __run_simulation(args: dict) -> None:
    """This function is used run_binned_wang_landau_simulation in order to
    launch multiple different bins via the multiprocessing module. It has to
    be at the module level in order for it to be pickled. """
    mc = WangLandauEnsemble(structure=args['structure'],
                            calculator=__calculator,
                            energy_spacing=args['energy_spacing'],
                            energy_limit_left=args['energy_limit_left'],
                            energy_limit_right=args['energy_limit_right'],
                            trial_move=args['trial_move'],
                            fill_factor_limit=args['fill_factor_limit'],
                            flatness_limit=args['flatness_limit'],
                            flatness_check_interval=args['flatness_check_interval'],
                            data_container=args['data_container'],
                            random_seed=args['random_seed'],
                            data_container_write_period=args['data_container_write_period'],
                            ensemble_data_write_interval=args['ensemble_data_write_interval'],
                            trajectory_write_interval=args['trajectory_write_interval'],
                            sublattice_probabilities=args['sublattice_probabilities'])
    for obs in args['observers']:
        mc.attach_observer(obs)
    mc.run(number_of_trial_steps=args['n_steps'])
