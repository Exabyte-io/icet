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

          P = \\min \\{ 1, \\, \\exp [ S(E_\mathrm{new}) - S(E_\mathrm{cur}) ] \\},

       where :math:`E_\mathrm{cur}` and :math:`E_\mathrm{new}` are the
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

       \\left<E\\right> = \\frac{\\sum_E \\rho(E) \\exp(-E / k_B T)}{\\sum_E \\rho(E) \\exp(-E / k_B T)}

    Parameters
    ----------
    structure : :class:`Atoms <ase.Atoms>`
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    calculator : :class:`BaseCalculator <mchammer.calculators.ClusterExpansionCalculator>`
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
    trial_move
        One can choose between two different trial moves for
        generating new configurations. In a 'swap' move two sites are
        selected and their occupations are swapped; in a 'flip' move
        one site is selected and its occupation is flipped to a
        different species. While 'swap' moves conserve the
        concentrations of the species in the system, 'flip' moves
        allow one in principle to sample the full composition space.
    energy_spacing
        defines the bin size of the energy grid on which the
        microcanonical entropy :math:`S(E)`, and thus the density
        :math:`\exp S(E)`, is evaluated
    fill_factor_limit
        If the fill_factor :math:`f` falls below this value, the
        algorithm is terminated.
    flatness_check_interval
        For computational efficiency the flatness condition is only
        evaluated every ``flatness_check_interval``-th trial step. By
        default (``None``) ``flatness_check_interval`` is set to 1000
        times the number of sites in ``structure``, i.e. 1000 Monte
        Carlo sweeps.
    flatness_limit
        The histogram :math:`H(E)` is deemed sufficiently flat if
        :math:`H(E) > \\chi \\left<H(E)\\right>\\,\\forall
        E`. ``flatness_limit`` is the parameter :math:`\\chi`.
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
                 trial_move: str = 'swap',
                 fill_factor_limit: float = 1e-8,
                 flatness_check_interval: int = None,
                 flatness_limit: float = 0.8,
                 user_tag: str = None,
                 data_container: DataContainer = None,
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

        # ensemble parameters
        self._energy_spacing = energy_spacing
        self._fill_factor_limit = fill_factor_limit
        self._flatness_check_interval = flatness_check_interval
        self._flatness_limit = flatness_limit

        self._ensemble_parameters = {}
        self._ensemble_parameters['energy_spacing'] = energy_spacing
        self._ensemble_parameters['trial_move'] = trial_move
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
        if not hasattr(self, '_converged'):
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
        self._entropy = last_saved.entropy

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

        # check flatness of histogram
        if self.step % self._flatness_check_interval == 0 and self.step > 0:

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
