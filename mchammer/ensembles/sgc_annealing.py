"""Definition of the canonical annealing class."""

import numpy as np

from ase import Atoms
from ase.data import chemical_symbols
from ase.units import kB
from typing import Dict, List

from .. import DataContainer
from ..calculators.base_calculator import BaseCalculator
from .thermodynamic_base_ensemble import ThermodynamicBaseEnsemble
from .semi_grand_canonical_ensemble import get_chemical_potentials
from .canonical_annealing import available_cooling_functions


class SGCAnnealing(ThermodynamicBaseEnsemble):
    """Instances of this class allow one to carry out simulated annealing
    in the semi grand canonical ensemble, i.e. the temperature is varied in
    pre-defined fashion while the chemical potential is kept fixed.  See
    :class:`mchammer.ensembles.SemiGrandCanonicalEnsemble` for more information
    about the ensemble.

    The temperature control scheme is selected via the
    ``cooling_function`` keyword argument, while the initial and final
    temperature are set via the ``T_start`` and ``T_stop`` arguments.
    Several pre-defined temperature control schemes are available
    including `linear` and `exponential`. In the latter case the
    temperature varies logarithmatically as a function of the MC step,
    emulating the exponential temperature dependence of the atomic
    exchange rate encountered in many materials.  It is also possible
    to provide a user defined cooling function via the keyword
    argument.  This function must comply with the following function
    header::

        def cooling_function(step, T_start, T_stop, n_steps):
            T = ...  # compute temperature
            return T

    Here ``step`` refers to the current MC trial step.

    Parameters
    ----------
    structure : :class:`Atoms <ase.Atoms>`
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    chemical_potentials : Dict[str, float]
        chemical potential for each species :math:`\\mu_i`; the key
        denotes the species, the value specifies the chemical potential in
        units that are consistent with the underlying cluster expansion
    calculator : :class:`BaseCalculator <mchammer.calculators.ClusterExpansionCalculator>`
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
    T_start : float
        temperature from which the annealing is started
    T_stop : float
        final temperature for annealing
    n_steps : int
        number of steps to take in the annealing simulation
    cooling_function : str/function
        to use the predefined cooling functions provide a string
        `linear` or `exponential`, otherwise provide a function
    boltzmann_constant : float
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e. units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K]
    user_tag : str
        human-readable tag for ensemble [default: None]
    random_seed : int
        seed for the random number generator used in the Monte Carlo
        simulation
    data_container : str
        name of file the data container associated with the ensemble
        will be written to; if the file exists it will be read, the
        data container will be appended, and the file will be
        updated/overwritten
    data_container_write_period : float
        period in units of seconds at which the data container is
        written to file; writing periodically to file provides both
        a way to examine the progress of the simulation and to back up
        the data [default: np.inf]
    ensemble_data_write_interval : int
        interval at which data is written to the data container; this
        includes for example the current value of the calculator
        (i.e. usually the energy) as well as ensembles specific fields
        such as temperature or the number of atoms of different species
    trajectory_write_interval : int
        interval at which the current occupation vector of the atomic
        configuration is written to the data container.
    sublattice_probabilities : List[float]
        probability for picking a sublattice when doing a random swap.
        This should be as long as the number of sublattices and should
        sum up to 1.

    """

    def __init__(self,
                 structure: Atoms,
                 calculator: BaseCalculator,
                 T_start: float,
                 T_stop: float,
                 n_steps: int,
                 chemical_potentials: Dict[str, float],
                 cooling_function: str = 'exponential',
                 boltzmann_constant: float = kB,
                 user_tag: str = None,
                 random_seed: int = None,
                 data_container: str = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None,
                 sublattice_probabilities: List[float] = None) -> None:

        self._ensemble_parameters = dict(n_steps=n_steps)

        self._chemical_potentials = get_chemical_potentials(chemical_potentials)

        # add chemical potentials to ensemble parameters
        self._chemical_potentials = get_chemical_potentials(chemical_potentials)
        for atnum, chempot in self.chemical_potentials.items():
            mu_sym = 'mu_{}'.format(chemical_symbols[atnum])
            self._ensemble_parameters[mu_sym] = chempot

        super().__init__(
            structure=structure,
            calculator=calculator,
            user_tag=user_tag,
            random_seed=random_seed,
            data_container=data_container,
            data_container_class=DataContainer,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval,
            boltzmann_constant=boltzmann_constant)

        self._temperature = T_start
        self._T_start = T_start
        self._T_stop = T_stop
        self._n_steps = n_steps

        self._ground_state_candidate = self.configuration.structure
        self._ground_state_candidate_potential = self.calculator.calculate_total(
            occupations=self.configuration.occupations)

        # setup cooling function
        if isinstance(cooling_function, str):
            available = sorted(available_cooling_functions.keys())
            if cooling_function not in available:
                raise ValueError(
                    'Select from the available cooling_functions {}'.format(available))
            self._cooling_function = available_cooling_functions[cooling_function]
        elif callable(cooling_function):
            self._cooling_function = cooling_function
        else:
            raise TypeError('cooling_function must be either str or a function')

        if sublattice_probabilities is None:
            self._flip_sublattice_probabilities = self._get_flip_sublattice_probabilities()
        else:
            self._flip_sublattice_probabilities = sublattice_probabilities

    @property
    def chemical_potentials(self) -> Dict[int, float]:
        """
        chemical potentials :math:`\\mu_i` (see parameters section above)
        """
        return self._chemical_potentials

    @property
    def temperature(self) -> float:
        """ Current temperature """
        return self._temperature

    @property
    def estimated_ground_state(self):
        """ Structure with lowest observed potential during run """
        return self._ground_state_candidate.copy()

    @property
    def estimated_ground_state_potential(self):
        """ Lowest observed potential during run """
        return self._ground_state_candidate_potential

    def run(self):
        """ Runs the annealing. """
        if self.step >= self._n_steps:
            raise Exception('Annealing has already finished')
        super().run(self._n_steps - self.step)

    def _do_trial_step(self) -> int:
        """ Carries out one Monte Carlo trial step. """
        self._temperature = self._cooling_function(
            self.step, self._T_start, self._T_stop, self._n_steps)
        sublattice_index = self.get_random_sublattice_index(self._flip_sublattice_probabilities)
        return self.do_sgc_flip(
            sublattice_index=sublattice_index, chemical_potentials=self.chemical_potentials)

    def _get_ensemble_data(self) -> Dict:
        """Returns the data associated with the ensemble. For the
        CanonicalAnnealing this specifically includes the temperature.
        """
        data = super()._get_ensemble_data()

        # species counts
        data.update(self._get_species_counts())

        data['temperature'] = self.temperature
        if data['potential'] < self._ground_state_candidate_potential:
            self._ground_state_candidate_potential = data['potential']
            self._ground_state_candidate = self.configuration.structure
        return data
