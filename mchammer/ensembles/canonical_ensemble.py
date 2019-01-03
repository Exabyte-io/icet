"""Definition of the canonical ensemble class."""

import numpy as np

from ase import Atoms
from ase.units import kB

from .. import DataContainer
from .base_ensemble import BaseEnsemble
from ..calculators.base_calculator import BaseCalculator


class CanonicalEnsemble(BaseEnsemble):
    """Instances of this class allow one to simulate systems in the
    canonical ensemble (:math:`N_iVT`), i.e. at constant temperature
    (:math:`T`), number of atoms of each species (:math:`N_i`), and
    volume (:math:`V`).

    The probability for a particular state in the canonical ensemble is
    proportional to the well-known Boltzmann factor,

    .. math::

        \\rho_{\\text{C}} \\propto \\exp [ - E / k_B T ].

    Since the concentrations or equivalently the number of atoms of each
    species is held fixed in the canonical ensemble, a trial step must
    conserve the concentrations. This is accomplished by randomly picking two
    unlike atoms and swapping their identities. The swap is accepted with
    probability

    .. math::

        P = \\min \\{ 1, \\, \\exp [ - \\Delta E / k_B T  ] \\},

    where :math:`\\Delta E` is the change in potential energy caused by the
    swap.

    The canonical ensemble provides an ideal framework for studying the
    properties of a system at a specific concentration. Properties such as
    potential energy or phenomena such as chemical ordering at a specific
    temperature can conveniently be studied by simulating at that temperature.
    The canonical ensemble is also a convenient tool for "optimizing" a
    system, i.e., finding its lowest energy chemical ordering. In practice,
    this is usually achieved by simulated annealing, i.e. the system is
    equilibrated at a high temperature, after which the temperature is
    continuously lowered until the acceptance probability is almost zero. In a
    well-behaved system, the chemical ordering at that point corresponds to a
    low-energy structure, possibly the global minimum at that particular
    concentration.

    Parameters
    ----------
    atoms : :class:`ase:Atoms`
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    calculator : :class:`BaseCalculator`
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
    temperature : float
        temperature :math:`T` in appropriate units [commonly Kelvin]
    boltzmann_constant : float
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e. units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K]
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
    """

    def __init__(self, atoms: Atoms, calculator: BaseCalculator,
                 temperature: float, user_tag: str = None,
                 boltzmann_constant: float = kB,
                 data_container: DataContainer = None, random_seed: int = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None) -> None:

        self._ensemble_parameters = dict(temperature=temperature)

        self._boltzmann_constant = boltzmann_constant

        super().__init__(
            atoms=atoms, calculator=calculator, user_tag=user_tag,
            data_container=data_container,
            random_seed=random_seed,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

    @property
    def temperature(self) -> float:
        """ temperature :math:`T` (see parameters section above) """
        return self.ensemble_parameters['temperature']

    @property
    def boltzmann_constant(self) -> float:
        """ Boltzmann constant :math:`k_B` (see parameters section above) """
        return self._boltzmann_constant

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        self._total_trials += 1

        sublattice_index = self.get_random_sublattice_index()
        sites, species = \
            self.configuration.get_swapped_state(sublattice_index)

        potential_diff = self._get_property_change(sites, species)

        if self._acceptance_condition(potential_diff):
            self._accepted_trials += 1
            self.update_occupations(sites, species)

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
        else:
            return np.exp(-potential_diff / (
                self.boltzmann_constant * self.temperature)) > \
                self._next_random_number()
