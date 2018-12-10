"""
Definition of the semi-grand canonical ensemble class.
"""

import numpy as np

from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.units import kB
from collections import OrderedDict
from typing import Dict, Union

from .. import DataContainer
from .base_ensemble import BaseEnsemble
from ..calculators.base_calculator import BaseCalculator


class SemiGrandCanonicalEnsemble(BaseEnsemble):
    """Instances of this class allow one to simulate systems in the
    semi-grand canonical (SGC) ensemble (:math:`N\\Delta\\mu_i VT`), i.e. at
    constant temperature (:math:`T`), total number of sites (:math:`N=\\sum_i
    N_i`), relative chemical potentials (:math:`\\Delta\\mu_i=\\mu_i - \\mu_1`,
    where :math:`i` denotes the species), and volume (:math:`V`).

    The probability for a particular state in the SGC ensemble for a
    :math:`m`-component system can be written

    .. math::

        \\rho_{\\text{SGC}} \\propto \\exp\\Big[ - \\big( E
        + \\sum_{i>1}^m \\Delta\\mu_i N_i \\big) \\big / k_B T \\Big]

    with the *relative* chemical potentials :math:`\\Delta\\mu_i = \\mu_i -
    \\mu_1` and species counts :math:`N_i`. Unlike the :ref:`canonical ensemble
    <canonical_ensemble>`, the number of the respective species (or,
    equivalently, the concentrations) are allowed to vary in the SGC ensemble.
    A trial step thus consists of randomly picking an atom and changing its
    identity with probability

    .. math::

        P = \\min \\Big\\{ 1, \\, \\exp \\big[ - \\big( \\Delta E
        + \\sum_i \\Delta \\mu_i \\Delta N_i \\big) \\big / k_B T \\big]
        \\Big\\},

    where :math:`\\Delta E` is the change in potential energy caused by the
    swap.

    There exists a simple relation between the differences in chemical
    potential and the canonical free energy :math:`F`. In a binary system, this
    relationship reads

    .. math:: \\Delta \\mu = - \\frac{1}{N} \\frac{\\partial F}{\\partial c} (
        N, V, T, \\langle c \\rangle).

    Here :math:`c` denotes concentration (:math:`c=N_i/N`) and :math:`\\langle
    c \\rangle` the average concentration observed in the simulation. By
    recording :math:`\\langle c \\rangle` while gradually changing
    :math:`\\Delta \\mu`, one can thus in principle calculate the difference in
    canonical free energy between the pure phases (:math:`c=0` or :math:`1`)
    and any concentration by integrating :math:`\\Delta \\mu` over that
    concentration range. In practice this requires that the average recorded
    concentration :math:`\\langle c \\rangle` varies continuously with
    :math:`\\Delta \\mu`. This is not the case for materials with multiphase
    regions (such as miscibility gaps), because in such regions :math:`\\Delta
    \\mu` maps to multiple concentrations. In a Monte Carlo simulation, this is
    typically manifested by discontinuous jumps in concentration. Such jumps
    mark the phase boundaries of a multiphase region and can thus be used to
    construct the phase diagram. To recover the free energy, however, such
    systems require sampling in other ensembles, such as the :ref:`variance-
    constrained semi-grand canonical ensemble <sgc_ensemble>`.

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
    chemical_potentials : Dict[str, float]
        chemical potential for each species :math:`\\mu_i`; the key
        denotes the species, the value specifies the chemical potential in
        units that are consistent with the underlying cluster expansion
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
                 temperature: float, chemical_potentials: Dict[str, float],
                 user_tag: str = None,
                 data_container: DataContainer = None, random_seed: int = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None,
                 boltzmann_constant: float = kB) -> None:

        self._ensemble_parameters = dict(temperature=temperature)

        self._chemical_potentials = None
        self._set_chemical_potentials(chemical_potentials)
        for atnum, chempot in self.chemical_potentials.items():
            mu_sym = 'mu_{}'.format(chemical_symbols[atnum])
            self._ensemble_parameters[mu_sym] = chempot

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

        # energy change
        sublattice_index = self.get_random_sublattice_index()
        index, species = \
            self.configuration.get_flip_state(sublattice_index)
        potential_diff = self._get_property_change([index], [species])

        # change in chemical potential
        old_species = self.configuration.occupations[index]
        chemical_potential_diff = \
            self.chemical_potentials[old_species] - \
            self.chemical_potentials[species]
        potential_diff += chemical_potential_diff

        if self._acceptance_condition(potential_diff):
            self._accepted_trials += 1
            self.update_occupations([index], [species])

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

    @property
    def chemical_potentials(self) -> Dict[int, float]:
        """
        chemical potentials :math:`\\mu_i` (see parameters section above)
        """
        return self._chemical_potentials

    def _set_chemical_potentials(self,
                                 chemical_potentials:
                                 Dict[Union[int, str], float]):
        """ Sets values of chemical potentials. """
        if not isinstance(chemical_potentials, dict):
            raise TypeError('chemical_potentials has the wrong type: {}'
                            .format(type(chemical_potentials)))

        cps = OrderedDict([(key, val) if isinstance(key, int)
                           else (atomic_numbers[key], val)
                           for key, val in chemical_potentials.items()])

        if self._chemical_potentials is None:
            # TODO: add check with respect to configuration_manager
            self._chemical_potentials = cps
        else:
            for num in cps:
                if num not in self._chemical_potentials:
                    raise ValueError(
                        'Unknown species {} in chemical_potentials'
                        .format(num))
            self._chemical_potentials.update(cps)

    def _get_ensemble_data(self) -> Dict:
        """Returns the data associated with the ensemble. For the SGC
        ensemble this specifically includes the species counts.
        """
        # generic data
        data = super()._get_ensemble_data()

        # species counts
        atoms = self.configuration.atoms
        unique, counts = np.unique(atoms.numbers, return_counts=True)
        # TODO: avoid accessing a protected member of a client class
        for atnum in self.configuration._allowed_species:
            data['{}_count'.format(chemical_symbols[atnum])] = 0
        for atnum, count in zip(unique, counts):
            data['{}_count'.format(chemical_symbols[atnum])] = count

        return data
