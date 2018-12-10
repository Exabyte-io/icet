"""
Definition of the variance-constrained semi-grand canonical ensemble class.
"""

import numpy as np

from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.units import kB
from typing import Dict, Union

from .. import DataContainer
from .base_ensemble import BaseEnsemble
from ..calculators.base_calculator import BaseCalculator


class VCSGCEnsemble(BaseEnsemble):
    """Instances of this class allow one to simulate systems in the
    variance-constrained semi-grand canonical (VCSGC) ensemble
    (:math:`N\\phi\\kappa VT`), i.e. at constant temperature (:math:`T`), total
    number of sites (:math:`N=\\sum_i N_i`), and two additional dimensionless
    parameters :math:`\\phi` and :math:`\\kappa`, which constrain average and
    variance of the concentration, respectively. The VCSGC ensemble is
    currently only implemented for binary systems.

    The probability for a particular state in the VCSGC ensemble for a
    :math:`2`-component system can be written

    .. math::

        \\rho_{\\text{VCSGC}} \\propto \\exp\\Big[ - E / k_B T
        + \\kappa N ( c_1 + \\phi_1 / 2 )^2 \\Big],

    where :math:`c_1` represents the concentration of species 1, i.e.
    :math:`c_1=N_1/N`. (Please note that the quantities :math:`\\kappa` and
    :math:`\\phi` correspond, respectively, to :math:`\\bar{\\kappa}` and
    :math:`\\bar{\\phi}` in [SadErh12]_.) This implementation requires
    :math:`\\phi` to be specified for both species. The sum of the specified
    :math:`\\phi` values is required to be :math:`-2`, because then the above
    expression is symmetric with respect to interchange of species 1 and 2,
    i.e., it does not matter if we use :math:`\\phi_1` and :math:`c_1` or
    :math:`\\phi_2` and :math:`c_2`.

    Just like the :ref:`semi-grand canonical ensemble <canonical_ensemble>`,
    the VCSGC ensemble allows concentrations to change. A trial step consists
    of changing the identity of a randomly chosen atom and accepting the change
    with probability

    .. math::

        P = \\min \\{ 1, \\, \\exp [ - \\Delta E / k_B T
        + \\kappa N \\Delta c_1 (\\phi_1 + \\Delta c_1 + 2 c_1 ) ] \\}.

    Note that for a sufficiently large value of :math:`\\kappa`, say 200, the
    probability density :math:`\\rho_{\\text{VCSGC}}` is sharply peaked around
    :math:`c_1=-\\phi_1 / 2`. In practice, this means that we can gradually
    change :math:`\\phi_1` from (using some margins) :math:`-2.1` to
    :math:`0.1` and take the system continuously from :math:`c_1 = 0` to
    :math:`1`. The parameter :math:`\\kappa` constrains the fluctuations (or
    the variance) of the concentration at each value of :math:`\\phi_1`, with
    higher values of :math:`\\kappa` meaning less fluctuations. Unlike the
    :ref:`semi-grand canonical ensemble <vcsgc_ensemble>`, one value of
    :math:`\\phi_1` maps to one and only one concentration also in multiphase
    regions. Since the derivative of the canonical free energy can be expressed
    in terms of parameters and observables of the VCSGC ensemble,

    .. math::

        k_B T \\kappa ( \\phi_1 + 2 \\langle c_1 \\rangle ) = - \\frac{1}{N}
        \\frac{\\partial F}{\\partial c_1} (N, V, T, \\langle c_1 \\rangle ),

    this ensemble allows for thermodynamic integration across multiphase
    regions. This means that we can construct phase diagrams by directly
    comparing the free energies of the different phases. This often makes the
    VCSGC ensemble more convenient than the :ref:`semi-grand canonical ensemble
    <sgc_ensemble>` when simulating materials with multiphase regions, such as
    alloys with miscibility gaps.

    When using the VCSGC ensemble, please cite Sadigh, B. and Erhart, P., Phys.
    Rev. B **86**, 134204 (2012) [SadErh12]_.

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
    phis : Dict[str, float]
        average constraint parameters :math:`\\phi_i`; the key denotes the
        species; there must be one entry for each species but their sum must be
        :math:`-2.0` (referred to as :math:`\\bar{\\phi}` in [SadErh12]_)
    kappa : float
        parameter that constrains the variance of the concentration
        (referred to as :math:`\\bar{\\kappa}` in [SadErh12]_)
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
                 temperature: float, phis: Dict[str, float],
                 kappa: float, boltzmann_constant: float = kB,
                 user_tag: str = None,
                 data_container: DataContainer = None,
                 random_seed: int = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None) -> None:

        self._ensemble_parameters = dict(temperature=temperature,
                                         kappa=kappa)
        self._set_phis(phis)
        for atnum, phi in self.phis.items():
            phi_sym = 'phi_{}'.format(chemical_symbols[atnum])
            self._ensemble_parameters[phi_sym] = phi

        self._boltzmann_constant = boltzmann_constant

        super().__init__(
            atoms=atoms, calculator=calculator, user_tag=user_tag,
            data_container=data_container,
            random_seed=random_seed,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

        if len(self.configuration._allowed_species) > 2:
            raise NotImplementedError('VCSGCEnsemble does not yet support '
                                      'cluster spaces with more than two '
                                      'species.')

        if set(self.configuration._allowed_species) != set(self.phis.keys()):
            raise ValueError('phis were not set for all species')

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        self._total_trials += 1

        # choose flip
        sublattice_index = self.get_random_sublattice_index()
        index, new_species = \
            self.configuration.get_flip_state(sublattice_index)
        old_species = self.configuration.occupations[index]

        # Calculate difference in VCSGC thermodynamic potential.
        # Note that this assumes that only one atom was flipped.
        N = len(self.atoms)
        occupations = self.configuration._occupations.tolist()
        potential_diff = 1.0  # dN
        potential_diff -= occupations.count(old_species)
        potential_diff -= 0.5 * N * self.phis[old_species]
        potential_diff += occupations.count(new_species)
        potential_diff += 0.5 * N * self.phis[new_species]
        potential_diff *= self.kappa
        potential_diff *= self.boltzmann_constant * self.temperature
        potential_diff /= N

        potential_diff += self._get_property_change([index], [new_species])

        if self._acceptance_condition(potential_diff):
            self._accepted_trials += 1
            self.update_occupations([index], [new_species])

    def _acceptance_condition(self, potential_diff: float) -> bool:
        """
        Evaluates Metropolis acceptance criterion.

        Parameters
        ----------
        potential_diff
            the change in the thermodynamic potential associated
            with the trial step
        """
        if potential_diff < 0:
            return True
        else:
            return np.exp(-potential_diff / (
                self.boltzmann_constant * self.temperature)) > \
                self._next_random_number()

    @property
    def temperature(self) -> float:
        """ temperature :math:`T` (see parameters section above) """
        return self.ensemble_parameters['temperature']

    @property
    def boltzmann_constant(self) -> float:
        """ Boltzmann constant :math:`k_B` (see parameters section above) """
        return self._boltzmann_constant

    @property
    def phis(self) -> Dict[int, float]:
        """
        phis :math:`\\phi_i`, one for each species but their sum must be
        :math:`-2.0` (referred to as :math:`\\bar{\\phi}` in [SadErh12]_)
        """
        return self._phis

    @property
    def kappa(self) -> float:
        """
        kappa :math:`\\bar{\\kappa}` constrain parameter
        (see parameters section above)
        """
        return self.ensemble_parameters['kappa']

    def _set_phis(self, phis: Dict[Union[int, str], float]):
        """ Sets values of phis."""
        if not isinstance(phis, dict):
            raise TypeError('phis has the wrong type: {}'.format(type(phis)))
        if abs(sum(phis.values()) + 2) > 1e-6:
            raise ValueError('The sum of all phis must equal to -2')

        self._phis = {}
        for key, phi in phis.items():
            if isinstance(key, str):
                atomic_number = atomic_numbers[key]
                self._phis[atomic_number] = phi
            elif isinstance(key, int):
                self._phis[key] = phi

    def _get_ensemble_data(self) -> Dict:
        """
        Returns a dict with the default data of the ensemble. This includes
        atom counts and free energy derivative.
        """
        data = super()._get_ensemble_data()

        # free energy derivative
        atnum_1 = min(self.phis.keys())
        concentration = self.configuration._occupations.tolist().count(
            atnum_1) / len(self.atoms)
        data['free_energy_derivative'] = self.kappa * \
            self.boltzmann_constant * self.temperature * \
            (- 2 * concentration - self.phis[atnum_1])

        # species counts
        atoms = self.configuration.atoms
        unique, counts = np.unique(atoms.numbers, return_counts=True)
        # TODO: avoid accessing a protected member of a client class
        for atnum in self.configuration._allowed_species:
            data['{}_count'.format(chemical_symbols[atnum])] = 0
        for atnum, count in zip(unique, counts):
            data['{}_count'.format(chemical_symbols[atnum])] = count

        return data
