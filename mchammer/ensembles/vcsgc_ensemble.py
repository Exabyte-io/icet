"""
Definition of the variance-constrained semi-grand canonical ensemble class.
"""

from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.units import kB
from typing import Dict, Union, List

from .. import DataContainer
from ..calculators.base_calculator import BaseCalculator
from .thermodynamic_base_ensemble import ThermodynamicBaseEnsemble


class VCSGCEnsemble(ThermodynamicBaseEnsemble):
    r"""Instances of this class allow one to simulate systems in the
    variance-constrained semi-grand canonical (VCSGC) ensemble
    (:math:`N\phi\kappa VT`), i.e. at constant temperature (:math:`T`), total
    number of sites (:math:`N=\sum_i N_i`), and two additional dimensionless
    parameters :math:`\phi` and :math:`\kappa`, which constrain average and
    variance of the concentration, respectively.

    The below examples treat the binary case, but the generalization of
    to ternaries and higher-order systems is straight-forward.
    The probability for a particular state in the VCSGC ensemble for a
    :math:`2`-component system can be written

    .. math::

        \rho_{\text{VCSGC}} \propto \exp\Big[ - E / k_B T
        + \kappa N ( c_1 + \phi_1 / 2 )^2 \Big],

    where :math:`c_1` represents the concentration of species 1, i.e.,
    :math:`c_1=N_1/N`. (Please note that the quantities :math:`\kappa` and
    :math:`\phi` correspond, respectively, to :math:`\bar{\kappa}` and
    :math:`\bar{\phi}` in [SadErh12]_.) The :math:`\phi` may refer to any
    of the two species. If :math:`\phi` is specified for species A, an
    equivalent simulation can be carried out by specifying :math:`\phi_B` as
    :math:`-2-\phi_A`. In general, simulations of :math:`N`-component
    systems requires the specification of :math:`\phi` for :math:`N-1`
    elements.

    Just like the :ref:`semi-grand canonical ensemble <canonical_ensemble>`,
    the VCSGC ensemble allows concentrations to change. A trial step consists
    of changing the identity of a randomly chosen atom and accepting the change
    with probability

    .. math::

        P = \min \{ 1, \, \exp [ - \Delta E / k_B T
        + \kappa N \Delta c_1 (\phi_1 + \Delta c_1 + 2 c_1 ) ] \}.

    Note that for a sufficiently large value of :math:`\kappa`, say 200, the
    probability density :math:`\rho_{\text{VCSGC}}` is sharply peaked around
    :math:`c_1=-\phi_1 / 2`. In practice, this means that we can gradually
    change :math:`\phi_1` from (using some margins) :math:`-2.1` to
    :math:`0.1` and take the system continuously from :math:`c_1 = 0` to
    :math:`1`. The parameter :math:`\kappa` constrains the fluctuations (or
    the variance) of the concentration at each value of :math:`\phi_1`, with
    higher values of :math:`\kappa` meaning less fluctuations. Unlike the
    :ref:`semi-grand canonical ensemble <sgc_ensemble>`, one value of
    :math:`\phi_1` maps to one and only one concentration also in multiphase
    regions. Since the derivative of the canonical free energy can be expressed
    in terms of parameters and observables of the VCSGC ensemble,

    .. math::

        k_B T \kappa ( \phi_1 + 2 \langle c_1 \rangle ) = - \frac{1}{N}
        \frac{\partial F}{\partial c_1} (N, V, T, \langle c_1 \rangle ),

    this ensemble allows for thermodynamic integration across multiphase
    regions. This means that we can construct phase diagrams by directly
    comparing the free energies of the different phases. This often makes the
    VCSGC ensemble more convenient than the :ref:`semi-grand canonical ensemble
    <sgc_ensemble>` when simulating materials with multiphase regions, such as
    alloys with miscibility gaps.

    When using the VCSGC ensemble, please cite
    Phys. Rev. B **86**, 134204 (2012) [SadErh12]_.

    Parameters
    ----------
    structure
        Atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector.
    calculator
        Calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion.
    temperature
        Temperature :math:`T` in appropriate units, commonly Kelvin.
    phis
        Average constraint parameters :math:`\phi_i`. The key denotes the
        species. For a :math:`N`-component sublattice, there should be :math:`N - 1`
        different :math:`\phi_i` (referred to as :math:`\bar{\phi}`
        in [SadErh12]_).
    kappa
        parameter that constrains the variance of the concentration
        (referred to as :math:`\bar{\kappa}` in [SadErh12]_)
    boltzmann_constant
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e., units that are consistent
        with the underlying cluster expansion
        and the temperature units. Default: eV/K.
    user_tag
        Human-readable tag for ensemble. Default: ``None``.
    random_seed
        Seed for the random number generator used in the Monte Carlo simulation.
    dc_filename
        Name of file the data container associated with the ensemble
        will be written to. If the file exists it will be read, the
        data container will be appended, and the file will be
        updated/overwritten.
    data_container_write_period
        Period in units of seconds at which the data container is
        written to file. Writing periodically to file provides both
        a way to examine the progress of the simulation and to back up
        the data. Default: 600 s.
    ensemble_data_write_interval
        Interval at which data is written to the data container. This
        includes for example the current value of the calculator
        (i.e., usually the energy) as well as ensembles specific fields
        such as temperature or the number of atoms of different species.
        Default: Number of sites in the :attr:`structure`.
    trajectory_write_interval
        Interval at which the current occupation vector of the atomic
        configuration is written to the data container.
        Default: Number of sites in the :attr:`structure`.
    sublattice_probabilities
        Probability for picking a sublattice when doing a random flip.
        The list should be as long as the number of sublattices and should
        sum up to 1.

    Example
    -------
    The following snippet illustrate how to carry out a simple Monte Carlo
    simulation in the variance-constrained semi-canonical ensemble. Here, the
    parameters of the cluster expansion are set to emulate a simple Ising model
    in order to obtain an example that can be run without modification. In
    practice, one should of course use a proper cluster expansion::

        >>> from ase.build import bulk
        >>> from icet import ClusterExpansion, ClusterSpace
        >>> from mchammer.calculators import ClusterExpansionCalculator

        >>> # prepare cluster expansion
        >>> # the setup emulates a second nearest-neighbor (NN) Ising model
        >>> # (zerolet and singlet ECIs are zero; only first and second neighbor
        >>> # pairs are included)
        >>> prim = bulk('Au')
        >>> cs = ClusterSpace(prim, cutoffs=[4.3], chemical_symbols=['Ag', 'Au'])
        >>> ce = ClusterExpansion(cs, [0, 0, 0.1, -0.02])

        >>> # set up and run MC simulation
        >>> structure = prim.repeat(3)
        >>> calc = ClusterExpansionCalculator(structure, ce)
        >>> phi = 0.6
        >>> mc = VCSGCEnsemble(structure=structure, calculator=calc,
        ...                   temperature=600,
        ...                   dc_filename='myrun_vcsgc.dc',
        ...                   phis={'Au': phi},
        ...                   kappa=200)
        >>> mc.run(100)  # carry out 100 trial swaps
    """

    def __init__(self, structure: Atoms,
                 calculator: BaseCalculator,
                 temperature: float,
                 phis: Dict[str, float],
                 kappa: float,
                 boltzmann_constant: float = kB,
                 user_tag: str = None,
                 random_seed: int = None,
                 dc_filename: str = None,
                 data_container: str = None,
                 data_container_write_period: float = 600,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None,
                 sublattice_probabilities: List[float] = None) -> None:

        self._ensemble_parameters = dict(temperature=temperature,
                                         kappa=kappa)
        self._boltzmann_constant = boltzmann_constant

        # Save ensemble parameters
        for sym, phi in phis.items():
            if isinstance(sym, str):
                chemical_symbol = sym
            else:
                chemical_symbol = chemical_symbols[sym]
            phi_sym = 'phi_{}'.format(chemical_symbol)
            self._ensemble_parameters[phi_sym] = phi

        super().__init__(
            structure=structure,
            calculator=calculator,
            user_tag=user_tag,
            random_seed=random_seed,
            dc_filename=dc_filename,
            data_container=data_container,
            data_container_class=DataContainer,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval,
            boltzmann_constant=boltzmann_constant
        )

        # Save phis (need self.configuration to check sublattices so
        # we do it last)
        self._phis = get_phis(phis)

        # Check that each sublattice has N - 1 phis
        for sl in self.sublattices.active_sublattices:
            count_specified_elements = 0
            for number in sl.atomic_numbers:
                if number in self._phis.keys():
                    count_specified_elements += 1
            if count_specified_elements != len(sl.atomic_numbers) - 1:
                raise ValueError('phis must be set for N - 1 elements on a '
                                 'sublattice with N species')

        if sublattice_probabilities is None:
            self._flip_sublattice_probabilities = self._get_flip_sublattice_probabilities()
        else:
            self._flip_sublattice_probabilities = sublattice_probabilities

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        sublattice_index = self.get_random_sublattice_index(
            probability_distribution=self._flip_sublattice_probabilities)
        return self.do_vcsgc_flip(
            phis=self.phis, kappa=self.kappa, sublattice_index=sublattice_index)

    @property
    def temperature(self) -> float:
        """ Temperature :math:`T` (see parameters section above). """
        return self.ensemble_parameters['temperature']

    @property
    def phis(self) -> Dict[int, float]:
        r"""
        Average constraint parameters :math:`\phi_i` for all species but one
        (referred to as :math:`\bar{\phi}` in [SadErh12]_).
        """
        return self._phis

    @property
    def kappa(self) -> float:
        r"""
        Variance constraint parameter :math:`\bar{\kappa}` (see parameters section above).
        """
        return self.ensemble_parameters['kappa']

    def _get_ensemble_data(self) -> Dict:
        """
        Returns a dict with the default data of the ensemble. This includes
        atom counts and free energy derivative.
        """
        data = super()._get_ensemble_data()

        # free energy derivative
        data.update(self._get_vcsgc_free_energy_derivatives(self.phis, self.kappa))

        # species counts
        data.update(self._get_species_counts())

        return data


def get_phis(phis: Union[Dict[int, float], Dict[str, float]]) -> Dict[int, float]:
    """Get phis as used in the VCSGC ensemble.

    Parameters
    ----------
    phis
        phis that will be transformed to the format used by the ensemble.
    """
    if not isinstance(phis, dict):
        raise TypeError(f'phis has the wrong type: {type(phis)}')

    # Translate to atomic numbers if necessary
    phis_ret = {}
    for key, phi in phis.items():
        if isinstance(key, str):
            atomic_number = atomic_numbers[key]
            phis_ret[atomic_number] = phi
        elif isinstance(key, int):
            phis_ret[key] = phi

    return phis_ret
