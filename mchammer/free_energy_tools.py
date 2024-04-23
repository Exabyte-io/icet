from typing import List
from icet import ClusterSpace
from icet.core.sublattices import Sublattices

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ase import Atoms
from ase.units import kB

from mchammer import DataContainer

import operator
from collections import Counter
from math import factorial
from functools import reduce


def _lambda_function_forward(n_steps: int, step: int) -> float:
    """
    Returns the current lambda value for a backward calculation.

    Parameters
    ----------
    n_steps
        Total number of steps..
    step
        Current step.
    """
    x = (step + 1) / (n_steps)
    lam = x**5*(70*x**4 - 315*x**3 + 540*x**2 - 420*x + 126)
    # Due to numerical precision lambda may be slightly outside the interval [0.0, 1.0]
    lam = np.clip(lam, 0.0, 1.0)
    return lam


def _lambda_function_backward(n_steps: int, step: int) -> float:
    """
    Returns the current lambda value for a backward calculation.

    Parameters
    ----------
    n_steps
        Total number of steps.
    step
        Current step.
    """
    return 1 - _lambda_function_forward(n_steps, step)


def _stirling(n: int) -> float:
    """ Stirling approximation of log(n!). """
    return n * np.log(n) - n + 0.5 * np.log(2*np.pi*n)


def _lognpermutations(array: List[int]) -> float:
    """If _npermutations becomes too large we resort to calculating the
    logarithm directly using Stirling's approximation, i.e., we
    calculate

    .. math::

        log(n!) - (log(a1!) + log(a2!) + log(a3!) + ... + log(ak!)

    where we denote in the code

    .. math::

        v  n = log(n!)
        den = (log(a1!) + log(a2!) + log(a3!) + ... + log(ak!)

    """

    n = _stirling(len(array))
    # Gets the number of each unique atom type
    mults = Counter(array).values()
    den = reduce(operator.add,
                 (_stirling(v) for v in mults),
                 1)
    return n - den


def _npermutations(array: List[int]) -> float:
    """
    Returns the total number of ways we can permutate the array which is given by

    .. math::

        n! / (a1! * a2! * a3! * ... * ak!)

    where we denote in the code

    .. math::

        den = (a1! * a2! * a3! * ... * ak!)
    """
    n = factorial(len(array))
    # Gets the number of each unique atom type
    a = Counter(array).values()
    den = reduce(operator.mul,
                 (factorial(v) for v in a),
                 1)
    return n / den


def _ideal_mixing_entropy(swap_sublattice_probabilities: List[int],
                          atoms_on_sublattices: np.ndarray,
                          boltzmann_constant: float) -> float:
    """
    Calculates the ideal mixing entropy for the supercell.

    Parameters
    ----------
    swap_sublattice_probabilites
        Sublattices that are active during the MC simulation.
    atoms_on_sublattices
        Sorted atomic numbers in sublattices lists.
    boltzmann_constant
        Value of Boltzmann constant in the units used in the MC simulation.
    """
    log_multiplicity = []
    sublattices = np.array(swap_sublattice_probabilities) > 0
    for aos in atoms_on_sublattices[sublattices]:
        try:
            log_multiplicity.append(np.log(_npermutations(aos)))
        except Exception:
            log_multiplicity.append(_lognpermutations(aos))
    return boltzmann_constant * np.sum(log_multiplicity)


def _get_atoms_on_sublattice(structure: Atoms,
                             sublattices: Sublattices) -> np.ndarray:
    """Sorts the atom symbols in the structure to lists involving
    specific sublattices, for example,

    [[5, 5, 3, 3, 3], [11, 11, 10, 10, 10 10, 10]]

    Parameters
    ----------
    structure
        Atomic structure.
    sublattices
        Sublattices for the supercell obtained from the cluster space.
    """
    _atoms_on_sublattices = []
    for sl in sublattices:
        atoms_on_sublattice = []
        for an in sl.atomic_numbers:
            natoms = np.where(structure.numbers == an)[0].size
            atoms_on_sublattice.extend([an] * natoms)
        _atoms_on_sublattices.append(atoms_on_sublattice)
    _atoms_on_sublattices = np.array(_atoms_on_sublattices,
                                     dtype=object)
    return _atoms_on_sublattices


def get_free_energy_thermodynamic_integration(
        dc: DataContainer,
        cluster_space: ClusterSpace,
        forward: bool,
        max_temperature: float = np.inf,
        sublattice_probabilities: List[float] = None,
        boltzmann_constant: float = kB,
) -> (np.ndarray, np.ndarray):
    r"""Returns the free energy calculated via thermodynamic integration
    using the :class:`ThermodynamicIntegrationEnsemble
    <mchammer.ensembles.ThermodynamicIntegrationEnsemble>`.

    The temperature dependence of the free energy can be extracted
    from the thermodynamic integration as

    .. math::

        F(T) =  \frac{F_0(\lambda)}{\lambda} + \frac{T_0}{\lambda} S_\text{B}

    where :math:`S_\text{B}` is the Boltzmann entropy,

    .. math::

        T = \frac{T_0}{\lambda}

    and

    .. math::

        F_0(\lambda) = \int_0^\lambda \left\langle\frac{\mathrm{d}H(\lambda)}
                       {\mathrm{d}\lambda}\right\rangle_{H} \mathrm{d}\lambda

    Parameters
    ----------
    dc
        Data container from the thermodynamic integration simulation.
    cluster_space
        Cluster space used to construct the cluster expansion used for the simulation.
    forward
        Whether or not the thermodynamic integration was carried out forward or backward.
    max_temperature
        Largest temperature to extract from the thermodynamic integration.
    sublattice_probababilites
        Sublattice probabilties that were provided to the thermodynamic integration
        simulation.
    boltzmann_constant
        Boltzmann constant in the units used for the thermodynamic integration
        simulation.

    """

    lambdas, potentials = dc.get('lambda', 'potential')
    if not forward:
        # We want to integrate from high to low temperature even though
        # the simulation was from low to high.
        potentials = potentials[::-1]
        lambdas = lambdas[::-1]

    sublattices = cluster_space.get_sublattices(dc.structure)
    if sublattice_probabilities is None:
        sublattice_probabilities = [True] * len(sublattices)

    atoms_on_sublattices = _get_atoms_on_sublattice(dc.structure,
                                                    sublattices)
    ideal_mixing_entropy = _ideal_mixing_entropy(sublattice_probabilities,
                                                 atoms_on_sublattices,
                                                 boltzmann_constant)

    temperature = dc._ensemble_parameters['temperature']
    with np.errstate(divide='ignore'):
        # First value of lambdas will be zero (we ignore this warning)
        temperatures = (1 / lambdas) * temperature
    temp_inds = np.where(temperatures <= max_temperature)[0]

    free_energy_change = cumulative_trapezoid(potentials, x=lambdas, initial=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        # First value of lambdas will be zero (we ignore this warning)
        free_energy = (free_energy_change / lambdas -
                       temperatures * ideal_mixing_entropy)
    free_energy[np.isnan(free_energy)] = 0
    return (temperatures[temp_inds], free_energy[temp_inds])


def get_free_energy_temperature_integration(
        dc: DataContainer,
        cluster_space: ClusterSpace,
        forward: bool,
        temperature_reference: float,
        free_energy_reference: float = None,
        sublattice_probabilities: List[float] = None,
        max_temperature: float = np.inf,
        boltzmann_constant: float = kB,
) -> (np.ndarray, np.ndarray):
    r""" Returns the free energy calculated using temperature integration and
    the corresponding temperature.

    .. math::

        \frac{A(T_{2})}{T_{2}} = \frac{A(T_{1})}{T_{1}}
        - \int_{T_{1}}^{T_{2}}\frac{U(T)}{T^2}\mathrm{d}T

    Parameters
    ----------
    dc
        Data container from a canonical annealing simulation.
        The first (last for :attr:`forward=False`) temperature in the
        data container has to be at the same temperature as
        :attr:`temperature_reference`.
    cluster_space
        Cluster space used to construct the cluster expansion
        that was used in the simulations.
    forward
        If ``True`` the canonical annealing simulation was carried out
        from high to low temperature, otherwise the opposite is assumed.
    temperature_reference
        Temperature at which :attr:`free_energy_reference` was calculated
    free_energy_reference
        Reference free energy. If set to ``None`` it will be assumeed that the
        free energy at :attr:`temperature_reference` can be approximated by
        :math:`T S_B` where :math:`S_B` is the ideal mixing entropy.
    sublattice_probababilites
        Sublattice probabilties that were provided to the canonical annealing
        simulation.
    max_temperature
        Largest temperature to extract from the temperature integration.
    boltzmann_constant
        Boltzmann constant in the units used for the thermodynamic integration
        simulation.
    """
    temperatures, potentials = dc.get('temperature', 'potential')
    if not forward:
        # We want to integrate from high to low temperature even though
        # the simulation was from low to high.
        potentials = potentials[::-1]
        temperatures = temperatures[::-1]

    if not np.isclose(temperatures[0], temperature_reference):
        raise ValueError(
            'The first or the last temperature in the temperature list'
            ' has to equal the reference temperature.'
            f' reference_temperature: {temperature_reference}'
            f' first canonical temperature: {temperatures[0]}')

    temp_inds = np.where(temperatures <= max_temperature)[0]

    if free_energy_reference is None:
        sublattices = cluster_space.get_sublattices(dc.structure)
        if sublattice_probabilities is None:
            sublattice_probabilities = [True] * len(sublattices)

        atoms_on_sublattices = _get_atoms_on_sublattice(dc.structure,
                                                        sublattices)
        ideal_mixing_entropy = _ideal_mixing_entropy(sublattice_probabilities,
                                                     atoms_on_sublattices,
                                                     boltzmann_constant)
        free_energy_reference = -ideal_mixing_entropy * temperature_reference

    reference = free_energy_reference / temperature_reference
    integral = cumulative_trapezoid(potentials / temperatures**2,
                                    x=temperatures, initial=0)
    free_energy = (reference - integral) * temperatures
    return (temperatures[temp_inds], free_energy[temp_inds])
