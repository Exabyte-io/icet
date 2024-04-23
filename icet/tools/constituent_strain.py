import numpy as np
import itertools
from ase import Atoms
from typing import List, Tuple, Callable
from ase.symbols import chemical_symbols as ase_chemical_symbols
from functools import partial
from .constituent_strain_helper_functions import (_get_structure_factor,
                                                  _get_partial_structure_factor)


class KPoint:
    """
    Class for handling each k point in a supercell separately.

    Parameters
    ----------
    kpt
        k-point coordinates.
    multiplicity
        Multiplicity of this k-point.
    structure_factor
        Current structure associated with this k-point.
    strain_energy_function
        Function that takes a concentration and a list of parameters
        and returns strain energy.
    damping
        Damping at this k-point in units of Ångstrom.
   """
    def __init__(self, kpt: np.ndarray, multiplicity: float, structure_factor: float,
                 strain_energy_function: Callable[[float, List[float]], float],
                 damping: float):

        self.kpt = kpt
        self.multiplicity = multiplicity
        self.structure_factor = structure_factor
        self.structure_factor_after = self.structure_factor
        self.strain_energy_function = strain_energy_function
        self.damping_factor = np.exp(-(damping * np.linalg.norm(self.kpt)) ** 2)


class ConstituentStrain:
    r"""
    Class for handling constituent strain in cluster expansions
    (see Laks et al., Phys. Rev. B **46**, 12587 (1992) [LakFerFro92]_).
    This makes it possible to use cluster expansions to describe systems
    with strain due to, for example, coherent phase separation.
    For an extensive example on how to use this module,
    please see :ref:`this example <constituent_strain_example>`.

    Parameters
    ----------
    supercell
        Defines supercell that will be used when
        calculating constituent strain.
    primitive_structure
        Primitive structure the supercell is based on.
    chemical_symbols
        List with chemical symbols involved, such as ``['Ag', 'Cu']``.
    concentration_symbol
        Chemical symbol used to define concentration,
        such as ``'Ag'``.
    strain_energy_function
        A function that takes two arguments, a list of parameters and
        concentration (e.g., ``[0.5, 0.5, 0.5]`` and ``0.3``),
        and returns the corresponding strain energy.
        The parameters are in turn determined by ``k_to_parameter_function``
        (see below). If ``k_to_parameter_function`` is None,
        the parameters list will be the k-point. For more information, see
        :ref:`this example <constituent_strain_example>`.
    k_to_parameter_function
        A function that takes a k-point as a list of three floats
        and returns a parameter vector that will be fed into
        :attr:`strain_energy_function` (see above). If ``None``, the k-point
        itself will be the parameter vector to :attr:`strain_energy_function`.
        The purpose of this function is to be able to precompute
        any factor in the strain energy that depends on the k-point
        but not the concentration. For more information, see
        :ref:`this example <constituent_strain_example>`.
    damping
        Damping factor :math:`\eta` used to suppress impact of
        large-magnitude k-points by multiplying strain with
        :math:`\exp(-(\eta \mathbf{k})^2)` (unit Ångstrom).
    tol
        Numerical tolerance when comparing k-points (units of
        inverse Ångstrom).
    """

    def __init__(self,
                 supercell: Atoms,
                 primitive_structure: Atoms,
                 chemical_symbols: List[str],
                 concentration_symbol: str,
                 strain_energy_function: Callable[[float, List[float]], float],
                 k_to_parameter_function: Callable[[List[float]], List[float]] = None,
                 damping: float = 1.0,
                 tol: float = 1e-6):
        self.natoms = len(supercell)

        if len(chemical_symbols) < 2:
            raise ValueError('Please specify two chemical symbols.')
        elif len(chemical_symbols) > 2:
            raise NotImplementedError('The constituent strain module currently'
                                      ' only works for binary systems.')
        spins = [1, -1]
        self.spin_variables = {ase_chemical_symbols.index(sym): spin
                               for sym, spin in zip(sorted(chemical_symbols), spins)}
        for key, value in self.spin_variables.items():
            if value == 1:
                self.spin_up = key
                break

        self.supercell = supercell
        self.concentration_number = ase_chemical_symbols.index(concentration_symbol)

        # Initialize k-points for this supercell
        self.kpoints = []
        initial_occupations = self.supercell.get_atomic_numbers()
        for kpt, multiplicity in _generate_k_points(primitive_structure, supercell, tol=tol):
            if np.allclose(kpt, [0, 0, 0], atol=tol):
                continue

            S = _get_structure_factor(occupations=initial_occupations,
                                      positions=self.supercell.get_positions(),
                                      kpt=kpt,
                                      spin_up=self.spin_up)

            if k_to_parameter_function is None:
                parameters = kpt
            else:
                parameters = k_to_parameter_function(kpt)
            self.kpoints.append(KPoint(kpt=kpt,
                                       multiplicity=multiplicity,
                                       structure_factor=S,
                                       damping=damping,
                                       strain_energy_function=partial(strain_energy_function,
                                                                      parameters)))

    def get_concentration(self, occupations: np.ndarray) -> float:
        """
        Calculate current concentration.

        Parameters
        ----------
        occupations
            Current occupations.
        """
        return sum(occupations == self.concentration_number) / len(occupations)

    def _get_constituent_strain_term(self, kpoint: KPoint,
                                     occupations: List[int],
                                     concentration: float) -> float:
        """Calculate constituent strain corresponding to a specific k-point.
        That value is returned and also stored in the corresponding
        :class:`KPoint <icet.tools.constituent_strain.KPoint>`.

        Parameters
        ----------
        kpoint
            The k-point to be calculated.
        occupations
            Current occupations of the structure.
        concentration
            Concentration in the structure.

        """
        if abs(concentration) < 1e-9 or abs(1 - concentration) < 1e-9:
            kpoint.structure_factor = 0.0
            return 0.0
        else:
            # Calculate structure factor
            S = _get_structure_factor(
                occupations, self.supercell.get_positions(), kpoint.kpt, self.spin_up)

            # Save it for faster calculation later on
            kpoint.structure_factor = S

            # Constituent strain excluding structure factor
            DE_CS = kpoint.strain_energy_function(concentration)
            return DE_CS * kpoint.multiplicity * abs(S)**2 * kpoint.damping_factor / \
                (4 * concentration * (1 - concentration))

    def get_constituent_strain(self, occupations: List[int]) -> float:
        """
        Calculate total constituent strain.

        Parameters
        ----------
        occupations
            Current occupations.
        """
        c = self.get_concentration(occupations)
        E_CS = 0.0
        for kpoint in self.kpoints:
            E_CS += self._get_constituent_strain_term(kpoint=kpoint,
                                                      occupations=occupations,
                                                      concentration=c)
        return E_CS

    def get_constituent_strain_change(self, occupations: np.ndarray, atom_index: int) -> float:
        """
        Calculate change in constituent strain upon change of the
        occupation of one site.

        .. warning ::
            This function is dependent on the internal state of the
            :class:`ConstituentStrain` object and **should typically only
            be used internally by mchammer**. Specifically, the structure
            factor is saved internally to speed up computation.
            The first time this function is called, :attr:`occupations`
            must be the same array as was used to initialize the
            :class:`ConstituentStrain` object, or the same as was last used
            when :func:`get_constituent_strain` was called. After the
            present function has been called, the same occupations vector
            need to be used the next time as well, unless :func:`accept_change`
            has been called, in which case :attr:`occupations` should incorporate
            the changes implied by the previous call to the function.

        Parameters
        ----------
        occupations
            Occupations before change.
        atom_index
            Index of site the occupation of which is to be changed.
        """
        # Determine concentration before and after
        n = sum(occupations == self.concentration_number)
        c_before = n / self.natoms
        if occupations[atom_index] == self.concentration_number:
            c_after = (n - 1) / self.natoms
        else:
            c_after = (n + 1) / self.natoms

        E_CS_change = 0.0
        position = self.supercell[atom_index].position
        spin = self.spin_variables[occupations[atom_index]]
        for kpoint in self.kpoints:
            # Change in structure factor
            S_before = kpoint.structure_factor
            dS = _get_partial_structure_factor(kpoint.kpt, position, self.natoms)
            S_after = S_before - 2 * spin * dS

            # Save the new value in a variable such that it can be
            # accessed later if the change is accepted
            kpoint.structure_factor_after = S_after

            # Energy before
            if abs(c_before) < 1e-9 or abs(c_before - 1) < 1e-9:
                E_before = 0.0
            else:
                E_before = kpoint.strain_energy_function(c_before)
                E_before *= abs(S_before)**2 / (4 * c_before * (1 - c_before))

            # Energy after
            if abs(c_after) < 1e-9 or abs(c_after - 1) < 1e-9:
                E_after = 0.0
            else:
                E_after = kpoint.strain_energy_function(c_after)
                E_after *= abs(S_after)**2 / (4 * c_after * (1 - c_after))

            # Difference
            E_CS_change += kpoint.multiplicity * kpoint.damping_factor * (E_after - E_before)
        return E_CS_change

    def accept_change(self) -> None:
        """
        Update structure factor for each kpoint to the value
        in :attr:`structure_factor_after`. This makes it possible to
        efficiently calculate changes in constituent strain with
        the :func:`get_constituent_strain_change` function; this function
        should be called if the last occupations used to call
        :func:`get_constituent_strain_change` should be the starting point
        for the next call of :func:`get_constituent_strain_change`.
        This is taken care of automatically by the Monte Carlo
        simulations in :program:`mchammer`.
        """
        for kpoint in self.kpoints:
            kpoint.structure_factor = kpoint.structure_factor_after


def _generate_k_points(primitive_structure: Atoms,
                       supercell: Atoms,
                       tol: float) -> Tuple[np.ndarray, float]:
    """
    Generate all k-points in the 1BZ of the primitive cell
    that potentially correspond to a nonzero structure factor.
    These are all the k-points that are integer multiples of
    the reciprocal supercell.

    Parameters
    ----------
    primitive_structure
        Primitive structure that the supercell is based on.
    supercell
        Supercell of primitive structure.
    """
    reciprocal_primitive = np.linalg.inv(primitive_structure.cell)  # column vectors
    reciprocal_supercell = np.linalg.inv(supercell.cell)  # column vectors

    # How many k-points are there?
    # This information is needed to know when to stop looking for more,
    # i.e., it implicitly determines the iteration limits.
    nkpoints = int(round(np.linalg.det(reciprocal_primitive)
                         / np.linalg.det(reciprocal_supercell)))

    # The gamma point is always present
    yield np.array([0., 0., 0.]), 1.0
    found_kpoints = [np.array([0., 0., 0.])]
    covered_nkpoints = 1

    # Now loop until we have found all k-points.
    # We loop by successively increasing the sum of the
    # absolute values of the components of the reciprocal
    # supercell vectors, and we stop when we have
    # found the correct number of k-points.
    component_sum = 0
    while covered_nkpoints < nkpoints:
        component_sum += 1
        for ijk_p in _ordered_combinations(component_sum, 3):
            for signs in itertools.product([-1, 1], repeat=3):
                if np.any([ijk_p[i] == 0 and signs[i] < 0 for i in range(3)]):
                    continue
                q_sc = np.multiply(ijk_p, signs)
                k = np.dot(reciprocal_supercell, q_sc)

                # Now check whether we can come closer to the gamma point
                # by translation with primitive reciprocal lattice vectors.
                # In other words, we translate the point to the 1BZ.
                k = _translate_to_1BZ(k, reciprocal_primitive, tol=tol)
                equivalent_kpoints = _find_equivalent_kpoints(k, reciprocal_primitive, tol=tol)

                # Have we already found this k-point?
                found = False
                for k in equivalent_kpoints:
                    for k_comp in found_kpoints:
                        if np.linalg.norm(k - k_comp) < tol:
                            # Then we had already found it
                            found = True
                            break
                    if found:
                        break
                else:
                    # Then this is a new k-point
                    # Yield it and its equivalent friends
                    covered_nkpoints += 1
                    for k in equivalent_kpoints:
                        found_kpoints.append(k)
                        yield k, 1 / len(equivalent_kpoints)

        # Break if we have found all k-points
        assert covered_nkpoints <= nkpoints
        if covered_nkpoints == nkpoints:
            break


def _ordered_combinations(s: int, n: int) -> List[int]:
    """
    Recursively generate all combinations of n integers
    that sum to s.

    Parameters
    ----------
    s
        Required sum of the combination
    n
        Number of intergers in the list
    """
    if n == 1:
        yield [s]
    else:
        for i in range(s + 1):
            for rest in _ordered_combinations(s - i, n - 1):
                rest.append(i)
                yield rest


def _translate_to_1BZ(kpt: np.ndarray, primitive: np.ndarray, tol: float) -> np.ndarray:
    """
    Translate k-point into 1BZ by translating it by
    primitive lattice vectors and testing if that takes
    us closer to gamma. The algorithm works by recursion
    such that we keep translating until we no longer get
    closer (will this always work?).

    Parameters
    ----------
    kpt
        coordinates of k-point that we translate
    primitive
        primitive reciprocal cell with lattice vectors as
        column vectors
    tol
        numerical tolerance when comparing k-points (units of
        inverse Ångstrom)

    Returns
    -------
    the k-point translated into 1BZ
    """
    original_distance_from_gamma = np.linalg.norm(kpt)
    min_distance_from_gamma = original_distance_from_gamma
    best_kpt = kpt
    # Loop through translations and see if any takes us closer to gamma
    for translation in _generate_primitive_translations(primitive):
        new_kpt = kpt + translation
        new_distance_from_gamma = np.linalg.norm(new_kpt)
        if new_distance_from_gamma < min_distance_from_gamma:
            min_distance_from_gamma = new_distance_from_gamma
            best_kpt = new_kpt
    if abs(original_distance_from_gamma - min_distance_from_gamma) < tol:
        # Then we did not come closer to gamma, so we are done
        return best_kpt
    else:
        # We came closer to gamma. Maybe we can come even closer?
        return _translate_to_1BZ(best_kpt, primitive, tol=tol)


def _find_equivalent_kpoints(kpt: np.ndarray,
                             primitive: np.ndarray,
                             tol: float) -> List[np.ndarray]:
    """
    Find k-points in 1BZ equivalent to this k-point, i.e.,
    points in the 1BZ related by a primitive translation
    (for example, if a k-point lies at the edge of the 1BZ, we
    will find it on the opposite side of the 1BZ too).

    Parameters
    ----------
    kpt
        k-point coordinates
    primitive
        primitive reciprocal lattice vectors as columns
    tol
        numerical tolerance when comparing k-points (units of
        inverse Ångstrom)

    Returns
    -------
    list of equivalent k-points
    """
    original_norm = np.linalg.norm(kpt)
    equivalent_kpoints = [kpt]
    for translation in _generate_primitive_translations(primitive):
        if abs(original_norm - np.linalg.norm(kpt + translation)) < tol:
            equivalent_kpoints.append(kpt + translation)
    return equivalent_kpoints


def _generate_primitive_translations(primitive: np.ndarray) -> np.ndarray:
    """
    Generate the 26 (3x3x3 - 1) primitive translation vectors
    defined by (0/+1/-1, 0/+1/-1, 0/+1/-1) (exlcluding (0, 0, 0))

    Parameters
    ----------
    primitive
        Lattice vectors as columns

    Yields
    ------
    traslation vector, a multiple of primitive lattice vectors
    """
    for ijk in itertools.product((0, 1, -1), repeat=3):
        if ijk == (0, 0, 0):
            continue
        yield np.dot(primitive, ijk)
