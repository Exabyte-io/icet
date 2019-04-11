import itertools
import random
from typing import List
import numpy as np
from mchammer.ensembles import TargetClusterVectorAnnealing
from mchammer.calculators import (TargetVectorCalculator,
                                  compare_cluster_vectors)
from icet.tools import (enumerate_structures,
                        enumerate_supercells)
from icet import ClusterSpace
from ase import Atoms
from ase.data import chemical_symbols as periodic_table


def generate_target_structure(cluster_space: ClusterSpace, max_size: int,
                              target_concentrations: dict,
                              target_cluster_vector: List[float],
                              include_smaller_cells: bool = True,
                              T_start: float = 5.0, T_stop: float = 0.001,
                              n_steps: float = None,
                              optimality_weight: float = 1.0,
                              random_seed: int = None,
                              tol: float = 1e-5) -> Atoms:
    """
    Given a ``cluster_space`` and a ``target_cluster_vector``, generate
    a structure that as closely as possible matches that cluster vector.
    The search is performed among all inequivalent supercells shapes up
    to a certain size.

    Internally the function uses a simulated annealing algorithm and the
    difference between two cluster vectors is calculated with the
    measure suggested by A. van de Walle et al. in Calphad **42**, 13-18
    (2013) [WalTiwJon13]_ (for more information, see
    :class:`mchammer.calculators.TargetVectorCalculator`).

    Parameters
    ----------
    cluster_space
        a cluster space defining the lattice to be decorated.
    max_size
        maximum supercell size
    target_concentrations
        concentration of each species in the target structure (for
        example ``{'Ag': 0.5, 'Pd': 0.5}``

        Concentrations are always expressed with respect to all atoms in
        the supercell, which implies that the sum of all concentrations
        should always be 1. In the case of multiple sublattices, a valid
        specification would thus be ``{'Au': 0.25, 'Pd': 0.25, 'H': 0.1,
        'V': 0.4}``.
    target_cluster_vector
        cluster vector that the generated structure should match as
        closely as possible
    include_smaller_cells
        if True, search among all supercell sizes including
        ``max_size``, else search only among those exactly matching
        ``max_size``
    T_start
        artificial temperature at which the simulated annealing starts
    T_stop
        artifical temperature at which the simulated annealing stops
    n_steps
        total number of Monte Carlo steps in the simulation
    optimality_weight
        controls weighting :math:`L` of perfect correlations, see
        :class:`mchammer.calculators.TargetVectorCalculator`
    random_seed
        seed for the random number generator used in the
        Monte Carlo simulation
    tol
        Numerical tolerance
    """
    _validate_concentrations(target_concentrations, cluster_space)

    supercells = []
    calculators = []
    if include_smaller_cells:
        sizes = range(1, max_size + 1)
    else:
        sizes = [max_size]

    for size in sizes:
        # For efficiency, make a first check that the current size is
        # commensurate with all concentrations (example: {'Au': 0.5,
        # 'Pd': 0.5} would not be commensurate with a supercell with 3
        # atoms). This step does *not* ensure that the concentrations
        # are commensurate with all sublattices (that is checked
        # internally later on).
        natoms = size * len(cluster_space.primitive_structure)
        if not _concentrations_fit_atom_count(natoms, target_concentrations):
            continue

        # Loop over all inequivalent supercells and intialize
        # them with a "random" occupation of symbols that
        # fulfill the target concentrations
        for supercell in enumerate_supercells(cluster_space.primitive_structure,
                                              [size]):
            _decorate_atoms_randomly(supercell, cluster_space,
                                     target_concentrations)
            supercells.append(supercell)
            calculators.append(TargetVectorCalculator(supercell, cluster_space,
                                                      target_cluster_vector,
                                                      optimality_weight=optimality_weight,
                                                      optimality_tol=tol))

    ens = TargetClusterVectorAnnealing(atoms=supercells, calculators=calculators,
                                       T_start=T_start, T_stop=T_stop,
                                       random_seed=random_seed)
    return ens.generate_structure(number_of_trial_steps=n_steps)


def generate_sqs(cluster_space: ClusterSpace, max_size: int,
                 target_concentrations: dict,
                 include_smaller_cells: bool = True,
                 T_start: float = 5.0, T_stop: float = 0.001,
                 n_steps: float = None,
                 optimality_weight: float = 1.0,
                 random_seed: int = None,
                 tol: float = 1e-5) -> Atoms:
    """
    Given a ``cluster_space``, generate a special quasirandom structure
    (SQS), i.e., a structure that for a given supercell size provides
    the best possible approximation to a random alloy [ZunWeiFer90]_.

    In the present case, this means that the generated structure will
    have a cluster vector that as closely as possible matches the
    cluster vector of an infintely large randomly decorated supercell.
    Internally the function uses a simulated annealing algorithm and the
    difference between two cluster vectors is calculated with the
    measure suggested by A. van de Walle et al. in Calphad **42**, 13-18
    (2013) [WalTiwJon13]_ (for more information, see
    :class:`mchammer.calculators.TargetVectorCalculator`).

    Parameters
    ----------
    cluster_space
        a cluster space defining the lattice to be decorated.
    max_size
        maximum supercell size
    target_concentrations
        concentration of each species in the target structure (for
        example ``{'Ag': 0.5, 'Pd': 0.5}``

        Concentrations are always expressed with respect to all atoms in
        the supercell, which implies that the sum of all concentrations
        should always be 1. In the case of multiple sublattices, a valid
        specification would thus be ``{'Au': 0.25, 'Pd': 0.25, 'H': 0.1,
        'V': 0.4}``.
    include_smaller_cells
        if True, search among all supercell sizes including
        ``max_size``, else search only among those exactly matching
        ``max_size``
    T_start
        artificial temperature at which the simulated annealing starts
    T_stop
        artifical temperature at which the simulated annealing stops
    n_steps
        total number of Monte Carlo steps in the simulation
    optimality_weight
        controls weighting :math:`L` of perfect correlations, see
        :class:`mchammer.calculators.TargetVectorCalculator`
    random_seed
        seed for the random number generator used in the
        Monte Carlo simulation
    tol
        Numerical tolerance
    """

    sqs_vector = _get_sqs_cluster_vector(cluster_space=cluster_space,
                                         target_concentrations=target_concentrations)
    return generate_target_structure(cluster_space=cluster_space,
                                     max_size=max_size,
                                     target_concentrations=target_concentrations,
                                     target_cluster_vector=sqs_vector,
                                     include_smaller_cells=include_smaller_cells,
                                     T_start=T_start, T_stop=T_stop,
                                     n_steps=n_steps,
                                     optimality_weight=optimality_weight,
                                     random_seed=random_seed,
                                     tol=tol)


def generate_sqs_by_enumeration(cluster_space: ClusterSpace, max_size: int,
                                target_concentrations: dict,
                                include_smaller_cells: bool = True,
                                optimality_weight: float = 1.0,
                                tol: float = 1e-5) -> Atoms:
    """
    Given a ``cluster_space``, generate a special quasirandom structure
    (SQS), i.e., a structure that for a given supercell size provides
    the best possible approximation to a random alloy [ZunWeiFer90]_.

    In the present case, this means that the generated structure will
    have a cluster vector that as closely as possible matches the
    cluster vector of an infintely large randomly decorated supercell.
    Internally the function uses a simulated annealing algorithm and the
    difference between two cluster vectors is calculated with the
    measure suggested by A. van de Walle et al. in Calphad **42**, 13-18
    (2013) [WalTiwJon13]_ (for more information, see
    :class:`mchammer.calculators.TargetVectorCalculator`).

    This functions generates SQS cells by exhaustive enumeration, which
    means that the generated SQS cell is guaranteed to be optimal with
    regard to the specified measure and cell size.

    Parameters
    ----------
    cluster_space
        a cluster space defining the lattice to be decorated.
    max_size
        maximum supercell size
    target_concentrations
        concentration of each species in the target structure (for
        example ``{'Ag': 0.5, 'Pd': 0.5}``

        Concentrations are always expressed with respect to all atoms in
        the supercell, which implies that the sum of all concentrations
        should always be 1. In the case of multiple sublattices, a valid
        specification would thus be ``{'Au': 0.25, 'Pd': 0.25, 'H': 0.1,
        'V': 0.4}``.
    include_smaller_cells
        if True, search among all supercell sizes including
        ``max_size``, else search only among those exactly matching
        ``max_size``
    optimality_weight
        controls weighting :math:`L` of perfect correlations, see
        :class:`mchammer.calculators.TargetVectorCalculator`
    tol
        Numerical tolerance
    """
    _validate_concentrations(target_concentrations, cluster_space)
    sqs_vector = _get_sqs_cluster_vector(cluster_space=cluster_space,
                                         target_concentrations=target_concentrations)
    # Translate concentrations to the format required for concentration
    # restricted enumeration
    cr = {key: (value, value) for key, value in target_concentrations.items()}

    orbit_data = cluster_space.orbit_data
    best_score = 1e9

    if include_smaller_cells:
        sizes = range(1, max_size + 1)
    else:
        sizes = [max_size]

    # Enumerate and calculate score for each structuer
    for structure in enumerate_structures(cluster_space.primitive_structure,
                                          sizes,
                                          cluster_space.chemical_symbols,
                                          concentration_restrictions=cr):
        cv = cluster_space.get_cluster_vector(structure)
        score = compare_cluster_vectors(cv_1=cv, cv_2=sqs_vector,
                                        orbit_data=orbit_data,
                                        optimality_weight=optimality_weight,
                                        tol=tol)

        if score < best_score:
            best_score = score
            best_structure = structure
    return best_structure


def _decorate_atoms_randomly(atoms: Atoms, cluster_space: ClusterSpace,
                             target_concentrations: dict):
    """
    Decorate an ``atoms`` object with "random" order but fulfilling
    ``target_concentrations``.

    Parameters
    ----------
    atoms
        Atoms object that will be decorated randomly
    cluster_space
        cluster space (becuse it carries information about sublattices
        etc)
    target_concentrations
        concentration of each species in the target structure (for
        example `{'Ag': 0.5, 'Pd': 0.5}`

        Concentrations are always expressed with respect to all atoms in
        the supercell, which implies that the sum of all concentrations
        should always be 1. In the case of multiple sublattices, a valid
        specification would thus be `{'Au': 0.25, 'Pd': 0.25, 'H': 0.1,
        'V': 0.4}`.
    """
    if not _concentrations_fit_atoms(atoms, cluster_space, target_concentrations):
        raise ValueError('Atoms object with {} atoms cannot accomodate '
                         'target concentrations {}'.format(len(atoms),
                                                           target_concentrations))

    # symbols_all will hold chemical_symbols of all sublattices
    symbols_all = [0] * len(atoms)
    for sublattice in cluster_space.get_sublattices(atoms):
        symbols = []  # chemical_symbols in one sublattice
        for chemical_symbol in sublattice.chemical_symbols:
            n_symbol = int(
                round(len(atoms) * target_concentrations[chemical_symbol]))
            symbols += [chemical_symbol] * n_symbol

        # Should not happen but you never know
        assert len(symbols) == len(sublattice.indices)

        # Shuffle to introduce some randomness
        random.shuffle(symbols)

        # Assign symbols to the right indices
        for symbol, lattice_site in zip(symbols, sublattice.indices):
            symbols_all[lattice_site] = symbol

    assert symbols_all.count(0) == 0
    atoms.set_chemical_symbols(symbols_all)


def _validate_concentrations(concentrations: dict,
                             cluster_space: ClusterSpace,
                             tol: float = 1e-5):
    """
    Validates concentration specification against a cluster space
    (raises `ValueError` if they do not match).

    Parameters
    ----------
    concentrations
        concentration specification
    cluster_space
        cluster space to check against
    tol
        Numerical tolerance
    """
    # Concentrations should sum to 1
    conc_sum = sum(list(concentrations.values()))
    if abs(conc_sum - 1.0) > tol:
        raise ValueError('Concentrations must sum up '
                         'to 1 (not {})'.format(conc_sum))

    # Symbols need to match
    # Flatten chemical symbols from cluster space
    chemical_symbols = [
        sym for syms in cluster_space.chemical_symbols for sym in syms]
    if tuple(sorted(chemical_symbols)) != tuple(sorted(concentrations.keys())):
        raise ValueError('Chemical symbols in cluster space ({}) are '
                         'not the same as those in the specified '
                         'concentrations ({})'.format(cluster_space.chemical_symbols,
                                                      list(concentrations.keys())))

    # Concentrations for each sublattice need to sum up to
    # its proportion of the total structure
    for sublattice in cluster_space.get_sublattices(cluster_space.primitive_structure):
        part = len(sublattice.indices) / len(cluster_space.primitive_structure)
        conc = sum([concentrations[sym]
                    for sym in sublattice.chemical_symbols])
        if abs(conc - part) > tol:
            raise ValueError('Concentrations must be specified such that '
                             'concentrations per element and per sublattice '
                             'match, sublattice with species {} contains a '
                             'fraction {:.2f} of all atoms, but the sum of '
                             'the concentrations of those symbols is '
                             '{:.2f}.'.format(sublattice.chemical_symbols,
                                              part,
                                              conc))


def _concentrations_fit_atom_count(atom_count: int,
                                   concentrations: dict,
                                   tol: float = 1e-5) -> bool:
    """
    Check if specified concentrations are commensurate with a
    certain number of atoms.

    Parameters
    ----------
    atom_count
        How many atoms?
    concentrations
        What concentrations? Example: ``{'Ag': 0.3, 'Au': 0.7}``
    tol
        Numerical tolerance
    """
    for conc in concentrations.values():
        n_symbol = conc * atom_count
        if abs(int(round(n_symbol)) - n_symbol) > tol:
            return False
    return True


def _concentrations_fit_atoms(atoms: Atoms,
                              cluster_space: ClusterSpace,
                              concentrations: dict,
                              tol: float = 1e-5) -> bool:
    """
    Check if specified concentrations are commensurate with a
    certain supercell (including sublattices)

    Parameters
    ----------
    atoms
        atoms object to be checked
    cluster_space
        corresponding cluster space
    concentrations
        What concentrations? Example: ``{'Ag': 0.3, 'Au': 0.7}``
    tol
        Numerical tolerance
    """
    # Check that concentrations are OK in each sublattice
    for sublattice in cluster_space.get_sublattices(atoms):
        # Since concentrations are specified relative to all atoms,
        # we must rescale when we check against the size of a sublattice
        scaling = len(atoms) / len(sublattice.indices)
        concentrations_sub = {elem: scaling * concentrations[elem]
                              for elem in sublattice.chemical_symbols}
        if not _concentrations_fit_atom_count(len(sublattice.indices),
                                              concentrations_sub,
                                              tol=tol):
            return False
    return True


def _get_sqs_cluster_vector(cluster_space: ClusterSpace,
                            target_concentrations: dict) -> np.ndarray:
    """
    Get the SQS vector for a certain cluster space and certain
    concentration. Here SQS vector refers to the cluster vector of an
    infintely large supercell with random decoration.

    Parameters
    ----------
    cluster_space
        The kind of lattice to be decorated
    target_concentrations
        concentration of each species in the target structure (for
        example `{'Ag': 0.5, 'Pd': 0.5}`

        Concentrations are always expressed with respect to all atoms in
        the supercell, which implies that the sum of all concentrations
        should always be 1. In the case of multiple sublattices, a valid
        specification would thus be `{'Au': 0.25, 'Pd': 0.25, 'H': 0.1,
        'V': 0.4}`.
    """
    _validate_concentrations(target_concentrations, cluster_space)

    sublattice_to_index = {letter: index for index,
                           letter in enumerate('ABCDEFGHIJKLMNOPQRSTVUWWXYZ')}
    all_sublattices = cluster_space.get_sublattices(
        cluster_space.primitive_structure)

    # Make a map from chemical symbol to integer, later on used
    # for evaluating cluster functions.
    # Internally, icet sorts species according to atomic numbers.
    # Also check that each symbol only occurs in one sublattice.
    symbol_to_integer_map = {}
    found_species = []
    for sublattice in all_sublattices:
        syms = sublattice.chemical_symbols
        atomic_numbers = [periodic_table.index(sym) for sym in syms]
        for i, species in enumerate(sorted(atomic_numbers)):
            if species in found_species:
                raise ValueError('Each chemical symbol can only occur on one '
                                 'sublattice, {} occurred more than '
                                 'once.'.format(periodic_table[species]))
            found_species.append(species)
            symbol_to_integer_map[periodic_table[species]] = i

    # Target concentrations refer to all atoms, but probabilities only
    # to the sublattice.
    probabilities = {}
    for sublattice in all_sublattices:
        mult_factor = len(cluster_space.primitive_structure) / \
            len(sublattice.indices)
        for symbol in sublattice.chemical_symbols:
            probabilities[symbol] = target_concentrations[symbol] * mult_factor

    # For every orbit, calculate average cluster function
    cv = [1.0]
    for orbit in cluster_space.orbit_data:
        if orbit['order'] < 1:
            continue

        # What sublattices are there in this orbit?
        sublattices = [all_sublattices[sublattice_to_index[letter]]
                       for letter in orbit['sublattices'].split('-')]

        # What chemical symbols do these sublattices refer to?
        symbol_groups = [
            sublattice.chemical_symbols for sublattice in sublattices]

        # How many allowed species in each of those sublattices?
        nbr_of_allowed_species = [len(symbol_group)
                                  for symbol_group in symbol_groups]

        # Calculate contribtion from every possible combination of
        # symbols weighted with their probability
        cluster_product_average = 0
        for symbols in itertools.product(*symbol_groups):
            cluster_product = 1
            for i, symbol in enumerate(symbols):
                mc_vector_component = orbit['multi_component_vector'][i]
                species_i = symbol_to_integer_map[symbol]
                prod = cluster_space.evaluate_cluster_function(nbr_of_allowed_species[i],
                                                               mc_vector_component,
                                                               species_i)
                cluster_product *= probabilities[symbol] * prod
            cluster_product_average += cluster_product
        cv.append(cluster_product_average)
    return np.array(cv)
