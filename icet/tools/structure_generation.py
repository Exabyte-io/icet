import itertools
from typing import List, Dict, Tuple, Union
import numpy as np
from mchammer.ensembles import TargetClusterVectorAnnealing
from mchammer.calculators import (TargetVectorCalculator,
                                  compare_cluster_vectors)
from icet.tools import (enumerate_structures,
                        enumerate_supercells)
from icet import ClusterSpace
from icet.input_output.logging_tools import logger
from ase import Atoms
from ase.data import chemical_symbols as periodic_table


def generate_target_structure_from_supercells(cluster_space: ClusterSpace,
                                              supercells: List[Atoms],
                                              target_concentrations: dict,
                                              target_cluster_vector: List[float],
                                              T_start: float = 5.0,
                                              T_stop: float = 0.001,
                                              n_steps: int = None,
                                              optimality_weight: float = 1.0,
                                              random_seed: int = None,
                                              random_start: bool = True,
                                              tol: float = 1e-5) -> Atoms:
    """
    Given a :attr:`cluster_space` and a :attr:`target_cluster_vector` and one
    or more :attr:`supercells`, generate a structure that as closely as
    possible matches that cluster vector.

    Internally the function uses a simulated annealing algorithm and the
    difference between two cluster vectors is calculated with the
    measure suggested by A. van de Walle et al. in Calphad **42**, 13-18
    (2013) [WalTiwJon13]_ (for more information, see
    :class:`mchammer.calculators.TargetVectorCalculator`).

    Parameters
    ----------
    cluster_space
        A cluster space defining the lattice to be occupied.
    supercells
        List of one or more supercells among which an optimal
        structure will be searched for.
    target_concentrations
        Concentration of each species in the target structure, per
        sublattice (for example ``{'Au': 0.5, 'Pd': 0.5}`` for a
        single sublattice Au-Pd structure, or
        ``{'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'H': 0.25, 'X': 0.75}}``
        for a system with two sublattices.
        The symbols defining sublattices ('A', 'B' etc) can be
        found by printing the :attr:`cluster_space`.
    target_cluster_vector
        Cluster vector that the generated structure should match as
        closely as possible.
    T_start
        Artificial temperature at which the simulated annealing starts.
    T_stop
        Artifical temperature at which the simulated annealing stops.
    n_steps
        Total number of Monte Carlo steps in the simulation.
    optimality_weight
        Controls weighting :math:`L` of perfect correlations, see
        :class:`mchammer.calculators.TargetVectorCalculator`.
    random_seed
        Seed for the random number generator used in the Monte Carlo simulation
        and used for initializing the occupation of the supercells if
        random_start is ``True``.
    random_start
        Randomly occupy starting structure, can be disabled
        if the user prefers to pass an initial structure.
    tol
        Numerical tolerance.
    """
    target_concentrations = _validate_concentrations(target_concentrations, cluster_space)

    calculators = []

    # Loop over all supercells and intialize
    # them with a "random" occupation of symbols that
    # fulfill the target concentrations
    valid_supercells = []
    warning_issued = False
    for supercell in supercells:
        supercell_copy = supercell.copy()
        if random_start:
            try:
                occupy_structure_randomly(supercell_copy, cluster_space,
                                          target_concentrations,
                                          random_seed)
            except ValueError:
                if not warning_issued:
                    logger.warning('At least one supercell was not commensurate with the specified '
                                   'target concentrations.')
                    warning_issued = True
                continue
        valid_supercells.append(supercell_copy)
        calculators.append(TargetVectorCalculator(supercell_copy, cluster_space,
                                                  target_cluster_vector,
                                                  optimality_weight=optimality_weight,
                                                  optimality_tol=tol))

    if len(valid_supercells) == 0:
        raise ValueError('No supercells that may host the specified '
                         'target_concentrations were supplied.')

    ens = TargetClusterVectorAnnealing(structure=valid_supercells, calculators=calculators,
                                       T_start=T_start, T_stop=T_stop,
                                       random_seed=random_seed)
    return ens.generate_structure(number_of_trial_steps=n_steps)


def generate_target_structure(cluster_space: ClusterSpace,
                              max_size: int,
                              target_concentrations: dict,
                              target_cluster_vector: List[float],
                              include_smaller_cells: bool = True,
                              pbc: Union[Tuple[bool, bool, bool], Tuple[int, int, int]] = None,
                              T_start: float = 5.0,
                              T_stop: float = 0.001,
                              n_steps: int = None,
                              optimality_weight: float = 1.0,
                              random_seed: int = None,
                              tol: float = 1e-5) -> Atoms:
    """
    Given a :attr:`cluster_space` and a :attr:`target_cluster_vector`, generate
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
        Cluster space defining the lattice to be occupied.
    max_size
        Maximum supercell size.
    target_concentrations
        Concentration of each species in the target structure, per
        sublattice (for example ``{'Au': 0.5, 'Pd': 0.5}`` for a
        single sublattice Au-Pd structure, or
        ``{'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'H': 0.25, 'X': 0.75}}``
        for a system with two sublattices.
        The symbols defining sublattices ('A', 'B' etc) can be
        found by printing the :attr:`cluster_space`.
    target_cluster_vector
        Cluster vector that the generated structure should match as
        closely as possible.
    include_smaller_cells
        If ``True``, search among all supercell sizes including
        :attr:`max_size`, else search only among those exactly matching
        :attr:`max_size`
    pbc
        Periodic boundary conditions for each direction, e.g.,
        ``(True, True, False)``. The axes are defined by
        the cell of :attr:`cluster_space.primitive_structure``.
        Default is periodic boundary in all directions.
    T_start
        Artificial temperature at which the simulated annealing starts.
    T_stop
        Artifical temperature at which the simulated annealing stops.
    n_steps
        Total number of Monte Carlo steps in the simulation.
    optimality_weight
        Controls weighting :math:`L` of perfect correlations, see
        :class:`mchammer.calculators.TargetVectorCalculator`.
    random_seed
        Seed for the random number generator used in the Monte Carlo simulation.
    tol
        Numerical tolerance.
    """
    target_concentrations = _validate_concentrations(target_concentrations, cluster_space)

    if pbc is None:
        pbc = (True, True, True)

    prim = cluster_space.primitive_structure
    prim.set_pbc(pbc)

    supercells = []
    if include_smaller_cells:
        sizes = list(range(1, max_size + 1))
    else:
        sizes = [max_size]

    for size in sizes:
        # For efficiency, make a first check that the current size is
        # commensurate with all concentrations (example: {'Au': 0.5,
        # 'Pd': 0.5} would not be commensurate with a supercell with 3
        # atoms).
        supercell = cluster_space.primitive_structure.repeat((size, 1, 1))
        if not _concentrations_fit_structure(structure=supercell,
                                             cluster_space=cluster_space,
                                             concentrations=target_concentrations):
            continue

        # Loop over all inequivalent supercells and intialize
        # them with a "random" occupation of symbols that
        # fulfill the target concentrations
        for supercell in enumerate_supercells(prim, [size]):
            supercell.set_pbc(True)
            supercells.append(supercell)

    return generate_target_structure_from_supercells(cluster_space=cluster_space,
                                                     supercells=supercells,
                                                     target_concentrations=target_concentrations,
                                                     target_cluster_vector=target_cluster_vector,
                                                     T_start=T_start,
                                                     T_stop=T_stop,
                                                     n_steps=n_steps,
                                                     optimality_weight=optimality_weight,
                                                     random_seed=random_seed,
                                                     tol=tol)


def generate_sqs_from_supercells(cluster_space: ClusterSpace,
                                 supercells: List[Atoms],
                                 target_concentrations: dict,
                                 T_start: float = 5.0,
                                 T_stop: float = 0.001,
                                 n_steps: int = None,
                                 optimality_weight: float = 1.0,
                                 random_seed: int = None,
                                 random_start: bool = True,
                                 tol: float = 1e-5) -> Atoms:
    """
    Given a :attr:`cluster_space`` and one or more :attr:`supercells`, generate
    a special quasirandom structure (SQS), i.e., a structure that for
    the provided supercells size provides the best possible
    approximation to a random alloy [ZunWeiFer90]_.

    In the present case, this means that the generated structure will
    have a cluster vector that as closely as possible matches the
    cluster vector of an infintely large randomly occupied supercell.
    Internally the function uses a simulated annealing algorithm and the
    difference between two cluster vectors is calculated with the
    measure suggested by A. van de Walle et al. in Calphad **42**, 13-18
    (2013) [WalTiwJon13]_ (for more information, see
    :class:`mchammer.calculators.TargetVectorCalculator`).

    Parameters
    ----------
    cluster_space
        Cluster space defining the lattice to be occupied.
    supercells
        List of one or more supercells among which an optimal
        structure will be searched for.
    target_concentrations
        Concentration of each species in the target structure, per
        sublattice (for example ``{'Au': 0.5, 'Pd': 0.5}`` for a
        single sublattice Au-Pd structure, or
        ``{'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'H': 0.25, 'X': 0.75}}``
        for a system with two sublattices.
        The symbols defining sublattices ('A', 'B' etc) can be
        found by printing the :attr:`cluster_space`.
    T_start
        Artificial temperature at which the simulated annealing starts.
    T_stop
        Artifical temperature at which the simulated annealing stops.
    n_steps
        Total number of Monte Carlo steps in the simulation.
    optimality_weight
        Controls weighting :math:`L` of perfect correlations, see
        :class:`mchammer.calculators.TargetVectorCalculator`.
    random_seed
        Seed for the random number generator used in the Monte Carlo simulation
        and used for initializing the occupation of the supercells if
        random_start is ``True``.
    random_start
        Randomly occupy starting structure, can be disabled
        if the user prefers to pass an initial structure.
    tol
        Numerical tolerance.
    """

    sqs_vector = _get_sqs_cluster_vector(cluster_space=cluster_space,
                                         target_concentrations=target_concentrations)
    return generate_target_structure_from_supercells(cluster_space=cluster_space,
                                                     supercells=supercells,
                                                     target_concentrations=target_concentrations,
                                                     target_cluster_vector=sqs_vector,
                                                     T_start=T_start, T_stop=T_stop,
                                                     n_steps=n_steps,
                                                     optimality_weight=optimality_weight,
                                                     random_seed=random_seed,
                                                     random_start=random_start,
                                                     tol=tol)


def generate_sqs(cluster_space: ClusterSpace,
                 max_size: int,
                 target_concentrations: dict,
                 include_smaller_cells: bool = True,
                 pbc: Union[Tuple[bool, bool, bool], Tuple[int, int, int]] = None,
                 T_start: float = 5.0,
                 T_stop: float = 0.001,
                 n_steps: int = None,
                 optimality_weight: float = 1.0,
                 random_seed: int = None,
                 tol: float = 1e-5) -> Atoms:
    """
    Given a :attr:`cluster_space`, generate a special quasirandom structure
    (SQS), i.e., a structure that for a given supercell size provides
    the best possible approximation to a random alloy [ZunWeiFer90]_.

    In the present case, this means that the generated structure will
    have a cluster vector that as closely as possible matches the
    cluster vector of an infintely large randomly occupied supercell.
    Internally the function uses a simulated annealing algorithm and the
    difference between two cluster vectors is calculated with the
    measure suggested by A. van de Walle et al. in Calphad **42**, 13-18
    (2013) [WalTiwJon13]_ (for more information, see
    :class:`mchammer.calculators.TargetVectorCalculator`).

    Parameters
    ----------
    cluster_space
        Cluster space defining the lattice to be occupied.
    max_size
        Maximum supercell size.
    target_concentrations
        Concentration of each species in the target structure, per
        sublattice (for example ``{'Au': 0.5, 'Pd': 0.5}`` for a
        single sublattice Au-Pd structure, or
        ``{'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'H': 0.25, 'X': 0.75}}``
        for a system with two sublattices.
        The symbols defining sublattices ('A', 'B' etc) can be
        found by printing the :attr:`cluster_space`.
    include_smaller_cells
        If ``True``, search among all supercell sizes including
        :attr:`max_size`, else search only among those exactly matching
        :attr:`max_size`
    pbc
        Periodic boundary conditions for each direction, e.g.,
        ``(True, True, False)``. The axes are defined by
        the cell of ``cluster_space.primitive_structure``.
        Default is periodic boundary in all directions.
    T_start
        Artificial temperature at which the simulated annealing starts.
    T_stop
        Artifical temperature at which the simulated annealing stops.
    n_steps
        Total number of Monte Carlo steps in the simulation.
    optimality_weight
        Controls weighting :math:`L` of perfect correlations, see
        :class:`mchammer.calculators.TargetVectorCalculator`.
    random_seed
        Seed for the random number generator used in the
        Monte Carlo simulation.
    tol
        Numerical tolerance.
    """

    sqs_vector = _get_sqs_cluster_vector(cluster_space=cluster_space,
                                         target_concentrations=target_concentrations)
    return generate_target_structure(cluster_space=cluster_space,
                                     max_size=max_size,
                                     target_concentrations=target_concentrations,
                                     target_cluster_vector=sqs_vector,
                                     include_smaller_cells=include_smaller_cells,
                                     pbc=pbc,
                                     T_start=T_start, T_stop=T_stop,
                                     n_steps=n_steps,
                                     optimality_weight=optimality_weight,
                                     random_seed=random_seed,
                                     tol=tol)


def generate_sqs_by_enumeration(cluster_space: ClusterSpace,
                                max_size: int,
                                target_concentrations: dict,
                                include_smaller_cells: bool = True,
                                pbc: Union[Tuple[bool, bool, bool], Tuple[int, int, int]] = None,
                                optimality_weight: float = 1.0,
                                tol: float = 1e-5) -> Atoms:
    """
    Given a :attr:`cluster_space`, generate a special quasirandom structure
    (SQS), i.e., a structure that for a given supercell size provides
    the best possible approximation to a random alloy [ZunWeiFer90]_.

    In the present case, this means that the generated structure will
    have a cluster vector that as closely as possible matches the
    cluster vector of an infintely large randomly occupied supercell.
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
        Cluster space defining the lattice to be occupied.
    max_size
        Maximum supercell size.
    target_concentrations
        Concentration of each species in the target structure, per
        sublattice (for example ``{'Au': 0.5, 'Pd': 0.5}`` for a
        single sublattice Au-Pd structure, or
        ``{'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'H': 0.25, 'X': 0.75}}``
        for a system with two sublattices.
        The symbols defining sublattices ('A', 'B' etc) can be
        found by printing the :attr:`cluster_space`.
    include_smaller_cells
        if ``True`` search among all supercell sizes including
        :attr:`max_size`, else search only among those exactly matching
        :attr:`max_size`.
    pbc
        Periodic boundary conditions for each direction, e.g.,
        ``(True, True, False)``. The axes are defined by
        the cell of ``cluster_space.primitive_structure``.
        Default is periodic boundary in all directions.
    optimality_weight
        Controls weighting :math:`L` of perfect correlations, see
        :class:`mchammer.calculators.TargetVectorCalculator`.
    tol
        Numerical tolerance.
    """
    target_concentrations = _validate_concentrations(target_concentrations, cluster_space)
    sqs_vector = _get_sqs_cluster_vector(cluster_space=cluster_space,
                                         target_concentrations=target_concentrations)
    # Translate concentrations to the format required for concentration
    # restricted enumeration
    cr: Dict[str, tuple] = {}
    sublattices = cluster_space.get_sublattices(cluster_space.primitive_structure)
    for sl in sublattices:
        mult_factor = len(sl.indices) / len(cluster_space.primitive_structure)
        if sl.symbol in target_concentrations:
            sl_conc = target_concentrations[sl.symbol]
        else:
            sl_conc = {sl.chemical_symbols[0]: 1.0}
        for species, value in sl_conc.items():
            c = value * mult_factor
            if species in cr:
                cr[species] = (cr[species][0] + c,
                               cr[species][1] + c)
            else:
                cr[species] = (c, c)

    # Check to be sure...
    c_sum = sum(c[0] for c in cr.values())
    assert abs(c_sum - 1) < tol  # Should never happen, but...

    as_list = cluster_space.as_list
    best_score = 1e9

    if include_smaller_cells:
        sizes = list(range(1, max_size + 1))
    else:
        sizes = [max_size]

    # Prepare primitive structure with the right boundary conditions
    prim = cluster_space.primitive_structure
    if pbc is None:
        pbc = (True, True, True)
    prim.set_pbc(pbc)

    # Enumerate and calculate score for each structuer
    for structure in enumerate_structures(prim,
                                          sizes,
                                          cluster_space.chemical_symbols,
                                          concentration_restrictions=cr):
        cv = cluster_space.get_cluster_vector(structure)
        score = compare_cluster_vectors(cv_1=cv, cv_2=sqs_vector,
                                        as_list=as_list,
                                        optimality_weight=optimality_weight,
                                        tol=tol)

        if score < best_score:
            best_score = score
            best_structure = structure
    return best_structure


def occupy_structure_randomly(structure: Atoms, cluster_space: ClusterSpace,
                              target_concentrations: dict,
                              random_seed: int = None) -> None:
    """
    Occupy a structure with quasirandom order but fulfilling
    :attr:`target_concentrations`.

    Parameters
    ----------
    structure
        ASE Atoms object that will be occupied randomly.
    cluster_space
        Cluster space (needed as it carries information about sublattices).
    target_concentrations
        Concentration of each species in the target structure, per
        sublattice (for example ``{'Au': 0.5, 'Pd': 0.5}`` for a
        single sublattice Au-Pd structure, or
        ``{'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'H': 0.25, 'X': 0.75}}``
        for a system with two sublattices.
        The symbols defining sublattices ('A', 'B' etc) can be
        found by printing the :attr:`cluster_space`.
    random_seed
        Seed for the random number generator.
    """
    rng = np.random.default_rng(random_seed)
    target_concentrations = _validate_concentrations(cluster_space=cluster_space,
                                                     concentrations=target_concentrations)

    if not _concentrations_fit_structure(structure, cluster_space, target_concentrations):
        raise ValueError('Structure with {} atoms cannot accomodate '
                         'target concentrations {}'.format(len(structure),
                                                           target_concentrations))

    symbols_all = [''] * len(structure)
    for sl in cluster_space.get_sublattices(structure):
        symbols: List[str] = []  # chemical_symbols in one sublattice
        chemical_symbols = sl.chemical_symbols
        if len(chemical_symbols) == 1:
            symbols += [chemical_symbols[0]] * len(sl.indices)
        else:
            sl_conc = target_concentrations[sl.symbol]
            for chemical_symbol in sl.chemical_symbols:
                n_symbol = int(round(len(sl.indices) * sl_conc[chemical_symbol]))
                symbols += [chemical_symbol] * n_symbol

        # Should not happen but you never know
        assert len(symbols) == len(sl.indices)

        # Shuffle to introduce randomness
        rng.shuffle(symbols)

        # Assign symbols to the right indices
        for symbol, lattice_site in zip(symbols, sl.indices):
            symbols_all[lattice_site] = symbol

    assert symbols_all.count('') == 0
    structure.set_chemical_symbols(symbols_all)


def _validate_concentrations(concentrations: dict,
                             cluster_space: ClusterSpace,
                             tol: float = 1e-5) -> dict:
    """
    Validates concentration specification against a cluster space
    (raises `ValueError` if they do not match).

    Parameters
    ----------
    concentrations
        Concentration specification.
    cluster_space
        Cluster space to check against.
    tol
        Numerical tolerance.

    Returns
    -------
        An adapted version of concentrations, which is always a dictionary
        even if there is only one sublattice.
    """
    sls = cluster_space.get_sublattices(cluster_space.primitive_structure)

    if not isinstance(list(concentrations.values())[0], dict):
        concentrations = {'A': concentrations}

    # Ensure concentrations sum to 1 at each sublattice
    for sl_conc in concentrations.values():
        conc_sum = sum(list(sl_conc.values()))
        if abs(conc_sum - 1.0) > tol:
            raise ValueError('Concentrations must sum up '
                             'to 1 for each sublattice (not {})'.format(conc_sum))

    # Symbols need to match on each sublattice
    for sl in sls:
        if sl.symbol not in concentrations:
            if len(sl.chemical_symbols) > 1:
                raise ValueError('A sublattice ({}: {}) is missing in '
                                 'target_concentrations'.format(sl.symbol,
                                                                list(sl.chemical_symbols)))
        else:
            sl_conc = concentrations[sl.symbol]
            if tuple(sorted(sl.chemical_symbols)) != tuple(sorted(list(sl_conc.keys()))):
                raise ValueError('Chemical symbols on a sublattice ({}: {}) are '
                                 'not the same as those in the specified '
                                 'concentrations {}'.format(sl.symbol, list(sl.chemical_symbols),
                                                            list(sl_conc.keys())))

    return concentrations


def _concentrations_fit_structure(structure: Atoms,
                                  cluster_space: ClusterSpace,
                                  concentrations: Dict[str, Dict[str, float]],
                                  tol: float = 1e-5) -> bool:
    """
    Check if specified concentrations are commensurate with a
    certain supercell (including sublattices).

    Parameters
    ----------
    structure
        Atomic configuration to be checked.
    cluster_space
        Corresponding cluster space.
    concentrations
        Which concentrations, per sublattice, e.g., ``{'A': {'Ag': 0.3, 'Au': 0.7}}``.
    tol
        Numerical tolerance.
    """
    # Check that concentrations are OK in each sublattice
    for sublattice in cluster_space.get_sublattices(structure):
        if sublattice.symbol in concentrations:
            sl_conc = concentrations[sublattice.symbol]
            for conc in sl_conc.values():
                n_symbol = conc * len(sublattice.indices)
                if abs(int(round(n_symbol)) - n_symbol) > tol:
                    return False
    return True


def _get_sqs_cluster_vector(cluster_space: ClusterSpace,
                            target_concentrations: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Get the SQS vector for a certain cluster space and certain
    concentration. Here SQS vector refers to the cluster vector of an
    infintely large supercell with random occupation.

    Parameters
    ----------
    cluster_space
        The kind of lattice to be occupied.
    target_concentrations
        Concentration of each species in the target structure,
        per sublattice (for example `{'A': {'Ag': 0.5, 'Pd': 0.5}}`).
    """
    target_concentrations = _validate_concentrations(concentrations=target_concentrations,
                                                     cluster_space=cluster_space)

    sublattice_to_index = {letter: index for index,
                           letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    all_sublattices = cluster_space.get_sublattices(
        cluster_space.primitive_structure)

    # Make a map from chemical symbol to integer, later on used
    # for evaluating cluster functions.
    # Internally, icet sorts species according to atomic numbers.
    # Also check that each symbol only occurs in one sublattice.
    symbol_to_integer_map = {}
    found_species: List[str] = []
    for sublattice in all_sublattices:
        if len(sublattice.chemical_symbols) < 2:
            continue
        atomic_numbers = [periodic_table.index(sym) for sym in sublattice.chemical_symbols]
        for i, species in enumerate(sorted(atomic_numbers)):
            found_species.append(species)
            symbol_to_integer_map[periodic_table[species]] = i

    # Target concentrations refer to all atoms, but probabilities only
    # to the sublattice.
    probabilities = {}
    for sl_conc in target_concentrations.values():
        if len(sl_conc) == 1:
            continue
        for symbol in sl_conc.keys():
            probabilities[symbol] = sl_conc[symbol]

    # For every orbit, calculate average cluster function
    cv = [1.0]
    for orbit in cluster_space.as_list:
        if orbit['order'] < 1:
            continue

        # What sublattices are there in this orbit?
        sublattices = [all_sublattices[sublattice_to_index[letter]]
                       for letter in orbit['sublattices']]

        # What chemical symbols do these sublattices refer to?
        symbol_groups = [sublattice.chemical_symbols for sublattice in sublattices]

        # How many allowed species in each of those sublattices?
        nbr_of_allowed_species = [len(symbol_group)
                                  for symbol_group in symbol_groups]

        # Calculate contribution from every possible combination of
        # symbols weighted with their probability
        cluster_product_average = 0
        for symbols in itertools.product(*symbol_groups):
            cluster_product = 1
            for i, symbol in enumerate(symbols):
                mc_vector_component = orbit['multicomponent_vector'][i]
                species_i = symbol_to_integer_map[symbol]
                prod = cluster_space.evaluate_cluster_function(nbr_of_allowed_species[i],
                                                               mc_vector_component,
                                                               species_i)
                cluster_product *= probabilities[symbol] * prod
            cluster_product_average += cluster_product
        cv.append(cluster_product_average)
    return np.array(cv)
