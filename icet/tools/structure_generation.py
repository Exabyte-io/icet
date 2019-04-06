from mchammer.ensembles.structure_annealing import TargetClusterVectorAnnealing
from mchammer.calculators.target_vector_calculator import TargetVectorCalculator
from icet.tools import enumerate_supercells
from icet import ClusterSpace
from ase.build import bulk
from ase.data import chemical_symbols as periodic_table
import numpy as np
import random
from typing import List
from ase import Atoms
import itertools


def generate_target_structure(cluster_space: ClusterSpace, maxsize: int,
                              target_concentrations: dict,
                              target_cluster_vector: List[float],
                              T_start: float = 0.5, T_stop: float = 0.001,
                              n_steps: float = None,
                              random_seed: int = None,
                              tol: float = 1e-5):

    if abs(sum(list(target_concentrations.values())) - 1.0) > tol:
        raise ValueError('Target concentration must be specified for all values '
                         'and they must sum up to 1')

    supercells = []
    calculators = []
    for size in range(1, maxsize + 1):
        # Check that the current size is commensurate
        # with all concentration (example: {'Au': 0.5, 'Pd': 0.5}
        # would not be commensurate with a supercell with 3 atoms
        natoms = size * len(cluster_space.primitive_structure)
        if not concentrations_fit_size(natoms, target_concentrations):
            continue

        # Loop over all inequivalent supercells and intialize
        # them with a "random" occupation of symbols that
        # fulfill the target concentrations
        for supercell in enumerate_supercells(atoms, [size]):
            decorate_atoms_randomly(supercell, target_concentrations)
            supercells.append(supercell)
            calculators.append(TargetVectorCalculator(supercell, cluster_space,
                                                      target_cluster_vector))

    ens = TargetClusterVectorAnnealing(atoms=supercells, calculators=calculators,
                                       T_start=T_start, T_stop=T_stop,
                                       random_seed=random_seed)
    return ens.generate_structure(number_of_trial_steps=n_steps)


def generate_sqs(cluster_space: ClusterSpace, maxsize: int,
                 target_concentrations: dict,
                 T_start: float = 0.5, T_stop: float = 0.001,
                 n_steps: float = None,
                 random_seed: int = None,
                 tol: float = 1e-5):

    sqs_vector = get_random_vector(cluster_space=cluster_space,
                                   target_concentrations=target_concentrations)
    exit(0)
    return generate_target_structure(cluster_space=cluster_space,
                                     maxsize=maxsize,
                                     target_concentrations=target_concentrations,
                                     target_vector=sqs_vector,
                                     T_start=T_start, T_stop=T_stop,
                                     n_steps=n_steps,
                                     random_seed=random_seed,
                                     tol=tol)


def decorate_atoms_randomly(atoms: Atoms, target_concentrations: dict):
    n_atoms = len(atoms)

    # symbols_all will hold chemical_symbols of all sublattices
    symbols_all = [0] * len(atoms)
    for sublattice in cs.get_sublattices(atoms):
        print(len(sublattice.indices))
        symbols = []  # chemical_symbols in one sublattice
        for chemical_symbol in sublattice.chemical_symbols:
            print(chemical_symbol)
            n_symbol = int(
                round(n_atoms * target_concentrations[chemical_symbol]))
            symbols += [chemical_symbol] * n_symbol

        print(len(symbols))
        # If any concentration was not commensurate with the sublattice,
        # raise an exception
        if len(symbols) != len(sublattice.indices):
            raise ValueError('target_concentrations {} do not match '
                             'sublattice sizes'.format(target_concentrations))

        # Shuffle to introduce some randomness
        random.shuffle(symbols)

        # Assign symbols to the right indices
        for symbol, lattice_site in zip(symbols, sublattice.indices):
            symbols_all[lattice_site] = symbol

    assert symbols_all.count(0) == 0
    atoms.set_chemical_symbols(symbols_all)


def concentrations_fit_size(size, concentrations, tol=1e-5):
    for symbol, conc in target_concentrations.items():
        n_symbol = conc * size
        if abs(int(round(n_symbol)) - n_symbol) > tol:
            return False
    return True


def get_random_vector(cluster_space: ClusterSpace,
                      target_concentrations: dict,
                      supercell_size: int = 3000,
                      number_of_averages: int = 10):
    """
    Generate cluster vector for a randomly decorated supercell fulfilling
    target_concentrations.
    """
    primitive_structure = cluster_space.primitive_structure

    # Heuristic for determining supercell size
    # Make a supercell with roughly 1000 atoms
    repeat = int(np.ceil((supercell_size / len(primitive_structure))**(1 / 3)))
    repeat = max(2, repeat)

    # The chosen cell size may not be commensurate with the target
    # concentrations
    while not concentrations_fit_size(len(primitive_structure) * repeat**3,
                                      target_concentrations):
        repeat += 1

    supercell = primitive_structure.repeat(repeat)

    # Decorate the supercell a number of times and return average cluster
    # vector
    cv = np.zeros(len(cluster_space))
    for _ in range(number_of_averages):
        # print(supercell)
        decorate_atoms_randomly(supercell, target_concentrations)
        # print(cs.get_cluster_vector(supercell))
        cv += cs.get_cluster_vector(supercell)

    cv /= number_of_averages
    return cv


def get_sqs_cluster_vector(cluster_space, target_concentrations):
    sublattice_to_index = {letter: index for index,
                           letter in enumerate('ABCDEFGHIJKLMNOPQ')}
    all_sublattices = cluster_space.get_sublattices(cs.primitive_structure)

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
                                 'sublattice, {} occurred more than once.'.format(periodic_table[species]))
            found_species.append(species)
            symbol_to_integer_map[periodic_table[species]] = i

    # Target concentrations refer to all atoms, but probabilities only
    # to the sublattice.
    probabilities = {}
    for sublattice in all_sublattices:
        mult_factor = len(cluster_space.primitive_structure) / len(sublattice.indices)
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

        # Calculate contribtion from every possible combination of symbols
        # weighted with their probability
        cluster_product_average = 0
        count = 0
        for symbols in itertools.product(*symbol_groups):
            cluster_product = 1
            for i, symbol in enumerate(symbols):
                mc_vector_component = orbit['multi_component_vector'][i]
                species_i = symbol_to_integer_map[symbol]
                prod = cs.evaluate_cluster_function(nbr_of_allowed_species[i],
                                                    mc_vector_component,
                                                    species_i)
                cluster_product *= probabilities[symbol] * prod
            cluster_product_average += cluster_product
        cv.append(cluster_product_average)
    return np.array(cv)

if __name__ == '__main__':
    from ase import Atom
    atoms = bulk('Au', a=4.0)
    atoms.append(Atom('H', position=(2, 2, 2)))

    
    
    #atoms.append(Atom('H', position=(2, 2, 2)))
    cs = ClusterSpace(atoms, [6.0], [['Au', 'Pd', 'Cu'], ['H', 'V']])

    maxsize = 10
    target_concentrations = {'Au': 0.6666666667 / 6,
                             'Pd': 1.333333333 / 6,
                             'Cu': 1 / 6,
                             'H': 2.5/6,
                             'V': 0.5/6,
                             }  # 'H': 1 / 4, 'V': 1 / 4}
    cv = get_sqs_cluster_vector(cs, target_concentrations)
    print(cv)
    cv_random = get_random_vector(cs, target_concentrations, supercell_size=2000)
    cv_diff = cv - cv_random

    print(cv_random)
    print(cv)
    print(max(np.abs(cv_diff)))

    exit(0)
    sqs = generate_sqs(cluster_space=cs,
                       maxsize=maxsize,
                       target_concentrations=target_concentrations)

    print(sqs)
    print(cs.get_cluster_vector(sqs))
