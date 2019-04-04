from mchammer.ensembles.structure_annealing import TargetClusterVectorAnnealing
from mchammer.calculators.target_vector_calculator import TargetVectorCalculator
from icet.tools import enumerate_supercells
from icet import ClusterSpace
from ase.build import bulk
import numpy as np
import random


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
        ok_size = True
        for symbol, conc in target_concentrations.items():
            n_symbol = conc * natoms
            if abs(int(round(n_symbol)) - n_symbol) > tol:
                ok_size = False
        if not ok_size:
            continue

        # Loop over all inequivalent supercells and intialize
        # them with a "random" occupation of symbols that
        # fulfill the target concentrations
        for supercell in enumerate_supercells(atoms, [size]):
            n_atoms = len(supercell)
            # Will hold chemical_symbols of all sublattices
            symbols_all = [0] * len(supercell)
            for sublattice in cs.get_sublattices(supercell):
                symbols = []  # chemical_symbols in one sublattice
                for chemical_symbol in sublattice.chemical_symbols:
                    n_symbol = int(
                        round(n_atoms * target_concentrations[chemical_symbol]))
                    symbols += [chemical_symbol] * n_symbol

                # If any concentration was not commensurate with
                if len(symbols) != len(sublattice.indices):
                    raise ValueError('target_concentrations {} do not match '
                                     'sublattice sizes'.format(target_concentrations))

                # Shuffle because it will probably make it more SQS-like
                random.shuffle(symbols)

                # Assign symbols to the right indices
                for symbol, lattice_site in zip(symbols, sublattice.indices):
                    symbols_all[lattice_site] = symbol

            assert symbols_all.count(0) == 0
            supercell.set_chemical_symbols(symbols_all)
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

    sqs_vector = get_sqs_vector(cluster_space=cluster_space,
                                cluster_space=target_concentrations)

    return generate_target_structure(cluster_space=cluster_space,
                                     maxsize=maxsize,
                                     target_concentrations=target_concentrations,
                                     target_vector=sqs_vector,
                                     T_start=T_start, T_stop=T_stop,
                                     n_steps=n_steps,
                                     random_seed=random_seed,
                                     tol=tol)


#def get_sqs_vector(cluster_space: ClusterSpace,
#                   target_concentrations: dict):
    



if __name__ == '__main__':
    from ase import Atom
    atoms = bulk('Au', a=4.0)

    target_cluster_vector = [1.0] + [0.0] * (len(cs) - 1)
    #atoms.append(Atom('H', position=(2, 2, 2)))
    cs = ClusterSpace(atoms, [8.0], [['Au', 'Pd']])
    maxsize = 10
    target_concentrations = {'Au': 0.5,
                             'Pd': 0.5, }  # 'H': 0.1 / 2, 'V': 0.9 / 2}
    sqs = generate_sqs(cluster_space=cs,
                       maxsize=maxsize,
                       target_concentrations=target_concentrations)
    print(sqs)
    print(cs.get_cluster_vector(sqs))
