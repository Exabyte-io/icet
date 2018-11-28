from mchammer.calculators import ClusterExpansionCalculator
from icet import ClusterSpace, ClusterExpansion
from ase.build import bulk
import numpy as np
import time


def time_local_energy(calculator, iters):
    """ Returns timing for local energy calculations. """
    indices = [2]
    occupations = calculator.atoms.numbers
    t0 = time.time()
    for _ in range(iters):
        calculator.calculate_local_contribution(local_indices=indices,
                                                occupations=occupations)
    t1 = time.time() - t0
    return t1


def time_total_energy(calculator, iters):
    """ Returns timing for total energy calculations. """
    occupations = calculator.atoms.numbers
    t0 = time.time()
    for _ in range(iters):
        calculator.calculate_total(occupations=occupations)
    t1 = time.time() - t0
    return t1


def print_timing_ratios(atoms, iters, sizes, cutoffs):
    """ Prints timing ratios between local and total energy calculations. """

    print('# $1 size, $2 iters, $3 atom size, $4 ce calc init time (s)'
          ', $5 t_local, $6 t_total, $7 t_total/t_local')
    cs = ClusterSpace(atoms, cutoffs, chemical_symbols=['Al', 'Ga'])
    parameters = np.array([1.2 for _ in range(len(cs))])
    ce = ClusterExpansion(cs, parameters)
    for size in sizes:
        atoms_cpy = atoms.repeat(size)
        t0 = time.time()
        calculator = ClusterExpansionCalculator(atoms_cpy, ce)
        time_ce_init = time.time() - t0

        t_local = time_local_energy(calculator, iters)
        t_total = time_total_energy(calculator, iters)
        print(size, iters, len(atoms_cpy), time_ce_init,
              t_local, t_total, t_total / t_local)


if __name__ == '__main__':
    iters = 2
    atoms = bulk('Al')
    cutoffs = [10, 6, 5]
    chemical_symbols = ['Al', 'Ga']
    sizes = [4, 6, 8]
    print_timing_ratios(atoms, iters, sizes, cutoffs)

    cs = ClusterSpace(atoms, cutoffs, chemical_symbols)
    atoms = atoms.repeat(5)
    parameters = np.array([1.2 for _ in range(len(cs))])
    ce = ClusterExpansion(cs, parameters)
    print('Constructing CE calculator')
    t0 = time.time()
    calculator = ClusterExpansionCalculator(atoms, ce)
    print('Done constructing CE calculator in {:.5f}s'
          .format(time.time() - t0))
    t_local = time_local_energy(calculator, iters)
    t_total = time_total_energy(calculator, iters)
    print('Number of sites: {}'.format(len(calculator.atoms)))
    print('Timing of local energy calculation: {:.5f}'.format(t_local))
    print('Timing of total energy calculation: {:.5f}'.format(t_total))
    print('Speed up for local calculator: {:.2f} '.format(t_total / t_local))

    print('Timing for calculating {} MC cycles ({} sites) local energies {}s'
          .format(iters / len(atoms), len(atoms),
                  t_local * len(calculator.atoms)))
