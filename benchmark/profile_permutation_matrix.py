import time
from ase.build import bulk
from icet.core.permutation_matrix import permutation_matrix_from_structure

from icet.core.permutation_matrix import _get_lattice_site_permutation_matrix \
    as get_lattice_site_permutation_matrix


if __name__ == '__main__':

    structure = bulk('Al')
    cutoff = 15
    position_tolerance = 1e-5
    fractional_position_tolerance = 2e-6
    symprec = 1e-5
    start = time.process_time()
    pm, prim_structure, _ = permutation_matrix_from_structure(structure, cutoff,
                                                              position_tolerance, symprec)

    pm_lattice_sites = get_lattice_site_permutation_matrix(  # noqa
        prim_structure, pm, fractional_position_tolerance, prune=True)
    elapsed_time = time.process_time() - start

    print('Time to initialize permutation matrix with cutoff: {}, {:.6} sec'
          .format(cutoff, elapsed_time))
