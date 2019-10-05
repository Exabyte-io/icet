import time
from ase.build import bulk
from icet.core.permutation_matrix import permutation_matrix_from_structure

from icet.core.permutation_matrix import _get_lattice_site_permutation_matrix \
    as get_lattice_site_permutation_matrix


if __name__ == '__main__':

    structure = bulk('Al')
    cutoff = 15

    start = time.process_time()
    pm, prim_structure, _ = permutation_matrix_from_structure(structure, cutoff)

    pm_lattice_sites = get_lattice_site_permutation_matrix(  # noqa
        prim_structure, pm, prune=True)
    elapsed_time = time.process_time() - start

    print('Time to initialize permutation matrix with cutoff: {}, {:.6} sec'
          .format(cutoff, elapsed_time))
