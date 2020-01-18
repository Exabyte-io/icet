import time
from ase.build import bulk
from icet.core.matrix_of_equivalent_positions import matrix_of_equivalent_positions_from_structure

from icet.core.matrix_of_equivalent_positions import \
    _get_lattice_site_matrix_of_equivalent_positions \
    as get_lattice_site_matrix_of_equivalent_positions


if __name__ == '__main__':

    structure = bulk('Al')
    cutoff = 15
    position_tolerance = 1e-5
    fractional_position_tolerance = 2e-6
    symprec = 1e-5
    start = time.process_time()
    pm, prim_structure, _ = matrix_of_equivalent_positions_from_structure(
        structure, cutoff, position_tolerance, symprec)

    pm_lattice_sites = get_lattice_site_matrix_of_equivalent_positions(
        prim_structure, pm, fractional_position_tolerance, prune=True)
    elapsed_time = time.process_time() - start

    print('Time to initialize matrix of equivalent positions with cutoff: {}, {:.6} sec'
          .format(cutoff, elapsed_time))
