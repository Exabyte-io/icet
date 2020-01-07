
from icet.core.orbit_list import OrbitList
from ase.build import bulk
import time


if __name__ == '__main__':

    structure = bulk('Al')
    cutoffs = [10, 7, 6]
    symprec = 1e-5
    position_tolerance = 1e-5
    fractional_position_tolerance = 2e-6

    t = time.process_time()
    orbit = OrbitList(structure, cutoffs, symprec, position_tolerance,
                      fractional_position_tolerance)  # noqa
    elapsed_time = time.process_time() - t

    print('Time to initialize OrbitList with cutoffs: {}, {:.6} sec'
          .format(cutoffs, elapsed_time))
