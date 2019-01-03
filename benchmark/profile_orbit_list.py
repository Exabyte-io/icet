
from icet.core.orbit_list import OrbitList
from ase.build import bulk
import time


if __name__ == '__main__':

    atoms = bulk('Al')
    cutoffs = [10, 7, 6]

    t = time.process_time()
    orbit = OrbitList(atoms, cutoffs)  # noqa
    elapsed_time = time.process_time() - t

    print('Time to initialize OrbitList with cutoffs: {}, {:.6} sec'
          .format(cutoffs, elapsed_time))
