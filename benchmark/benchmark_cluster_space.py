import time
from ase.build import bulk
from icet.core.cluster_space import ClusterSpace as ClusterSpace_cpp


if __name__ == '__main__':

    structure = bulk('Al')

    cutoffs = [10, 7, 6]
    chemical_symbols = ['Al', 'Ti']

    start = time.process_time()
    cs = ClusterSpace_cpp(structure, cutoffs, chemical_symbols)  # noqa
    elapsed_time = time.process_time() - start

    print('Time to initialize ClusterSpace in with cutoffs: {}, {:.6} sec'
          .format(cutoffs, elapsed_time))
