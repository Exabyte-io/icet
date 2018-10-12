from icet.core.cluster_space import ClusterSpace as ClusterSpace_cpp
from icet.core_py.cluster_space import ClusterSpace as ClusterSpace_py

from ase.build import bulk

import time


def setup_cluster_space_py(atoms, cutoffs, species):
    """
    Initialize the python version of the cluster space object.

    Parameters
    ----------
    atoms : ase Atoms object
    cutoffs : list of floats
    species : list of str
    """

    t = time.process_time()
    cs = ClusterSpace_py(atoms, cutoffs, species)  # noqa
    elapsed_time = time.process_time() - t
    return elapsed_time


def setup_cluster_space_cpp(atoms, cutoffs, species):
    """
    Initialize the C++ version of the cluster space object.

    Parameters
    ----------
    atoms : ase Atoms object
    cutoffs : list of floats
    species : list of str
    """

    t = time.process_time()
    cs = ClusterSpace_cpp(atoms, cutoffs, species)  # noqa
    elapsed_time = time.process_time() - t
    return elapsed_time


if __name__ == '__main__':

    atoms = bulk('Al')

    cutoff = [10, 7, 6]
    species = ['Al', 'Ti']
    python_time = setup_cluster_space_py(atoms, cutoff, species)

    cpp_time = setup_cluster_space_cpp(atoms, cutoff, species)
    print('Time to initialize ClusterSpace in pure '
          'python with cutoff: {}, {}s '.format(
              cutoff, python_time))
    print('Time to initialize ClusterSpace in C++/Python with cutoff: {}, {}s '
          .format(cutoff, cpp_time))
    print('Speed up in C++ {}'.format(python_time / cpp_time))
