import time
from ase.build import bulk
from icet import Structure
from icet.core.neighbor_list import NeighborList
from icet.core.many_body_neighbor_list import ManyBodyNeighborList


def build_many_body_neighbor_list_cpp(structure, order, cutoff):
    """
    Builds a many-body neighbor list up to `order` using the neighbor list
    implemented in C++.
    """
    cutoffs = (order - 1) * [cutoff]
    neighbor_lists = []

    for co in cutoffs:
        nl = NeighborList(co)
        nl.build(structure)
        neighbor_lists.append(nl)

    t = time.process_time()
    mbnl = ManyBodyNeighborList()
    for i in range(len(structure)):
        mbnl.build(neighbor_lists, i, False)
    elapsed_time = time.process_time() - t
    return elapsed_time


if __name__ == '__main__':

    order = 3
    cutoff = 10.0
    structure = bulk('Al').repeat(2)
    structure = Structure.from_atoms(structure)
    print('Cutoff: {:.3f}'.format(cutoff))
    print('Order: {:}'.format(order))
    print('Number of atoms: {}'.format(len(structure)))

    t = time.process_time()
    mbnl_time_cpp = build_many_body_neighbor_list_cpp(structure,
                                                      order, cutoff)
    elapsed_time_cpp = time.process_time() - t
    print('Time for constructing many-body neighbor list: {:.6f} sec'
          .format(elapsed_time_cpp))
