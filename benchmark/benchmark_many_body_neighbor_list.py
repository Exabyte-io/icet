import time
from ase.build import bulk
import ase.neighborlist as asenl
from icet import Structure
from icet.core.neighbor_list import NeighborList
from icet.core_py.many_body_neighbor_list \
    import ManyBodyNeighborList as TestManyBodyNeighborList
from icet.core.many_body_neighbor_list import ManyBodyNeighborList


def build_many_body_neighbor_list_cpp(structure, order, cutoff):
    '''
    Build a many-body neighbor list up to `order` using the neighbor list
    implemented in C++.
    '''
    cutoffs = (order - 1)*[cutoff]
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


def build_many_body_neighbor_list_python(atoms, order, cutoff):
    '''
    Build a many-body neighbor list up to `order` based on the
    Python implementation from ASE.
    '''
    mbnl_T = TestManyBodyNeighborList()
    ase_nl = asenl.NeighborList(len(atoms) * [cutoff / 2.0], skin=1e-8,
                                bothways=True, self_interaction=False)
    ase_nl.update(atoms)
    cutoffs = (order - 1)*[cutoff]
    neighbor_lists = []
    for co in cutoffs:
        ase_nl.update(structure)
        neighbor_lists.append(ase_nl)

    bothways = False
    t = time.process_time()
    for i in range(len(atoms)):
        mbnl_T.build(neighbor_lists, i, bothways=bothways)
    elapsed_time = time.process_time() - t
    return elapsed_time


if __name__ == "__main__":

    order = 3
    cutoff = 10.0
    atoms = bulk('Al').repeat(2)
    structure = Structure.from_atoms(atoms)
    print('Cutoff: {:.3f}'.format(cutoff))
    print('Order: {:}'.format(order))
    print('Number of atoms: {}'.format(len(atoms)))

    t = time.process_time()
    mbnl_time_cpp = build_many_body_neighbor_list_cpp(structure, order, cutoff)
    elapsed_time_cpp = time.process_time() - t
    print('Timing C++: {:.6f} sec'.format(elapsed_time_cpp))

    t = time.process_time()
    mbnl_time_py = build_many_body_neighbor_list_python(atoms, order, cutoff)
    elapsed_time_python = time.process_time() - t
    print('Timing Python (ASE): {:.6f} s'.format(elapsed_time_python))
    print('C++ speedup: {:.3f}'.format(elapsed_time_python / elapsed_time_cpp))
    print('C++ speedup (only mbnl build): {:.3f}'
          .format(mbnl_time_py / mbnl_time_cpp))
