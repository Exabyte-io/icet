import time
from ase.build import bulk
import ase.neighborlist as asenl
from icet import Structure
from icet.core.neighbor_list import NeighborList


def build_neighbor_list_cpp(structure, cutoff):
    """
    Build a neighbor list using the function implemented in C++.
    """
    nl = NeighborList(cutoff)
    nl.build(structure)


def build_neighbor_list_python(atoms, cutoff):
    """
    Build a neighbor list using the Python implementation from ASE.
    """
    ase_nl = asenl.NeighborList(len(atoms) * [cutoff / 2.0], skin=1e-8,
                                bothways=True, self_interaction=False)
    ase_nl.update(atoms)


if __name__ == '__main__':

    atoms = bulk('Al').repeat(3)
    structure = Structure.from_atoms(atoms)
    cutoff = 10
    print('Cutoff: {:.3f}'.format(cutoff))
    print('Number of atoms: {}'.format(len(atoms)))

    t = time.process_time()
    build_neighbor_list_cpp(structure, cutoff)
    elapsed_time_cpp = time.process_time() - t
    print('Timing C++: {:.6f} sec'.format(elapsed_time_cpp))

    t = time.process_time()
    build_neighbor_list_python(atoms, cutoff)
    elapsed_time_python = time.process_time() - t
    print('Timing Python (ASE): {:.6f} s'.format(elapsed_time_python))

    print('C++ speedup: {:.3f}'.format(elapsed_time_python / elapsed_time_cpp))
