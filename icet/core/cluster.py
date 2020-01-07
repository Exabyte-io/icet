"""
This module provides the Cluster class.
"""

from typing import List
from ase import Atoms
from _icet import Cluster
from icet.core.lattice_site import LatticeSite
from icet.core.structure import Structure

__all__ = ['Cluster']


def _from_python(ase_structure: Atoms,
                 lattice_sites: List[LatticeSite],
                 cluster_index: int = -1):
    """
    Constructs a cluster from an ASE Atoms object and Python lattice sites.

    Parameters
    ----------
    ase_structure
        structure as ASE Atoms object
    lattice_sites
        lattice site objects
    cluster_index
        index used to identify cluster
    """

    structure = Structure.from_atoms(ase_structure)

    lattice_sites_cpp = [
        LatticeSite(ls.index, ls.unitcell_offset) for ls in lattice_sites]

    return Cluster(structure, lattice_sites_cpp, cluster_index)


Cluster.from_python = _from_python
