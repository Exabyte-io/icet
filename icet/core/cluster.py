"""
This module provides the Cluster class.
"""

from _icet import Cluster
from icet.core.lattice_site import LatticeSite
from icet.core.structure import Structure
__all__ = ['Cluster']


def _from_python(atoms, lattice_sites, sort=False, clusterIndex=-1):
    """
    Construct a cluster from atoms and python lattice sites
    """

    structure = Structure.from_atoms(atoms)

    lattice_sites_cpp = [
        LatticeSite(ls.index, ls.unitcell_offset) for ls in lattice_sites]

    return Cluster(structure, lattice_sites_cpp, sort, clusterIndex)


Cluster.from_python = _from_python
