"""
This module provides the Structure class.
"""
from typing import List, Sequence
from ase import Atom, Atoms

from _icet import _Structure
from .lattice_site import LatticeSite


class Structure(_Structure):
    """
    This class stores the cell metric, positions, chemical symbols,
    and periodic boundary conditions that describe a structure. It
    also holds information pertaining to the components that are
    allowed on each site and provides functionality for computing
    distances between sites.

    Note
    ----
    As a user you will usually not interact directly with objects of this type.

    Parameters
    ----------
    positions
        List of positions in Cartesian coordinates.
    atomic_numbers
        Chemical symbol of each case.
    cell
        Cell metric.
    pbc
        Periodic boundary conditions.
    """

    def __init__(self,
                 positions: List[List[float]],
                 atomic_numbers: List[int],
                 cell: List[List[float]],
                 pbc: List[bool]) -> None:
        _Structure.__init__(self,
                            positions=positions,
                            atomic_numbers=atomic_numbers,
                            cell=cell,
                            pbc=pbc)

    @classmethod
    def from_atoms(self, conf: Atoms):
        """
        Returns the input configuration as an icet Structure object.

        Parameters
        ----------
        conf
            Input configuration.

        Returns
        -------
        Structure
        """
        return self(conf.positions,
                    conf.get_atomic_numbers(),
                    conf.cell,
                    conf.pbc.tolist())

    def find_lattice_sites_by_positions(self,
                                        positions: List[Sequence],
                                        fractional_position_tolerance: float) -> List[LatticeSite]:
        """
        Returns the lattice sites that match the positions.

        Parameters
        ----------
        positions
            List of positions in Cartesian coordinates.
        fractional_position_tolerance
            Tolerance for positions in fractional coordinates.
        """
        lattice_sites = []
        for position in positions:
            lattice_sites.append(self.find_lattice_site_by_position(
                position=position,
                fractional_position_tolerance=fractional_position_tolerance))
        return lattice_sites


def _structure_to_atoms(obj) -> Atoms:
    """
    Returns the structure as an :class:`Atoms <ase.Atoms>` object.
    """
    conf = Atoms(pbc=obj.pbc)
    conf.set_cell(obj.cell)
    for atomic_number, position in zip(obj.atomic_numbers, obj.positions):
        conf.append(Atom(atomic_number, position))
    return conf


# We want to be able to create ASE Atoms objects also if we
# have a _Structure object (returned from the C++ side) and not
# only if we have a Structure object, so we bind this function
# to _Structure instead of Structure
_Structure.to_atoms = _structure_to_atoms
