import numpy as np
import spglib
from ase import Atoms
from typing import Tuple, List, Sequence, TypeVar

from ase.neighborlist import NeighborList as ase_NeighborList
from icet.core.neighbor_list import NeighborList
from icet.core.structure import Structure
from icet.core.lattice_site import LatticeSite
from icet.core_py.lattice_site import LatticeSite as LatticeSite_py

Vector = List[float]
T = TypeVar('T')


def get_scaled_positions(positions: np.ndarray,
                         cell: np.ndarray,
                         wrap: bool=True,
                         pbc: List[bool]=[True, True, True]) \
                         -> np.ndarray:
    """
    Returns the positions in reduced (or scaled) coordinates.

    Parameters
    ----------
    positions
        atomic positions in cartesian coordinates
    cell
        cell metric
    wrap
        if True, positions outside the unit cell will be wrapped into
        the cell in the directions with periodic boundary conditions
        such that the scaled coordinates are between zero and one.
    pbc
        periodic boundary conditions flags
    """

    fractional = np.linalg.solve(cell.T, positions.T).T

    if wrap:
        for i, periodic in enumerate(pbc):
            if periodic:
                # Yes, we need to do it twice.
                # See the scaled_positions.py test.
                fractional[:, i] %= 1.0
                fractional[:, i] %= 1.0

    return fractional


def add_vacuum_in_non_pbc(atoms: Atoms) -> Atoms:
    """
    Adds vacuum in non-periodic directions.

    Parameters
    ----------
    atoms
        input atomic structur

    Returns
    -------
    atoms
        input atomic structure with vacuum in non-pbc directions
    """
    atoms_cpy = atoms.copy()
    vacuum_axis = []
    for i, pbc in enumerate(atoms.pbc):
        if not pbc:
            vacuum_axis.append(i)

    if len(vacuum_axis) > 0:
        atoms_cpy.center(30, axis=vacuum_axis)
        atoms_cpy.wrap()

    return atoms_cpy


def get_primitive_structure(atoms: Atoms, no_idealize: bool=True) -> Atoms:
    """
    Determines primitive structure using spglib.

    Parameters
    ----------
    atoms
        input atomic structure
    no_idealize
        If True lengths and angles are not idealized

    Returns
    -------
    atoms_prim
        primitive structure
    """
    atoms_cpy = atoms.copy()
    atoms_with_vacuum = add_vacuum_in_non_pbc(atoms_cpy)

    atoms_as_tuple = ase_atoms_to_spglib_cell(atoms_with_vacuum)

    lattice, scaled_positions, numbers = spglib.standardize_cell(
        atoms_as_tuple, to_primitive=True, no_idealize=no_idealize)
    scaled_positions = [np.round(pos, 12) for pos in scaled_positions]
    atoms_prim = Atoms(scaled_positions=scaled_positions,
                       numbers=numbers, cell=lattice, pbc=atoms.pbc)
    atoms_prim.wrap()

    return atoms_prim


def get_fractional_positions_from_neighbor_list(
        structure: Structure, neighbor_list: NeighborList) -> List[Vector]:
    """
    Returns the fractional positions of the lattice sites in structure from
    a neighbor list.

    Parameters
    ----------
    atoms
        input atomic structure
    neighbor_list
        list of lattice neighbors of the input structure
    """
    neighbor_positions = []
    fractional_positions = []
    lattice_site = LatticeSite(0, [0, 0, 0])

    for i in range(len(neighbor_list)):
        lattice_site.index = i
        position = structure.get_position(lattice_site)
        neighbor_positions.append(position)
        for neighbor in neighbor_list.get_neighbors(i):
            position = structure.get_position(neighbor)
            neighbor_positions.append(position)

    if len(neighbor_positions) > 0:
        fractional_positions = get_scaled_positions(
            np.array(neighbor_positions),
            structure.cell, wrap=False,
            pbc=structure.pbc)

    return fractional_positions


def get_fractional_positions_from_ase_neighbor_list(
        atoms: Atoms, neighbor_list: ase_NeighborList) -> List[Vector]:
    """
    Returns the fractional positions of the lattice sites in atomic structure
    from a neighbor list.

    Parameters
    ----------
    atoms
        input atomic structure
    neighbor_list
        list of neighbors of the input structure
    """
    neighbor_positions = []
    fractional_positions = []

    for i in range(len(atoms)):
        lattice_site = LatticeSite_py(i, [0., 0., 0.])
        position = get_position_from_lattice_site(atoms, lattice_site)
        neighbor_positions.append(position)
        indices, offsets = neighbor_list.get_neighbors(i)
        for index, offset in zip(indices, offsets):
            lattice_site = LatticeSite_py(index, offset)
            position = get_position_from_lattice_site(atoms, lattice_site)
            neighbor_positions.append(position)
    if len(neighbor_positions) > 0:
        fractional_positions = get_scaled_positions(
            np.array(neighbor_positions),
            atoms.cell, wrap=False,
            pbc=atoms.pbc)
    return fractional_positions


def get_position_from_lattice_site(atoms: Atoms, lattice_site: LatticeSite):
    """
    Gets the corresponding position from the lattice site.

    Parameters
    ---------
    atoms
        input atomic structure
    lattice_site
        specific lattice site of the input structure
    """
    return atoms[lattice_site.index].position + \
        np.dot(lattice_site.unitcell_offset, atoms.get_cell())


def find_lattice_site_by_position(atoms: Atoms, position: List[float],
                                  tol: float=1e-4) -> LatticeSite_py:
    """
    Tries to construct a lattice site equivalent from
    position in reference to the atoms object.

    atoms
        input atomic structure
    position
        presumed cartesian coordinates of a lattice site
    """

    for i, atom in enumerate(atoms):
        pos = position - atom.position
        # Direct match
        if np.linalg.norm(pos) < tol:
            return LatticeSite_py(i, np.array((0., 0., 0.)))

        fractional = np.linalg.solve(atoms.cell.T, np.array(pos).T).T
        unit_cell_offset = [np.floor(round(x)) for x in fractional]
        residual = np.dot(fractional - unit_cell_offset, atoms.cell)
        if np.linalg.norm(residual) < tol:
            latNbr = LatticeSite_py(i, unit_cell_offset)
            return latNbr

    # found nothing, raise error
    raise RuntimeError("Did not find site in find_lattice_site_by_position")


def fractional_to_cartesian(atoms: Atoms,
                            frac_positions: List[Vector]) -> List[Vector]:
    """
    Turns fractional positions into cartesian positions.

    Parameters
    ----------
    atoms
        input atomic structure
    frac_positions
        fractional positions
    """
    return np.dot(frac_positions, atoms.cell)


def get_permutation(container: Sequence[T],
                    permutation: List[int]) -> Sequence[T]:
    """
    Returns the permuted version of container.
    """
    if len(permutation) != len(container):
        raise RuntimeError("Container and permutation"
                           " not of same size {} != {}".format(
                               len(container), len(permutation)))
    if len(set(permutation)) != len(permutation):
        raise Exception
    return [container[s] for s in permutation]


def find_permutation(target: Sequence[T],
                     permutated: Sequence[T]) -> List[int]:
    """
    Returns the permutation vector that takes permutated to target.

    Parameters
    ----------
    target
        a container that supports indexing and its elements contain
        objects with __eq__ method
    permutated
        a container that supports indexing and its elements contain
        objects with __eq__ method
    """
    permutation = []
    for element in target:
        index = permutated.index(element)
        permutation.append(index)
    return permutation


def ase_atoms_to_spglib_cell(atoms: Atoms) \
                             -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a tuple of three components: cell metric, atomic positions, and
    atomic species of the input ASE Atoms object.
    """
    return (atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
