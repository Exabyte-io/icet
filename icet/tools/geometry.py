import numpy as np
import spglib
from typing import Tuple, List, Sequence, TypeVar
from ase import Atoms
from ase.data import chemical_symbols
from icet.core.lattice_site import LatticeSite
from icet.core.neighbor_list import NeighborList
from icet.core.structure import Structure

Vector = List[float]
T = TypeVar('T')


def get_scaled_positions(positions: np.ndarray,
                         cell: np.ndarray,
                         wrap: bool = True,
                         pbc: List[bool] = [True, True, True]) \
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


def get_primitive_structure(structure: Atoms,
                            no_idealize: bool = True,
                            to_primitive: bool = True,
                            symprec: float = 1e-5) -> Atoms:
    """
    Determines primitive structure using spglib.

    Parameters
    ----------
    structure
        input atomic structure
    no_idealize
        if True lengths and angles are not idealized
    to_primitive
        convert to primitive structure
    symprec
        tolerance imposed when analyzing the symmetry using spglib

    Returns
    -------
    structure_prim
        primitive structure
    """
    structure_cpy = structure.copy()
    structure_as_tuple = ase_atoms_to_spglib_cell(structure_cpy)

    lattice, scaled_positions, numbers = spglib.standardize_cell(
        structure_as_tuple, to_primitive=to_primitive,
        no_idealize=no_idealize, symprec=symprec)
    scaled_positions = [np.round(pos, 12) for pos in scaled_positions]
    structure_prim = Atoms(scaled_positions=scaled_positions,
                           numbers=numbers, cell=lattice, pbc=structure.pbc)
    structure_prim.wrap()

    return structure_prim


def get_fractional_positions_from_neighbor_list(
        structure: Structure, neighbor_list: NeighborList) -> List[Vector]:
    """
    Returns the fractional positions of the lattice sites in structure from
    a neighbor list.

    Parameters
    ----------
    structure
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


def get_position_from_lattice_site(structure: Atoms, lattice_site: LatticeSite):
    """
    Gets the corresponding position from the lattice site.

    Parameters
    ---------
    structure
        input atomic structure
    lattice_site
        specific lattice site of the input structure
    """
    return structure[lattice_site.index].position + \
        np.dot(lattice_site.unitcell_offset, structure.get_cell())


def find_lattice_site_by_position(structure: Atoms, position: List[float],
                                  tol: float = 1e-4) -> LatticeSite:
    """
    Tries to construct a lattice site equivalent from
    position in reference to the ASE Atoms object.

    structure
        input atomic structure
    position
        presumed cartesian coordinates of a lattice site
    """

    for i, atom in enumerate(structure):
        pos = position - atom.position
        # Direct match
        if np.linalg.norm(pos) < tol:
            return LatticeSite(i, np.array((0., 0., 0.)))

        fractional = np.linalg.solve(structure.cell.T, np.array(pos).T).T
        unit_cell_offset = [np.floor(round(x)) for x in fractional]
        residual = np.dot(fractional - unit_cell_offset, structure.cell)
        if np.linalg.norm(residual) < tol:
            latNbr = LatticeSite(i, unit_cell_offset)
            return latNbr

    # found nothing, raise error
    raise RuntimeError('Did not find site in find_lattice_site_by_position')


def fractional_to_cartesian(structure: Atoms,
                            frac_positions: List[Vector]) -> List[Vector]:
    """
    Turns fractional positions into cartesian positions.

    Parameters
    ----------
    structure
        input atomic structure
    frac_positions
        fractional positions
    """
    return np.dot(frac_positions, structure.cell)


def get_permutation(container: Sequence[T],
                    permutation: List[int]) -> Sequence[T]:
    """
    Returns the permuted version of container.
    """
    if len(permutation) != len(container):
        raise RuntimeError('Container and permutation'
                           ' not of same size {} != {}'.format(
                               len(container), len(permutation)))
    if len(set(permutation)) != len(permutation):
        raise Exception
    return [container[s] for s in permutation]


def ase_atoms_to_spglib_cell(structure: Atoms) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a tuple of three components: cell metric, atomic positions, and
    atomic species of the input ASE Atoms object.
    """
    return (structure.get_cell(), structure.get_scaled_positions(), structure.get_atomic_numbers())


def get_occupied_primitive_structure(
        structure: Atoms,
        allowed_species: List[List[str]],
        symprec: float = 1e-5) -> Tuple[Atoms, List[List[str]]]:
    """
    Returns an occupied primitive structure.
    Will put hydrogen on sublattice 1, Helium on sublattice 2 and
    so on

    Parameters
    ----------
    structure
        input structure
    allowed_species
        chemical symbols that are allowed on each site
    symprec
        tolerance imposed when analyzing the symmetry using spglib

    Todo
    ----
    simplify the revert back to unsorted symbols
    """
    if len(structure) != len(allowed_species):
        raise ValueError(
            'structure and chemical symbols need to be the same size.')
    symbols = sorted({tuple(sorted(s)) for s in allowed_species})

    decorated_primitive = structure.copy()
    for i, sym in enumerate(allowed_species):
        sublattice = symbols.index(tuple(sorted(sym))) + 1
        decorated_primitive[i].symbol = chemical_symbols[sublattice]

    decorated_primitive = get_primitive_structure(decorated_primitive,
                                                  symprec=symprec)
    decorated_primitive.wrap()
    primitive_chemical_symbols = []
    for atom in decorated_primitive:
        sublattice = chemical_symbols.index(atom.symbol)
        primitive_chemical_symbols.append(symbols[sublattice - 1])

    for symbols in allowed_species:
        if tuple(sorted(symbols)) in primitive_chemical_symbols:
            index = primitive_chemical_symbols.index(tuple(sorted(symbols)))
            primitive_chemical_symbols[index] = symbols
    return decorated_primitive, primitive_chemical_symbols


def atomic_number_to_chemical_symbol(numbers: List[int]) -> List[str]:
    """Returns the chemical symbols equivalent to the input atomic
    numbers.

    Parameters
    ----------
    numbers
        atomic numbers
    """

    symbols = [chemical_symbols[number] for number in numbers]
    return symbols


def chemical_symbols_to_numbers(symbols: List[str]) -> List[int]:
    """Returns the atomic numbers equivalent to the input chemical
    symbols.

    Parameters
    ----------
    symbols
        chemical symbols
    """

    numbers = [chemical_symbols.index(symbols) for symbols in symbols]
    return numbers
