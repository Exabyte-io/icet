import numpy as np
import spglib as spglib

from ase import Atoms
from _icetdev import PermutationMap
from icetdev.neighborlist import get_neighborlists, Neighborlist
from icetdev.lattice_site import LatticeSite
from icetdev.structure import Structure
from icetdev.tools.geometry import get_scaled_positions


def __vacuum_on_non_pbc(atoms):
    '''
    Add vacuum in non-periodic directions.

    Parameters
    ----------
    atoms : ASE Atoms object
        input structure

    Returns
    -------
    ASE Atoms object
        output structure
    '''

    vacuum_axis = []
    for i, pbc in enumerate(atoms.pbc):
        if not pbc:
            vacuum_axis.append(i)

    if len(vacuum_axis) > 0:
        atoms.center(30, axis=vacuum_axis)
    atoms.wrap()

    return atoms


def __get_primitive_structure(atoms):
    '''
    Determines primitive structure using spglib.

    Parameters
    ----------
    atoms : ASE Atoms object
        input structure

    Returns
    -------
    ASE Atoms object
        output structure
    '''

    atoms = __vacuum_on_non_pbc(atoms)
    lattice, scaled_positions, numbers = spglib.standardize_cell(
        atoms, to_primitive=True, no_idealize=True)
    scaled_positions = [np.round(pos, 12) for pos in scaled_positions]
    atoms_prim = Atoms(scaled_positions=scaled_positions,
                       numbers=numbers, cell=lattice, pbc=atoms.pbc)
    atoms_prim.wrap()
    return atoms_prim

def vacuum_on_non_pbc(atoms):
    '''
    Returns atoms with non pbc after adding vacuum along non 
    periodic directions 

    '''
    atoms = __vacuum_on_non_pbc(atoms)
    return atoms


def __get_fractional_positions_from_neigborlist(structure, neighborlist):
    '''
    Returns the fractional positions in structure from the neighbors in the
    neighbor list.
    '''
    neighbor_positions = []
    fractional_positions = []
    lattice_site = LatticeSite(0, [0, 0, 0])
    for i in range(len(neighborlist)):
        lattice_site.index = i
        position = structure.get_position(lattice_site)
        neighbor_positions.append(position)
        for neighbor in neighborlist.get_neighbors(i):
            position = structure.get_position(neighbor)
            neighbor_positions.append(position)
    if len(neighbor_positions) > 0:
        fractional_positions = get_scaled_positions(
            np.array(neighbor_positions),
            structure.cell, wrap=False,
            pbc=structure.pbc)
    return fractional_positions


def permutation_matrices_from_atoms(atoms, cutoffs,
                                    find_prim=True, verbosity=0):
    '''Set up a list of permutation maps from an atoms object.

    Parameters
    ----------
    atoms : ASE Atoms object
        input structure
    cutoffs : list of float
        cutoff radii
    find_primitive : boolean
        if True the symmetries of the primitive structure will be employed
    verbosity : int
        level of verbosity

    Returns
    -------
    PermutationMap, icet Structure object, list of NeighborList objects
        the tuple comprises the permutation matrices, the primitive structure,
        and the neighbor lists
    '''

    atoms = atoms.copy()
    # set each element to the same since we only care about geometry when
    # taking primitive
    atoms.set_chemical_symbols(len(atoms) * [atoms[0].symbol])

    atoms_prim = atoms
    if find_prim:
        atoms_prim = __get_primitive_structure(atoms)

    if verbosity >= 3:
        print('size of atoms_prim {}'.format(len(atoms_prim)))
    # Get symmetry information and load into a permutation map object
    symmetry = spglib.get_symmetry(atoms_prim)
    translations = symmetry['translations']
    rotations = symmetry['rotations']
    permutation_matrices = PermutationMap(translations, rotations)

    # Create neighborlists from the different cutoffs
    prim_structure = Structure.from_atoms(atoms_prim)
    neighborlists = get_neighborlists(prim_structure,
                                      cutoffs=cutoffs)
    # get fractional positions for each neighborlist
    for i, neighborlist in enumerate(neighborlists):
        if verbosity >= 3:
            print('building permutation map {}/{}'.format(i,
                                                          len(neighborlists)))
        frac_positions = __get_fractional_positions_from_neigborlist(
            prim_structure, neighborlist)
        if verbosity >= 3:
            print('number of positions: {}'.format(len(frac_positions)))
        if len(frac_positions) > 0:
            permutation_matrices[i].build(frac_positions)

    return permutation_matrices, prim_structure, neighborlists


def permutation_matrix_from_atoms(atoms, cutoff,
                                  find_prim=True, verbosity=0):
    '''Set up a list of permutation maps from an atoms object.

    Parameters
    ----------
    atoms : ASE Atoms object
        input structure
    cutoff : float
        cutoff radius
    find_primitive : boolean
        if True the symmetries of the primitive structure will be employed
    verbosity : int
        level of verbosity

    Returns
    -------
    matrix, icet Structure object, NeighborList object
        the tuple comprises the permutation matrix, the primitive structure,
        and the neighbor list
    '''

    atoms = atoms.copy()
    # set each element to the same since we only care about geometry when
    # taking primitive
    if len(atoms) > 0:
        atoms.set_chemical_symbols(len(atoms) * [atoms[0].symbol])
    else:
        raise Exception('Len of atoms are {}'.format(len(atoms)))

    atoms_prim = atoms
    if find_prim:
        atoms_prim = __get_primitive_structure(atoms)

    if verbosity >= 3:
        print('size of atoms_prim {}'.format(len(atoms_prim)))
    # Get symmetry information and load into a permutation map object
    symmetry = spglib.get_symmetry(atoms_prim)
    translations = symmetry['translations']
    rotations = symmetry['rotations']

    permutation_matrix = PermutationMap(translations, rotations)

    # Create neighborlists from the different cutoffs
    prim_structure = Structure.from_atoms(atoms_prim)
    neighborlist = Neighborlist(cutoff)
    neighborlist.build(prim_structure)

    # get fractional positions for neighborlist
    frac_positions = __get_fractional_positions_from_neigborlist(
        prim_structure, neighborlist)
    if verbosity >= 3:
        print('number of positions: {}'.format(len(frac_positions)))
    if len(frac_positions) > 0:
        permutation_matrix.build(frac_positions)

    return permutation_matrix, prim_structure, neighborlist
