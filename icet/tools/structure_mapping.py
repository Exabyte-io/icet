import logging
from typing import List, Tuple

import numpy as np
from ase import Atoms
from ase.build import cut

from icet.io.logging import logger
logger = logger.getChild('structure_mapping')


def map_structure_to_reference(input_structure: Atoms,
                               reference_structure: Atoms,
                               tolerance_mapping: float,
                               vacancy_type: str = None,
                               inert_species: List[str] = None,
                               tolerance_cell: float = 0.05,
                               tolerance_positions: float = 0.01) \
                               -> Tuple[Atoms, float, float]:
    """Maps a relaxed structure onto a reference structure.
    The function returns a tuple comprising

    * the ideal supercell most closely matching the input structure,
    * the largest deviation of any input coordinate from its ideal
      coordinate, and
    * the average deviation of the input coordinates from the ideal
      coordinates.

    Parameters
    ----------
    input_structure
        relaxed input structure
    reference_structure
        reference structure, which can but need not represent the primitive
        cell
    tolerance_mapping
        maximum allowed displacement for mapping an atom in the relaxed (but
        rescaled) structure to the reference supercell

        *Note*: A reasonable choice is up to 20-30% of the first
        nearest neighbor distance (`r1`).  A value above 50% of `r1`
        will most likely lead to atoms being multiply assigned,
        whereby the mapping fails.
    vacancy_type
        If this parameter is set to a non-zero string unassigned sites in the
        reference structure will be assigned to this type.

        *Note 1*: By default (``None``) the method will fail if there
        are *any* unassigned sites in the reference structure.

        *Note 2*: ``vacancy_type`` must be a valid species as
        enforced by the :class:`ase.Atoms` class.
    inert_species
        List of chemical symbols (e.g., ``['Au', 'Pd']``) that are never
        substituted for a vacancy. Used to make an initial rescale of the cell
        and thus increases the probability for a successful mapping. Need not
        be specified if ``vacancy_type`` is ``None``.
    tolerance_cell
        tolerance factor applied when computing permutation matrix to generate
        supercell
    tolerance_positions
        tolerance factor applied when scanning for overlapping positions in
        Angstrom (forwarded to :func:`ase.build.cut`)

    Example
    -------
    The following code snippet illustrates the general usage. It first creates
    a primitive FCC cell, which is latter used as reference structure. To
    emulate a relaxed structure obtained from, e.g., a density functional
    theory calculation, the code then creates a 4x4x4 conventional FCC
    supercell, which is populated with two different atom types, has distorted
    cell vectors, and random displacements to the atoms. Finally, the present
    function is used to map the structure back the ideal lattice::

        from ase.build import bulk
        reference = bulk('Au', a=4.09)
        atoms = bulk('Au', cubic=True, a=4.09).repeat(4)
        atoms.set_chemical_symbols(10 * ['Ag'] + (len(atoms) - 10) * ['Au'])
        atoms.set_cell(atoms.cell * 1.02, scale_atoms=True)
        atoms.rattle(0.1)
        mapped_atoms = map_structure_to_reference(atoms, reference, 1.0)

    """
    assert np.all(input_structure.pbc == reference_structure.pbc), \
        ('The periodic boundary conditions differ'
         ' between input and reference structure')

    # Scale input cell and construct supercell of the reference structure
    scaled_cell = _get_scaled_cell(input_structure, reference_structure,
                                   vacancy_type=vacancy_type,
                                   inert_species=inert_species)
    P = _get_transformation_matrix(scaled_cell,
                                   reference_structure.cell,
                                   tolerance_cell=tolerance_cell)
    scaled_structure, ideal_supercell = \
        _rescale_structures(input_structure,
                            reference_structure,
                            P,
                            tolerance_positions=tolerance_positions)

    assert len(ideal_supercell) == len(scaled_structure) or \
        vacancy_type is not None, \
        ('Number of atoms in ideal supercell does not match '
         'input structure.\n'
         'ideal: {}\ninput: {}'.format(len(ideal_supercell),
                                       len(scaled_structure)))

    if logger.isEnabledFor(logging.DEBUG):
        np.set_printoptions(suppress=True, precision=6)
        logger.debug('Number of atoms in reference structure:'
                     ' {}'.format(len(reference_structure)))
        logger.debug('Number of atoms in input structure:'
                     ' {}\n'.format(len(input_structure)))
        logger.debug('Reference cell metric:\n'
                     '{}'.format(reference_structure.cell))
        logger.debug('Input cell metric:\n'
                     '{}\n'.format(input_structure.cell))
        logger.debug('Transformation matrix connecting reference structure'
                     ' and idealized input structure:\n {}'.format(P))
        logger.debug('Determinant of transformation matrix:'
                     ' {:.3f}\n'.format(np.linalg.det(P)))
        logger.debug('Cell metric of ideal supercell:\n'
                     '{}'.format(ideal_supercell.cell))
        logger.debug('Cell metric of rescaled input structure:\n'
                     '{}\n'.format(scaled_structure.cell))

    # map atoms in input structure to closest site in ideal supercell
    dr_max = 0.0
    dr_sum = 0.0
    dr_sumsq = 0.0
    # per-atom-list for keeping track of mapped atoms
    mapped = [-1] * len(ideal_supercell)
    # distances between ideal and input sites
    drs = [None] * len(ideal_supercell)
    for ideal_site in ideal_supercell:
        for atom in scaled_structure:
            # in order to compute the distance the current atom from
            # the input structure is temporarily added to the
            # ideal supercell. This allows one to simply use the ASE
            # Atoms method for computing the interatomic distance
            ideal_supercell.append(atom)
            dr = ideal_supercell.get_distance(ideal_site.index,
                                              ideal_supercell[-1].index,
                                              mic=True)
            del ideal_supercell[-1]
            if dr < tolerance_mapping:
                if mapped[ideal_site.index] >= 0:
                    raise Exception('More than one atom from the relaxed'
                                    ' (and rescaled) structure have been'
                                    ' mapped onto the same ideal site.\n'
                                    ' Try reducing `tolerance_mapping`.')
                mapped[ideal_site.index] = atom.index
                drs[ideal_site.index] = dr
                ideal_site.symbol = atom.symbol
                dr_max = max(dr, dr_max)
                dr_sum += dr
                dr_sumsq += dr * dr
                break
        else:
            assert vacancy_type is not None, \
                ('Failed to assign an atom from the relaxed (and'
                 ' rescaled) structure to the ideal lattice.'
                 ' Try increasing `tolerance_mapping`.\n'
                 ' {}'.format(ideal_site))
            ideal_site.symbol = vacancy_type

    dr_avg = dr_sum / len(ideal_supercell)
    dr_sdv = np.sqrt(dr_sumsq / len(ideal_supercell) - dr_avg ** 2)

    # check that not more than one atom was assigned to the same site
    for k in set(mapped):
        assert k < 0 or mapped.count(k) <= 1, \
            ('Site {} has been assigned more than once.'.format(k))

    # check that the chemical composition of input and ideal supercell matches
    for symbol in set(input_structure.get_chemical_symbols()):
        n1 = input_structure.get_chemical_symbols().count(symbol)
        n2 = ideal_supercell.get_chemical_symbols().count(symbol)
        assert n1 == n2, ('Number of atoms of type {} differs between'
                          ' input structure ({}) and ideal'
                          ' supercell ({}).'.format(symbol, n1, n2))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Maximum, average and standard deviation of atomic'
                     ' displacements: {} {} {}'.format(dr_max, dr_avg, dr_sdv))

        logger.debug('{:52} {:}'.format('Input structure:',
                                        'Scaled structure:'))
        for k, (input_atom, scaled_atom) in enumerate(zip(input_structure,
                                                          scaled_structure)):
            msg = '{:4}  {:2}'.format(k, input_atom.symbol)
            msg += (3 * ' {:12.6f}').format(*input_atom.position)
            msg += '    -->'
            msg += (3 * ' {:12.6f}').format(*scaled_atom.position)
            logger.debug(msg)

        logger.debug('\n')

        logger.debug('{:52} {}'.format('Ideal supercell:',
                                       'Scaled structure:'))
        for ideal_atom, k, dr in zip(ideal_supercell, mapped, drs):
            msg = ' {:2}'.format(ideal_atom.symbol)
            msg += (3 * '  {:12.6f}').format(*ideal_atom.position)
            msg += '    -->'
            msg += ' {:4}'.format(k)

            if k >= 0:
                scaled_pos = scaled_structure[k].position
                msg += (3 * ' {:12.6f}').format(*scaled_pos)
                msg += '    --> {:.4f}'.format(dr)
            logger.debug(msg)

    return ideal_supercell, dr_max, dr_avg


def _get_scaled_cell(input_structure: Atoms,
                     reference_structure: Atoms,
                     vacancy_type: str = None,
                     inert_species: List[str] = None) -> np.ndarray:
    """
    The input structure needs to be scaled in order to match the lattice
    structure of the reference structure. The reference structure can be a
    primitive cell, in which case the input structure would usually be a
    supercell thereof. Also, we need an ideal supercell that matches the input
    structure.

    Parameters
    ----------
    input_structure
        relaxed input structure
    reference_structure: ASE Atoms object
        reference structure, which can but need not represent the primitive
        cell
    vacancy_type
        if not None, the cell is scaled if and only if `inert_species` is not
        None
    inert_species
        list of chemical symbols(e.g., `['Au', 'Pd']`) that are never
        substituted for a vacancy. Needless if `vacancy_type` is `None`
    """
    modcell = input_structure.get_cell()
    if vacancy_type is None:
        # Without scale factor we can just rescale with number of atoms
        atvol_in = input_structure.get_volume() / len(input_structure)
        atvol_ref = reference_structure.get_volume() / len(reference_structure)
        scale = atvol_in / atvol_ref

    if vacancy_type is not None:
        if inert_species is None:
            scale = 1.0
        else:
            # We cannot use the number of atoms since there may be vacancies
            # in the input_structure. Instead we count the species that we
            # know should always be present.
            n_in = 0
            n_ref = 0
            symbols_in = input_structure.get_chemical_symbols()
            symbols_ref = reference_structure.get_chemical_symbols()
            for species in inert_species:
                n_in += symbols_in.count(species)
                n_ref += symbols_ref.count(species)
            atvol_in = input_structure.get_volume() / n_in
            atvol_ref = reference_structure.get_volume() / n_ref
            scale = atvol_in / atvol_ref
    modcell *= (1.0 / scale) ** (1.0 / 3.0)
    return modcell


def _get_transformation_matrix(input_cell: np.ndarray,
                               reference_cell: np.ndarray,
                               tolerance_cell: float = 0.05) -> np.ndarray:
    """
    Obtains the (in general non-integer) transformation matrix connecting the
    input structure to the reference structure L=L_p.P - -> P=L_p ^ -1.L

    Parameters
    ----------
    input_cell
        cell metric of input structure(possibly scaled)
    reference_cell
        cell metric of reference structure
    tolerance_cell
        tolerance for how much the elements of P are allowed to deviate from
        the nearest integer before they are rounded

    Returns
    -------
    transformation matrix P of integers
    """
    P = np.dot(input_cell, np.linalg.inv(reference_cell))

    # ensure that the transformation matrix does not deviate too
    # strongly from the nearest integer matrix
    if np.linalg.norm(P - np.around(P)) / 9 > tolerance_cell:
        print('reference:\n {}\n'.format(reference_cell))
        print('input:\n {}\n'.format(input_cell))
        print('P:\n {}\n'.format(P))
        print('det P = {}\n'.format(np.linalg.det(P)))
        print('P_round:\n {}\n'.format(np.around(P)))
        print('Deviation: {}\n'.format(np.linalg.norm(P - np.around(P)) / 9))
        raise Exception('Failed to map structure to reference structure'
                        ' (tolerance_cell exceeded). If there are vacancies,'
                        ' one can try specifying `inert_species`. Otherwise,'
                        ' one can try raising `tolerance_cell`.')

    # reduce the (real) transformation matrix to the nearest integer one
    P = np.around(P)
    return P


def _rescale_structures(input_structure: Atoms,
                        reference_structure: Atoms,
                        P: np.ndarray,
                        tolerance_positions: float = 0.01) \
                        -> Tuple[Atoms, Atoms]:
    """
    Rescales `input_structure` with `P` so that it matches
    `reference_structure`, and creates a supercell of `reference_structure`
    using `P`

    Parameters
    ----------
    input_structure
        relaxed input structure
    reference_structure
        reference structure, which can but need not represent the primitive
        cell
    P
        transformation matrix of integers
    tolerance_positions
        tolerance factor applied when scanning for overlapping positions in
        Angstrom(forwarded to `ase.build.cut`)

    Returns
    -------
    a tuple with the scaled version of `input_structure` and the supercell of
    `reference_structure` matching cell metric of `scaled_structure`
    """
    scaled_structure = input_structure.copy()
    scaled_structure.set_cell(np.dot(P, reference_structure.cell),
                              scale_atoms=True)

    # generate supercell of (presumably primitive) reference structure
    ideal_supercell = cut(reference_structure, *P,
                          tolerance=tolerance_positions)
    n_mapped = int(np.round(len(reference_structure) * np.linalg.det(P)))
    if len(ideal_supercell) != n_mapped:
        print('len(reference_structure)', len(reference_structure))
        print('det(P)', np.linalg.det(P))
        print('len(ideal_supercell)', len(ideal_supercell))
        print('n_mapped', n_mapped)
        raise Exception('Supercell construction of reference structure failed'
                        ' (number of atoms do not match).\n'
                        'Permutation matrix used:\n{}'.format(P) +
                        '\nYou can try to change tolerance_positions.')

    return scaled_structure, ideal_supercell
