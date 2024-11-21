from typing import List, Tuple

import numpy as np

from ase import Atoms
from ase.build import make_supercell
from ase.geometry import get_distances
from scipy.optimize import linear_sum_assignment
from icet.input_output.logging_tools import logger
import scipy.linalg


def calculate_strain_tensor(A: np.ndarray,
                            B: np.ndarray) -> np.ndarray:
    """Calculates the strain tensor for mapping a cell A onto cell B. The
    strain calculated is the Biot strain tensor and is rotationally invariant.

    Parameters
    ----------
    A
        Reference cell (row-major format).
    B
        Target cell (row-major format).
    """
    assert A.shape == (3, 3)
    assert B.shape == (3, 3)

    # Calculate deformation gradient (F) in column-major format
    F = np.linalg.solve(A, B).T

    # Calculate right stretch tensor (U)
    _, U = scipy.linalg.polar(F)

    # return Biot strain tensor
    return U - np.eye(3)


def map_structure_to_reference(structure: Atoms,
                               reference: Atoms,
                               inert_species: List[str] = None,
                               tol_positions: float = 1e-4,
                               suppress_warnings: bool = False,
                               assume_no_cell_relaxation: bool = False,
                               remove_translation: bool = True) \
        -> Tuple[Atoms, dict]:
    """Maps a structure onto a reference structure.
    This is often desirable when, for example, a structure has been
    relaxed using DFT, and one wants to use it as a training structure
    in a cluster expansion.

    The function returns a tuple comprising the ideal supercell most
    closely matching the input structure and a dictionary with
    supplementary information concerning the mapping. The latter
    includes for example the largest deviation of any position in the
    input structure from its reference position (``drmax``), the average
    deviation of the positions in the input structure from the
    reference positions (``dravg``), and the strain tensor for the input
    structure relative to the reference structure (``strain_tensor``).

    The returned Atoms object provides further supplemental information via
    custom per-atom arrays including the atomic displacements
    (``Displacement``, ``Displacement_Magnitude``),the distances to the
    three closest sites (``Minimum_Distances``), as well as a mapping
    between the indices of the returned Atoms object and those of the input
    structure (``IndexMapping``).

    Note
    ----
    The input and reference structures should adhere to the same crystallographic setting.
    In other words, they should not be rotated or translated with respect to each other.

    Parameters
    ----------
    structure
        Input structure, typically a relaxed structure.
    reference
        Reference structure, which can but need not be the primitive structure.
    inert_species
        List of chemical symbols (e.g., ``['Au', 'Pd']``) that are never
        substituted for a vacancy. The number of inert sites is used to rescale
        the volume of the input structure to match the reference structure.
    tol_positions
        Tolerance factor applied when scanning for overlapping positions in
        Ångstrom (forwarded to :func:`ase.build.make_supercell`).
    suppress_warnings
        if ``True`` print no warnings of large strain or relaxation distances.
    assume_no_cell_relaxation
        If ``False`` volume and cell metric of the input structure are rescaled
        to match the reference structure. This can be unnecessary (and
        counterproductive) for some structures, e.g., with many vacancies.
    remove_translation
        If ``True`` mean translation between rescaled input structure and 
        primitive structure is subtracted from displacements.

        Note
        ----
        When setting this parameter to ``False`` the reference cell metric
        must be obtainable via an *integer* transformation matrix from the
        reference cell metric. In other words the input structure should not
        involve relaxations of the volume or the cell metric.

    Example
    -------
    The following code snippet illustrates the general usage. It first creates
    a primitive FCC cell, which is latter used as reference structure. To
    emulate a relaxed structure obtained from, e.g., a density functional
    theory calculation, the code then creates a 4x4x4 conventional FCC
    supercell, which is populated with two different atom types, has distorted
    cell vectors, and random displacements to the atoms. Finally, the present
    function is used to map the structure back the ideal lattice::

        >>> from ase.build import bulk
        >>> reference = bulk('Au', a=4.09)
        >>> structure = bulk('Au', cubic=True, a=4.09).repeat(4)
        >>> structure.set_chemical_symbols(10 * ['Ag'] + (len(structure) - 10) * ['Au'])
        >>> structure.set_cell(structure.cell * 1.02, scale_atoms=True)
        >>> structure.rattle(0.1)
        >>> mapped_structure, info = map_structure_to_reference(structure, reference)
    """

    # Obtain supercell of reference structure that is compatible
    # with relaxed structure
    reference_supercell, P = _get_reference_supercell(
        structure, reference, inert_species=inert_species,
        tol_positions=tol_positions, assume_no_cell_relaxation=assume_no_cell_relaxation)

    # Calculate strain tensor
    strain_tensor = calculate_strain_tensor(reference_supercell.cell, structure.cell)

    # Symmetric matrix has real eigenvalues
    eigenvalues, _ = np.linalg.eigh(strain_tensor)
    volumetric_strain = sum(eigenvalues)

    # Rescale the relaxed atoms object
    structure_scaled = structure.copy()
    structure_scaled.set_cell(reference_supercell.cell, scale_atoms=True)

    # Match positions
    mapped_structure, drmax, dravg, warning = _match_positions(structure_scaled,
                                                               reference_supercell,
                                                               remove_translation)

    if warning:
        warnings = [warning]
    else:
        warnings = []

    if not suppress_warnings:
        s = 'Consider excluding this structure when training a cluster expansion.'
        if assume_no_cell_relaxation:
            trigger_levels = {'volumetric_strain': 1e-3,
                              'eigenvalue_diff': 1e-3}
        else:
            trigger_levels = {'volumetric_strain': 0.25,
                              'eigenvalue_diff': 0.1}
        if abs(volumetric_strain) > trigger_levels['volumetric_strain']:
            warnings.append('high_volumetric_strain')
            logger.warning('High volumetric strain ({:.2f} %). {}'.format(
                100 * volumetric_strain, s))
        if max(eigenvalues) - min(eigenvalues) > trigger_levels['eigenvalue_diff']:
            warnings.append('high_anisotropic_strain')
            logger.warning('High anisotropic strain (the difference between '
                           'largest and smallest eigenvalues of strain tensor is '
                           '{:.5f}). {}'.format(max(eigenvalues) - min(eigenvalues), s))
        if drmax > 1.0:
            warnings.append('large_maximum_relaxation_distance')
            logger.warning('Large maximum relaxation distance '
                           '({:.5f} Angstrom). {}'.format(drmax, s))
        if dravg > 0.5:
            warnings.append('large_average_relaxation_distance')
            logger.warning('Large average relaxation distance '
                           '({:.5f} Angstrom). {}'.format(dravg, s))

    # Populate dictionary with supplementary information
    info = {'drmax': drmax,
            'dravg': dravg,
            'strain_tensor': strain_tensor,
            'volumetric_strain': volumetric_strain,
            'strain_tensor_eigenvalues': eigenvalues,
            'warnings': warnings,
            'transformation_matrix': P}

    return mapped_structure, info


def _get_reference_supercell(structure: Atoms,
                             reference: Atoms,
                             inert_species: List[str] = None,
                             tol_positions: float = 1e-5,
                             assume_no_cell_relaxation: bool = False) -> Tuple[Atoms, np.ndarray]:
    """
    Returns a tuple comprising a supercell of the reference structure that is compatible with the
    cell metric of the input structure as well as the transformation matrix that maps from the
    primitive to the supercell structure.

    Parameters
    ----------
    structure
        Input structure, typically a relaxed structure.
    reference
        Reference structure, which can but need not be the primitive structure.
    inert_species
        List of chemical symbols (e.g., ``['Au', 'Pd']``) that are never
        substituted for a vacancy. The number of inert sites is used to rescale
        of the input structure to match the reference structure.
    tol_positions
        Tolerance factor applied when scanning for overlapping positions in
        Ångstrom (forwarded to :func:`ase.build.make_supercell`).
    assume_no_cell_relaxation
        If ``False`` volume and cell metric of the input structure are rescaled
        to match the reference structure. This can be unnecessary (and
        counterproductive) for some structures, e.g., with many vacancies.

        Note
        ----
        When setting this parameter to ``False``, the input structure
        must be obtainable via an *integer* transformation matrix from the
        reference cell metric. In other words the input structure should not
        involve relaxations of the volume or the cell metric.

    Raises
    ------
    ValueError
        If the boundary conditions of the reference and the input structure
        do not match.
    ValueError
        If the transformation matrix deviates too strongly from the nearest
        integer matrix.
    ValueError
        If :attr:`assume_no_cell_relaxation` is ``True`` but the input structure is
        not obtainable via an integer transformation from the reference
        cell metric.
    """

    if not np.all(reference.pbc == structure.pbc):
        msg = 'The boundary conditions of reference and input structures do not match.'
        msg += '\n  reference: ' + str(reference.pbc)
        msg += '\n  input: ' + str(structure.pbc)
        raise ValueError(msg)

    # Step 1:
    # rescale cell metric of relaxed cell to match volume per atom of reference cell
    if not assume_no_cell_relaxation:
        if inert_species is None:
            n_ref = len(reference)
            n_rlx = len(structure)
        else:
            n_ref = sum([reference.get_chemical_symbols().count(s) for s in inert_species])
            n_rlx = sum([structure.get_chemical_symbols().count(s) for s in inert_species])
        vol_scale = reference.get_volume() / n_ref
        vol_scale /= structure.get_volume() / n_rlx
        scaled_structure_cell = structure.cell * vol_scale ** (1 / 3)
    else:
        scaled_structure_cell = structure.cell

    # Step 2:
    # get transformation matrix
    P = np.dot(scaled_structure_cell, np.linalg.inv(reference.cell))

    # reduce the (real) transformation matrix to the nearest integer one
    P = np.around(P)

    # Step 3:
    # generate supercell of reference structure
    reference_supercell = make_supercell(reference, P, tol=tol_positions)

    return reference_supercell, P


def _match_positions(
        structure: Atoms,
        reference: Atoms,
        remove_translation: bool = True) \
        -> Tuple[Atoms, float, float]:
    """Matches the atoms in the input :attr:`structure` to the sites in the
    :attr:`reference` structure. The function returns tuple the first element of which
    is a copy of the :attr:`reference` structure, in which the chemical species are
    assigned to comply with the :attr:`structure`. The second and third element
    of the tuple represent the maximum and average distance between relaxed and
    reference sites.

    Parameters
    ----------
    structure
        Structure with relaxed positions.
    reference
        Structure with idealized positions.
    remove_translation
        If ``True`` mean translation between structure and  reference is 
        subtracted from displacements.

    Raises
    ------
    ValueError
        If the cell metrics of the two input structures do not match.
    ValueError
        If the periodic boundary conditions of the two input structures do not match.
    ValueError
        If the input structure contains more atoms than the reference structure.
    """

    if not np.all(reference.pbc == structure.pbc):
        msg = 'The boundary conditions of reference and relaxed structures do not match.'
        msg += '\n  reference: ' + str(reference.pbc)
        msg += '\n  relaxed: ' + str(structure.pbc)
        raise ValueError(msg)

    if len(structure) > len(reference):
        msg = 'The relaxed structure contains more atoms than the reference structure.'
        msg += '\n  reference: ' + str(len(reference))
        msg += '\n  relaxed: ' + str(len(structure))
        raise ValueError(msg)

    if not np.all(np.isclose(reference.cell, structure.cell)):
        msg = 'The cell metrics of reference and relaxed structures do not match.'
        msg += '\n  reference: ' + str(reference.cell)
        msg += '\n  relaxed: ' + str(structure.cell)
        raise ValueError(msg)

    # compute distances between reference and relaxed positions
    _, dists = get_distances(reference.positions, structure.positions,
                             cell=reference.cell, pbc=reference.pbc)
    # pad matrix with zeros to obtain square matrix
    n, m = dists.shape
    cost_matrix = np.pad(dists, ((0, 0), (0, abs(n - m))),
                         mode='constant', constant_values=0)
    # find optimal mapping using Kuhn-Munkres (Hungarian) algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # compile new configuration with supplementary information
    mapped = reference.copy()
    displacement_magnitudes = []
    displacements = []
    minimum_distances = []
    n_dist_max = min(len(mapped), 3)
    warning = None
    for i, j in zip(row_ind, col_ind):
        atom = mapped[i]
        if j >= len(structure):
            # vacant site in reference structure
            atom.symbol = 'X'
            displacement_magnitudes.append(None)
            displacements.append(3 * [None])
            minimum_distances.append(n_dist_max * [None])
        else:
            atom.symbol = structure[j].symbol
            dvecs, drs = get_distances([structure[j].position],
                                       [reference[i].position],
                                       cell=reference.cell, pbc=reference.pbc)
            displacement_magnitudes.append(drs[0][0])
            displacements.append(dvecs[0][0])
            # distances to the next three available sites
            minimum_distances.append(sorted(dists[:, j])[:n_dist_max])
            if len(structure) > 1:
                if drs[0][0] > min(dists[:, j]) + 1e-6:
                    logger.warning('An atom was mapped to a site that was further '
                                   'away than the closest site (that site was already '
                                   'occupied by another atom).')
                    warning = 'possible_ambiguity_in_mapping'
                elif minimum_distances[-1][0] > 0.9 * minimum_distances[-1][1]:
                    logger.warning('An atom was approximately equally far from its '
                                   'two closest sites.')
                    warning = 'possible_ambiguity_in_mapping'

    if remove_translation:
        displacements = np.array(displacements)
        if not displacements.any(None):
            trans = displacements.mean(axis=0)
            displacements = displacements - trans
            displacement_magnitudes = np.sqrt((displacements**2).sum(axis=1))

    displacement_magnitudes = np.array(displacement_magnitudes, dtype=np.float64)
    mapped.new_array('Displacement', displacements, float, shape=(3, ))
    mapped.new_array('Displacement_Magnitude', displacement_magnitudes, float)
    mapped.new_array('Minimum_Distances', minimum_distances,
                     float, shape=(n_dist_max,))
    mapped.new_array('IndexMapping', col_ind, int)

    drmax = np.nanmax(displacement_magnitudes)
    dravg = np.nanmean(displacement_magnitudes)

    return mapped, drmax, dravg, warning
