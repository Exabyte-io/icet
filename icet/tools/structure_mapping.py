from typing import List, Tuple

import numpy as np

from ase import Atoms
from ase.build import make_supercell
from ase.geometry import get_distances
from scipy.optimize import linear_sum_assignment
from icet.io.logging import logger


def map_structure_to_reference(relaxed: Atoms,
                               reference: Atoms,
                               inert_species: List[str] = None,
                               tol_positions: float = 1e-4,
                               suppress_warnings: bool = False,
                               assume_no_cell_relaxation: bool = False) \
        -> Tuple[Atoms, dict]:
    """Maps a relaxed structure onto a reference structure.
    The function returns a tuple comprising the ideal supercell most
    closely matching the relaxed structure and a dictionary with
    supplementary information concerning the mapping. The latter
    includes for example the largest deviation of any position in the
    relaxed structure from its reference position (`drmax`), the average
    deviation of the positions in the relaxed structure from the
    reference positions (`dravg`), and the strain tensor for the relaxed
    structure relative to the reference structure (`strain_tensor`).

    The Atoms object that provide further supplemental information via
    custom per-atom arrays including the atomic displacements
    (`Displacement`, `Displacement_Magnitude`) as well as the
    distances to the three closest sites (`Minimum_Distances`).

    Parameters
    ----------
    relaxed
        relaxed input structure
    reference
        reference structure, which can but need not be the primitive structure
    inert_species
        list of chemical symbols (e.g., ``['Au', 'Pd']``) that are never
        substituted for a vacancy; the number of inert sites is used to rescale
        the volume of the relaxed structure to match the reference structure.
    tol_positions
        tolerance factor applied when scanning for overlapping positions in
        Angstrom (forwarded to :func:`ase.build.make_supercell`)
    suppress_warnings
        if True, print no warnings of large strain or relaxation distances
    assume_no_cell_relaxation
        if False volume and cell metric of the relaxed structure are rescaled
        to match the reference structure; this can be unnecessary (and
        counterproductive) for some structures, e.g., with many vacancies

        **Note**: When setting this parameter to False the reference cell metric
        must be obtainable via an *integer* transformation matrix from the
        reference cell metric. In other words the relaxed structure should not
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

        from icet.tools import map_structure_to_reference
        from ase.build import bulk
        reference = bulk('Au', a=4.09)
        structure = bulk('Au', cubic=True, a=4.09).repeat(4)
        structure.set_chemical_symbols(10 * ['Ag'] + (len(structure) - 10) * ['Au'])
        structure.set_cell(structure.cell * 1.02, scale_atoms=True)
        structure.rattle(0.1)
        mapped_structure, info = map_structure_to_reference(structure, reference)

    """

    # Obtain supercell of reference structure that is compatible
    # with relaxed structure
    reference_supercell = _get_reference_supercell(
        relaxed, reference, inert_species=inert_species,
        tol_positions=tol_positions, assume_no_cell_relaxation=assume_no_cell_relaxation)

    # Compute strain
    epsilon = np.dot(relaxed.cell.T - reference_supercell.cell.T,
                     np.linalg.inv(reference_supercell.cell.T))
    strain_tensor = 0.5 * (epsilon + epsilon.T)
    eigenvalues = np.linalg.eig(strain_tensor)[0]
    volumetric_strain = sum(eigenvalues)

    # Rescale the relaxed atoms object
    relaxed_scaled = relaxed.copy()
    relaxed_scaled.set_cell(reference_supercell.cell, scale_atoms=True)

    # Match positions
    mapped_structure, drmax, dravg = _match_positions(relaxed_scaled, reference_supercell)

    if not suppress_warnings:
        s = 'Consider excluding this structure when training a cluster expansion.'
        if assume_no_cell_relaxation:
            trigger_levels = {'volumetric_strain': 1e-3,
                              'eigenvalue_diff': 1e-3}
        else:
            trigger_levels = {'volumetric_strain': 0.25,
                              'eigenvalue_diff': 0.1}
        if abs(volumetric_strain) > trigger_levels['volumetric_strain']:
            logger.warning('High volumetric strain ({:.2f} %). {}'.format(
                100 * volumetric_strain, s))
        if max(eigenvalues) - min(eigenvalues) > trigger_levels['eigenvalue_diff']:
            logger.warning('High anisotropic strain (the difference between '
                           'largest and smallest eigenvalues of strain tensor is '
                           '{:.5f}). {}'.format(max(eigenvalues) - min(eigenvalues), s))
        if drmax > 1.0:
            logger.warning('Large maximum relaxation distance '
                           '({:.5f} Angstrom). {}'.format(drmax, s))
        if dravg > 0.5:
            logger.warning('Large average relaxation distance '
                           '({:.5f} Angstrom). {}'.format(dravg, s))

    # Populate dictionary with supplementary information
    info = {'drmax': drmax,
            'dravg': dravg,
            'strain_tensor': strain_tensor,
            'volumetric_strain': volumetric_strain,
            'strain_tensor_eigenvalues': eigenvalues}

    return mapped_structure, info


def _get_reference_supercell(relaxed: Atoms,
                             reference: Atoms,
                             inert_species: List[str] = None,
                             tol_positions: float = 1e-5,
                             assume_no_cell_relaxation: bool = False) -> Atoms:
    """
    Returns a supercell of the reference structure that is compatible with the
    cell metric of the relaxed structure.

    Parameters
    ----------
    relaxed
        relaxed input structure
    reference
        reference structure, which can but need not be the primitive structure
    inert_species
        list of chemical symbols (e.g., ``['Au', 'Pd']``) that are never
        substituted for a vacancy; the number of inert sites is used to rescale
        of the relaxed structure to match the reference structure.
    tol_positions
        tolerance factor applied when scanning for overlapping positions in
        Angstrom (forwarded to :func:`ase.build.make_supercell`)
    assume_no_cell_relaxation
        if False volume and cell metric of the relaxed structure are rescaled
        to match the reference structure; this can be unnecessary (and
        counterproductive) for some structures, e.g., with many vacancies

        **Note**: When setting this parameter to False relaxed structure
        must be obtainable via an *integer* transformation matrix from the
        reference cell metric. In other words the relaxed structure should not
        involve relaxations of the volume or the cell metric.

    Raises
    ------
    ValueError
        if the boundary conditions of the reference and the relaxed structure
        do not match
    ValueError
        if the transformation matrix deviates too strongly from the nearest
        integer matrix
    ValueError
        if assume_no_cell_relaxation is True but the relaxed structure is
        not obtainable via an integer transformation from the reference
        cell metric
    """

    if not np.all(reference.pbc == relaxed.pbc):
        msg = 'The boundary conditions of reference and relaxed structures do not match.'
        msg += '\n  reference: ' + str(reference.pbc)
        msg += '\n  relaxed: ' + str(relaxed.pbc)
        raise ValueError(msg)

    # Step 1:
    # rescale cell metric of relaxed cell to match volume per atom of reference cell
    if not assume_no_cell_relaxation:
        if inert_species is None:
            n_ref = len(reference)
            n_rlx = len(relaxed)
        else:
            n_ref = sum([reference.get_chemical_symbols().count(s) for s in inert_species])
            n_rlx = sum([relaxed.get_chemical_symbols().count(s) for s in inert_species])
        vol_scale = reference.get_volume() / n_ref
        vol_scale /= relaxed.get_volume() / n_rlx
        scaled_relaxed_cell = relaxed.cell * vol_scale ** (1 / 3)
    else:
        scaled_relaxed_cell = relaxed.cell

    # Step 2:
    # get transformation matrix
    P = np.dot(scaled_relaxed_cell, np.linalg.inv(reference.cell))

    # reduce the (real) transformation matrix to the nearest integer one
    P = np.around(P)

    # Step 3:
    # generate supercell of reference structure
    reference_supercell = make_supercell(reference, P, tol=tol_positions)

    return reference_supercell


def _match_positions(relaxed: Atoms, reference: Atoms) -> Tuple[Atoms, float, float]:
    """Matches the atoms in the `relaxed` structure to the sites in the
    `reference` structure. The function returns tuple the first element of which
    is a copy of the `reference` structure, in which the chemical species are
    assigned to comply with the `relaxed` structure. The second and third element
    of the tuple represent the maximum and average distance between relaxed and
    reference sites.

    Parameters
    ----------
    relaxed
        structure with relaxed positions
    reference
        structure with idealized positions

    Raises
    ------
    ValueError
        if the cell metrics of the two input structures do not match
    ValueError
        if the periodic boundary conditions of the two input structures do not match
    ValueError
        if the relaxed structure contains more atoms than the reference structure
    """

    if not np.all(reference.pbc == relaxed.pbc):
        msg = 'The boundary conditions of reference and relaxed structures do not match.'
        msg += '\n  reference: ' + str(reference.pbc)
        msg += '\n  relaxed: ' + str(relaxed.pbc)
        raise ValueError(msg)

    if len(relaxed) > len(reference):
        msg = 'The relaxed structure contains more atoms than the reference structure.'
        msg += '\n  reference: ' + str(len(reference))
        msg += '\n  relaxed: ' + str(len(relaxed))
        raise ValueError(msg)

    if not np.all(np.isclose(reference.cell, relaxed.cell)):
        msg = 'The cell metrics of reference and relaxed structures do not match.'
        msg += '\n  reference: ' + str(reference.cell)
        msg += '\n  relaxed: ' + str(relaxed.cell)
        raise ValueError(msg)

    # compute distances between reference and relaxed positions
    _, dists = get_distances(reference.positions, relaxed.positions,
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
    for i, j in zip(row_ind, col_ind):
        atom = mapped[i]
        if j >= len(relaxed):
            # vacant site in reference structure
            atom.symbol = 'X'
            displacement_magnitudes.append(None)
            displacements.append(3 * [None])
            minimum_distances.append(3 * [None])
        else:
            atom.symbol = relaxed[j].symbol
            dvecs, drs = get_distances([relaxed[j].position],
                                       [reference[i].position],
                                       cell=reference.cell, pbc=reference.pbc)
            displacement_magnitudes.append(drs[0][0])
            displacements.append(dvecs[0][0])
            # distances to the next three available sites
            minimum_distances.append(sorted(dists[:, j])[:3])

    displacement_magnitudes = np.array(displacement_magnitudes, dtype=np.float64)
    mapped.new_array('Displacement', displacements, float, (3, ))
    mapped.new_array('Displacement_Magnitude', displacement_magnitudes, float)
    mapped.new_array('Minimum_Distances', minimum_distances,
                     float, (min(3, len(mapped)), ))

    drmax = np.nanmax(displacement_magnitudes)
    dravg = np.nanmean(displacement_magnitudes)

    return mapped, drmax, dravg
