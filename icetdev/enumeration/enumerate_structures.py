'''
This module has the purpose of enumerating structures. Given a lattice
(possibly with as basis) and a number of elements, the code generates all
the derivative superstructures having a certain size defined by the user.

The algorithm was developed by Gus L. W Hart and Rodney W. Forcade in
* Hart, G. L. W. and Forcade, R. W., Phys. Rev. B 77, 224115 (2008)
* Hart, G. L. W. and Forcade, R. W., Phys. Rev. B 80, 014120 (2009)
'''

from itertools import product
import itertools
import numpy as np
from spglib import get_symmetry, niggli_reduce
from ase import Atoms
from icetdev.enumeration.hermite_normal_form import get_reduced_hnfs
from icetdev.enumeration.smith_normal_form import get_unique_snfs


def _translate_labelings(labeling, snf, nsites, include_self=False):
    '''
    Yield labelings that are equivalent to original labeling
    under translations as dictated by snf.

    Paramters
    ---------
    labeling : tuple
        labeling to be translated
    snf : SmithNormalForm object
    nsites : int
        Number of sites in the primtive cell.
    include_self : bool
        Inlcude original labeling or not.

    Yields
    ------
    tuple of ints
        Translated labeling.
    '''

    # Compute size of each block within which translations occur
    sizes = [nsites * block for block in snf.blocks]

    # Loop over all possible translations within group as defined by snf
    for trans in product(range(snf.S[0]), range(snf.S[1]), range(snf.S[2])):
        if not include_self and sum(trans) == 0:
            continue

        labeling_trans = ()
        for i in range(snf.S[0]):
            group = (i + trans[0]) % snf.S[0]
            block_i = labeling[sizes[0] * group:sizes[0] * (group + 1)]
            for j in range(snf.S[1]):
                group = (j + trans[1]) % snf.S[1]
                block_j = block_i[sizes[1] * group:sizes[1] * (group + 1)]
                for k in range(snf.S[2]):
                    group = (k + trans[2]) % snf.S[2]
                    labeling_trans += tuple(block_j[sizes[2] * group:
                                                    sizes[2] * (group + 1)])
        yield labeling_trans


def _get_labelings(snf, iter_elements, nsites):
    '''
    Get all labelings corresponding to a Smith Normal Form matrix.
    Superperiodic labelings as well as labelings that are equivalent under
    translations for this particular SNF will not be included. However,
    labelings that are equivalent by rotations that leave the cell (but not
    the labeling) unchanged will still be included, since these have to be
    removed for each HNF.

    Paramters
    ---------
    snf : SmithNormalForm object
    nelements : int
        Number of elements in enumeration.
    nsites : int
        Number of sites per primtive cell.

    Returns
    -------
    list of tuples
        Inequivalent labelings.
    '''
    labelings = []
    for labeling in itertools.product(*iter_elements*snf.ncells):
        unique = True
        for labeling_trans in _translate_labelings(labeling, snf, nsites,
                                                   include_self=False):
            # Check whether it translates into itself. If so,
            # then it has been added with a smaller cell.
            if labeling == labeling_trans:
                unique = False
                break

            # Check with previous labelings,
            # if labeling can be translated into a previously
            # added labeling, then it is not unique
            if labeling_trans in labelings:
                unique = False
                break
        if unique:
            labelings.append(labeling)
    return labelings


def _permute_labeling(labeling, snf, transformation, nsites):
    '''
    Permute labeling according to transformations defined by transformation.

    Paramters
    ---------
    labeling : tuple
        Labeling to be rotated
    snf : SmithNormalForm object
    transformation : list of ndarrays
        Transformations consisting of rotation, translation and basis shift
    nsites : int
        Number of sites in the primtive cell.

    Returns
    -------
    tuple of ints
        Permuted labeling.
    '''

    # Calculate transformation imposed by LRL multiplication
    new_group_order = np.dot(snf.group_order, transformation[0].T)

    # Loop over every atom to find its new position
    labeling_new = [0]*len(labeling)
    for member_index, member in enumerate(new_group_order):

        # Transform according to Gp,
        # but each site in the primitive cell also transforms in its own way
        for basis in range(nsites):
            new_cell = member + transformation[1][basis]

            # Calculate new index, first by finding the right block,
            # then by adding the basis index to that block
            new_index = 0
            for i in range(3):
                new_index += (new_cell[i] % snf.S[i]) * snf.blocks[i] * nsites
            new_index += transformation[2][basis]

            # Add the contribution to the hash key
            element = labeling[member_index * nsites + basis]
            labeling_new[new_index] = element
    return tuple(labeling_new)


def _yield_unique_labelings(labelings, snf, hnf, nsites):
    '''
    Yield labelings that are unique in every imaginable sense.

    Parameters
    ----------
    labelkeys : list of ints
        List of hash keys to labelings that may still contain labelings that
        are equivalent under rotations that leaves the supercell shape
        unchanged.
    snf : SmithNormalForm object
    hnf : HermiteNormalForm object
    nsites : int
        Number of sites in the primitive cell.

    Yields
    ------
    tuple
        Labeling, each and every one unique.
    '''
    saved_labelings = []
    for labeling in labelings:

        # Check whether labeling is just a rotated version of a previous
        # labeling. Apply transformation that is specific to the hnf
        # and check all translations of the transformed labeling.
        unique = True
        for transformation in hnf.transformations:

            labeling_rot = _permute_labeling(labeling, snf, transformation,
                                             nsites)

            # Commonly, the transformation leaves the labeling
            # unchanged, so check that first as a special case
            # (yields a quite significant speedup)
            if labeling_rot == labeling:
                continue

            # Translate in all possible ways
            for labeling_rot_trans in \
                    _translate_labelings(labeling_rot, snf, nsites,
                                         include_self=True):
                if labeling_rot_trans in saved_labelings:
                    # Then we have rotated and translated the labeling
                    # into one that was already yielded
                    unique = False
                    break
            if not unique:
                break
        if unique:
            # Then we have finally found a unique structure
            # defined by an HNF matrix and a labeling
            saved_labelings.append(labeling)
            yield labeling


def get_symmetry_operations(atoms, tol=1e-3):
    '''
    Use spglib to calculate the symmetry operations of atoms. The
    symmetry operations consist of three parts: rotations, translation
    and "basis_shifts". The latter define the way that the sublattices
    shift upon rotation (correponds to d_Nd in HarFor09).

    Parameters
    ----------
    atoms : ASE Atoms
        Structure for which the symmetry operations are sought.

    Returns
    -------
    dict of lists
        Containing rotations, translations and basis_shifts.

    '''

    symmetries = get_symmetry(atoms)
    rotations = symmetries['rotations']
    translations = symmetries['translations']

    basis = atoms.get_scaled_positions()

    # Calculate how atoms within the primitive cell are shifted (from one site
    # to another) and translated (from one primtive cell to another) upon
    # operation with rotation matrix. Note that the translations are needed
    # here because different sites translate differently.
    basis_shifts = np.zeros((len(rotations), len(atoms)), dtype='int64')
    sites_translations = []
    for i, rotation in enumerate(rotations):
        translation = translations[i]
        site_translations = []
        for j, basis_element in enumerate(basis):

            # Calculate how the site is transformed when operated on by
            # symmetry of parent lattice (rotation and translation)
            site_rot_trans = np.dot(rotation, basis_element) + translation

            # The site may now have been moved to a different site in a
            # different cell. We want to separate the two. (In HarFor09,
            # basis_shift (site_rot_trans) corresponds to d_Nd and
            # site_translation to t_Nd)
            site_translation = [0, 0, 0]
            for index in range(3):
                while site_rot_trans[index] < -tol:
                    site_rot_trans[index] += 1
                    site_translation[index] -= 1
                while site_rot_trans[index] > 1 - tol:
                    site_rot_trans[index] -= 1
                    site_translation[index] += 1
            site_translations.append(site_translation)

            # Find the basis element that the shifted basis correponds to
            found = False
            for basis_index, basis_element_comp in enumerate(basis):
                if (abs(site_rot_trans - basis_element_comp) < tol).all():
                    assert not found
                    basis_shifts[i, j] = basis_index
                    found = True
            assert found

        sites_translations.append(np.array(site_translations))

    symmetries['translations'] = sites_translations
    symmetries['basis_shifts'] = basis_shifts
    return symmetries


def _labeling_to_atoms(labeling, hnf, cell, new_cell, basis, elements):
    '''
    Get ASE Atoms object from labeling, HNF matrix and parent lattice.

    Parameters
    ---------
    labeling : tuple
        Permutation of index of elements.
    hnf : ndarray
        HNF object defining the supercell.
    cell : ndarray
        Basis vectors of primtive cell listed row-wise.
    new_cell : ndarray
        New cell shape.
    basis : ndarray
        Scaled coordinates to all sites in the primitive cell.
    elements : list of str
        List of elements, e.g. ['Au', 'Ag']

    Returns
    -------
    ASE Atoms
        Atoms object corresponding to the given labeling.
    '''
    symbols = []
    positions = []
    count = 0
    for i in range(hnf.H[0, 0]):
        coord = i * hnf.H[1, 0]
        offset10 = coord // hnf.H[0, 0] + coord % hnf.H[0, 0]
        coord = i * hnf.H[2, 0]
        offset20 = coord // hnf.H[0, 0] + coord % hnf.H[0, 0]
        for j in range(hnf.H[1, 1]):
            coord = j * hnf.H[2, 1]
            offset21 = coord // hnf.H[1, 1] + coord % hnf.H[1, 1]
            for k in range(hnf.H[2, 2]):
                for basis_vector in basis:
                    positions.append(i * cell[0] +
                                     (j + offset10) * cell[1] +
                                     (k + offset20 + offset21) * cell[2] +
                                     np.dot(cell.T, basis_vector))
                    symbols.append(elements[labeling[count]])
                    count += 1
    atoms = Atoms(symbols, positions, cell=new_cell, pbc=(True, True, True))
    atoms.wrap()
    return atoms


def enumerate_structures(atoms, sizes, subelements, niggli_reduction=True):
    '''
    Generate enumerated structures, i.e. all inequivalent structures up to a
    certain size

    Paramters
    ---------
    atoms : ASE Atoms
        Primitive structure from which derivative superstructures should be
        generated.
    sizes : list of ints
        Maximum number of atoms in the returned structures.
    subelements : list of str
        Elements to decorate the structure, e.g. ['Au', 'Ag']
    niggle_reduction : bool
        If True, perform a Niggli reduction with spglib for each structure.

    Yields
    ------
    ASE Atoms
        Enumerated structure, each and every unique.

    '''

    nsites = len(atoms)
    basis = atoms.get_scaled_positions()

    # Construct descriptor of where species are allowed to be
    if isinstance(subelements[0], str):
        iter_elements = [range(len(subelements))]*nsites
        elements = subelements
    elif len(subelements) == nsites:
        assert isinstance(subelements[0][0], str)
        elements = []
        for site in subelements:
            for element in site:
                if element not in elements:
                    elements.append(element)
        iter_elements = []
        for site in subelements:
            iter_elements.append([elements.index(i) for i in site])
    else:
        raise Exception('subelements needs to be a list of strings '
                        'or a list of list of strings.')

    symmetries = get_symmetry_operations(atoms)

    # Loop over each cell size
    for ncells in sizes:
        if ncells == 0:
            continue

        hnfs = get_reduced_hnfs(ncells, symmetries)
        snfs = get_unique_snfs(hnfs)

        for snf in snfs:
            labelings = _get_labelings(snf, iter_elements, nsites)
            for hnf in snf.hnfs:
                if niggli_reduction:
                    new_cell = niggli_reduce(np.dot(atoms.cell.T, hnf.H).T)
                else:
                    new_cell = np.dot(atoms.cell.T, hnf.H).T
                for labeling in _yield_unique_labelings(labelings, snf, hnf,
                                                        nsites):
                    yield _labeling_to_atoms(labeling, hnf, atoms.cell,
                                             new_cell, basis, elements)
