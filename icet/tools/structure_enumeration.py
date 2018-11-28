from itertools import product

import numpy as np

from ase import Atoms
from spglib import get_symmetry
from spglib import niggli_reduce as spg_nigg_red
from .structure_enumeration_support.hermite_normal_form \
    import get_reduced_hnfs, HermiteNormalForm
from .structure_enumeration_support.smith_normal_form \
    import get_unique_snfs, SmithNormalForm
from .structure_enumeration_support.labeling_generation \
    import LabelingGenerator
from typing import Dict, List, Tuple, Generator


def _translate_labelings(
        labeling: tuple, snf: SmithNormalForm, nsites: int,
        include_self: bool = False)-> Generator[Tuple[int], None, None]:
    """
    Yields labelings that are equivalent to the original labeling
    under translations as dictated by the Smith normal form (SNF) provided.

    Parameters
    ----------
    labeling
        labeling to be translated
    snf
        SmithNormalForm object
    nsites
        number of sites in the primitive cell
    include_self
        if True original labeling will be included
    """

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


def _get_all_labelings(snf: SmithNormalForm,
                       labeling_generator: LabelingGenerator,
                       nsites: int) -> List[tuple]:
    """
    Returns inequivalent labelings corresponding to a Smith normal
    form (SNF) matrix. Superperiodic labelings as well as labelings that
    are equivalent under translations for this particular SNF will not
    be included. However, labelings that are equivalent by rotations
    that leave the cell (but not the labeling) unchanged will still be
    included, since these have to be removed for each Hermite normal form (HNF)
    separately.

    Parameters
    ----------
    snf
        SmithNormalForm object
    labeling_generator
        LabelingGenerator object
    nsites
        number of sites per primitive cell
    """
    labelings = []
    for labeling in labeling_generator.yield_labelings(snf.ncells):
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


def _permute_labeling(labeling: tuple, snf: SmithNormalForm,
                      transformation: List[np.ndarray],
                      nsites: int) -> Tuple[int]:
    """
    Returns permuted labeling according to transformations defined by
    transformation.

    Parameters
    ----------
    labeling
        labeling to be rotated
    snf
        SmithNormalForm object
    transformation
        transformation consisting of rotation, translation and basis shift
    nsites
        number of sites in the primitive cell
    """

    # Calculate transformation imposed by LRL multiplication
    new_group_order = np.dot(snf.group_order, transformation[0].T)

    # Loop over every atom to find its new position
    labeling_new = [0] * len(labeling)
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


def _yield_unique_labelings(labelings: List[int], snf: SmithNormalForm,
                            hnf: HermiteNormalForm, nsites: int) -> tuple:
    """
    Yields labelings that are unique in every imaginable sense.

    Parameters
    ----------
    labelkeys
        list of hash keys to labelings that may still contain labelings that
        are equivalent under rotations that leave the supercell shape
        unchanged
    snf
        SmithNormalForm object
    hnf
        HermiteNormalForm object
    nsites
        number of sites in the primitive cell
    """
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


def _labeling_to_atoms(labeling: tuple, hnf: np.ndarray, cell: np.ndarray,
                       new_cell: np.ndarray,
                       basis: np.ndarray, chemical_symbols: List[str],
                       pbc: List[bool]) -> Atoms:
    """
    Returns structure object corresponding to the given labeling using
    labeling, HNF matrix and parent lattice.

    Parameters
    ---------
    labeling
        permutation of index of elements
    hnf
        HNF object defining the supercell
    cell
        basis vectors of primtive cell listed row-wise
    new_cell
        new cell shape
    basis
        scaled coordinates to all sites in the primitive cell
    chemical_symbols
        list of elements, e.g. ``['Au', 'Ag']``
    pbc
        periodic boundary conditions of the primitive structure
    """
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
                    symbols.append(chemical_symbols[labeling[count]])
                    count += 1
    atoms = Atoms(symbols, positions, cell=new_cell, pbc=(True, True, True))
    atoms.wrap()
    atoms.pbc = pbc
    return atoms


def get_symmetry_operations(atoms: Atoms,
                            tolerance: float = 1e-3) -> Dict[str, list]:
    """
    Returns symmetry operations permissable for a given structure as
    obtained via `spglib <https://atztogo.github.io/spglib/>`_. The
    symmetry operations consist of three parts: rotation, translation
    and basis shifts. The latter define the way that sublattices
    shift upon rotation (correponds to `d_Nd` in [HarFor09]_).

    Parameters
    ----------
    atoms
        structure for which symmetry operations are sought
    tolerance
        numerical tolerance imposed during symmetry analysis
    """

    symmetries = get_symmetry(atoms)
    assert symmetries, ('spglib.get_symmetry() failed. Please make sure that'
                        ' the atoms object is sensible.')
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
                while site_rot_trans[index] < -tolerance:
                    site_rot_trans[index] += 1
                    site_translation[index] -= 1
                while site_rot_trans[index] > 1 - tolerance:
                    site_rot_trans[index] -= 1
                    site_translation[index] += 1
            site_translations.append(site_translation)

            # Find the basis element that the shifted basis correponds to
            found = False
            for basis_index, basis_element_comp in enumerate(basis):
                distance = site_rot_trans - basis_element_comp

                # Make sure that they do not differ with a basis vector
                for dist_comp_i, dist_comp in enumerate(distance):
                    if abs(abs(dist_comp) - 1) < tolerance:
                        distance[dist_comp_i] = 0

                if (abs(distance) < tolerance).all():
                    assert not found
                    basis_shifts[i, j] = basis_index
                    found = True
            assert found

        sites_translations.append(np.array(site_translations))

    symmetries['translations'] = sites_translations
    symmetries['basis_shifts'] = basis_shifts
    return symmetries


def enumerate_structures(atoms: Atoms, sizes: List[int],
                         chemical_symbols: list,
                         concentration_restrictions: dict = None,
                         niggli_reduce: bool = None) -> Atoms:
    """Yields a sequence of enumerated structures. The function generates
    *all* inequivalent structures that are permissible given a certain
    lattice. Using the ``chemical_symbols`` and ``concentration_restrictions``
    keyword arguments it is possible to specify which chemical_symbols are to
    be included on which site and in which concentration range.

    The function is sensitive to the boundary conditions of the input
    structure. An enumeration of, for example, a surface can thus be performed
    by setting ``atoms.pbc = [True, True, False]``.

    The algorithm implemented here was developed by Gus L. W. Hart and
    Rodney W. Forcade in Phys. Rev. B **77**, 224115 (2008)
    [HarFor08]_ and Phys. Rev. B **80**, 014120 (2009) [HarFor09]_.

    Parameters
    ----------
    atoms
        primitive structure from which derivative superstructures should be
        generated
    sizes
        number of sites (included in the enumeration)
    chemical_symbols
        chemical species with which to decorate the structure, e.g., ``['Au',
        'Ag']``; see below for more examples
    concentration_restrictions
        allowed concentration range for one or more element in
        `chemical_symbols`, e.g., ``{'Au': (0, 0.2)}`` will only
        enumerate structures in which the Au content is between 0 and
        20 %; here, concentration is always defined as the number of
        atoms of the specified kind divided by the number of *all*
        atoms.
    niggli_reduction
        if True perform a Niggli reduction with spglib for each structure;
        the default is ``True`` if ``atoms`` is periodic in all directions,
        ``False`` otherwise.

    Examples
    --------

    The following code snippet illustrates how to enumerate structures
    with up to 6 atoms in the unit cell for a binary alloy without any
    constraints::

        from ase.build import bulk
        prim = bulk('Ag')
        enumerate_structures(atoms=prim, sizes=range(1, 7),
                             chemical_symbols=['Ag', 'Au'])

    To limit the concentration range to 10 to 40% Au the code should
    be modified as follows::

        enumerate_structures(atoms=prim, sizes=range(1, 7),
                             chemical_symbols=['Ag', 'Au'],
                             concentration_restrictions={'Au': (0.1, 0.4)})

    Often one would like to consider mixing on only one
    sublattice. This can be achieved as illustrated for a
    Ga(1-x)Al(x)As alloy as follows::

        prim = bulk('GaAs', crystalstructure='zincblende', a=5.65)
        enumerate_structures(atoms=prim, sizes=range(1, 9),
                             chemical_symbols=[['Ga', 'Al'], ['As']])

    """

    nsites = len(atoms)
    basis = atoms.get_scaled_positions()

    # Construct descriptor of where species are allowed to be
    if isinstance(chemical_symbols[0], str):
        iter_chemical_symbols = [tuple(range(len(chemical_symbols)))] * nsites
        elements = chemical_symbols
    elif len(chemical_symbols) == nsites:
        assert isinstance(chemical_symbols[0][0], str)
        elements = []
        for site in chemical_symbols:
            for element in site:
                if element not in elements:
                    elements.append(element)
        iter_chemical_symbols = []
        for site in chemical_symbols:
            iter_chemical_symbols.append(tuple(elements.index(i)
                                               for i in site))
    else:
        raise Exception('chemical_symbols needs to be a list of strings '
                        'or a list of list of strings.')

    # Adapt concentration restrictions to iter_chemical_symbols
    if concentration_restrictions:
        concentrations = {}
        for key, concentration_range in concentration_restrictions.items():
            assert len(concentration_range) == 2, \
                ('Each concentration range' +
                 ' needs to be specified as (c_low, c_high)')
            if key not in elements:
                raise ValueError('{} found in concentration_restrictions but'
                                 ' not in chemical_symbols'.format(key))
            concentrations[elements.index(key)] = concentration_range
    else:
        concentrations = None

    # Construct labeling generator
    labeling_generator = LabelingGenerator(iter_chemical_symbols,
                                           concentrations)

    # Niggli reduce by default if all directions have
    # periodic boundary conditions
    if niggli_reduce is None:
        niggli_reduce = (sum(atoms.pbc) == 3)

    symmetries = get_symmetry_operations(atoms)

    # Loop over each cell size
    for ncells in sizes:
        if ncells == 0:
            continue

        hnfs = get_reduced_hnfs(ncells, symmetries, atoms.pbc)
        snfs = get_unique_snfs(hnfs)

        for snf in snfs:
            labelings = _get_all_labelings(snf, labeling_generator, nsites)
            for hnf in snf.hnfs:
                if niggli_reduce:
                    new_cell = spg_nigg_red(np.dot(atoms.cell.T, hnf.H).T)
                else:
                    new_cell = np.dot(atoms.cell.T, hnf.H).T
                for labeling in _yield_unique_labelings(labelings, snf, hnf,
                                                        nsites):
                    yield _labeling_to_atoms(labeling, hnf, atoms.cell,
                                             new_cell, basis, elements,
                                             atoms.pbc)
