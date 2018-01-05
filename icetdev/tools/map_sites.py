from ase.build import cut
import numpy as np


def map_structure_to_reference(input_structure,
                               reference_structure,
                               tolerance_mapping,
                               vacancy_type=None,
                               inert_species=None,
                               tolerance_cell=0.05,
                               tolerance_positions=0.01,
                               verbose=False):
    '''
    Map a relaxed structure onto a reference structure.

    Parameters
    ----------
    input_structure :  ASE Atoms object
        relaxed input structure
    reference_structure : ASE Atoms object
        reference structure, which can but need not represent the primitive
        cell
    tolerance_mapping : float
        maximum allowed displacement for mapping an atom in the relaxed (but
        rescaled) structure to the reference supercell

        *Note*: A reasonable choice is up to 20-30% of the first
        nearest neighbor distance (`r1`). A value above 50% of `r1`
        will most likely lead to atoms being multiply assigned, which
        will raise an exception.
    vacancy_type : str
        If this parameter is set to a non-zero string unassigned sites in the
        reference structure will be assigned to this type.

        *Note 1*: By default (``None``) the method will print an error
        message and raise an exception if there are *any* unassigned
        sites in the reference structure.

        *Note 2*: `vacancy_type` must be a valid element type as
        enforced by the ASE Atoms class.
    inert_species : list of str
        List of chemical symbols (e.g., `['Au', 'Pd']`) that are never
        substituted for a vacancy. Used to make an initial rescale of the cell
        and thus increases the probability for a successful mapping. Need not
        be specified if `vacancy_type` is None.
    tolerance_cell : float
        tolerance factor applied when computing permutation matrix to generate
        supercell (h = P h_p)
    tolerance_positions : float
        tolerance factor applied when scanning for overlapping positions in
        Angstrom (forwarded to `ase.build.cut`)
    verbose : bool
        turn on verbose output

    Returns
    -------
    ASE Atoms, float, float
        - the ideal supercell most closely matching the input structure
        - the largets deviation of any input coordinate from its ideal
          coordinate
        - the average deviation of the input coordinates from the ideal
          coordinates

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

    '''
    if np.any(input_structure.pbc != reference_structure.pbc):
        s = '''The periodic boundary conditions differ
        between input and reference structure'''
        raise Exception(s)

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

    if (len(ideal_supercell) != len(scaled_structure) and
            vacancy_type is None):
        s = 'Number of atoms in ideal supercell does not'
        s += ' match input structure.\n'
        s += 'ideal: {}\n'.format(len(ideal_supercell))
        s += 'input: {}'.format(len(scaled_structure))
        raise Exception(s)

    if verbose:
        np.set_printoptions(suppress=True, precision=6)
        print('Number of atoms in reference structure:'
              ' {}'.format(len(reference_structure)))
        print('Number of atoms in input structure:'
              ' {}\n'.format(len(input_structure)))
        print('Reference cell metric:\n'
              '{}'.format(reference_structure.cell))
        print('Input cell metric:\n'
              '{}\n'.format(input_structure.cell))
        print('Transformation matrix connecting reference structure'
              ' and idealized input structure:\n {}'.format(P))
        print('Determinant of tranformation matrix:'
              ' {:.3f}\n'.format(np.linalg.det(P)))
        print('Cell metric of ideal supercell:\n'
              '{}'.format(ideal_supercell.cell))
        print('Cell metric of rescaled input structure:\n'
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
                    s = 'More than one atom from the relaxed'
                    s += ' (and rescaled) structure have been'
                    s += ' mapped onto the same ideal site.\n'
                    s += ' Try reducing `tolerance_mapping`.'
                    raise Exception(s)
                mapped[ideal_site.index] = atom.index
                drs[ideal_site.index] = dr
                ideal_site.symbol = atom.symbol
                dr_max = max(dr, dr_max)
                dr_sum += dr
                dr_sumsq += dr * dr
                break
        else:
            if vacancy_type is not None:
                try:
                    ideal_site.symbol = vacancy_type
                except:
                    s = 'Failed to assign "{}" as vacancy type.\n'
                    s += 'Check whether `vacancy_type` represents a'
                    s += ' valid element symbol.'.format(vacancy_type)
                    raise Exception(s)
            else:
                s = 'Failed to assign an atom from the relaxed (and'
                s += ' rescaled) structure to the ideal lattice.'
                s += ' Try increasing `tolerance_mapping`.\n'
                s += ' {}'.format(ideal_site)
                raise Exception(s)

    dr_avg = dr_sum / len(ideal_supercell)
    dr_sdv = np.sqrt(dr_sumsq / len(ideal_supercell) - dr_avg ** 2)

    # Check that not more than one atom was assigned to the same site
    for k in set(mapped):
        if k >= 0 and mapped.count(k) > 1:
            s = 'Site {} has been assigned more than once.'.format(k)
            raise Exception(s)

    # Check that the chemical composition of input and ideal supercell matches
    for element in set(input_structure.get_chemical_symbols()):
        n1 = input_structure.get_chemical_symbols().count(element)
        n2 = ideal_supercell.get_chemical_symbols().count(element)
        if n1 != n2:
            s = '''Number of atoms of type {} differs between
            input structure ({}) and ideal
            supercell ({}).\n'''.format(element, n1, n2)
            print(s)
            raise Exception(s)

    if verbose:
        print('Maximum, average and standard deviation of atomic'
              ' displacements: {} {} {}'.format(dr_max, dr_avg, dr_sdv))

        print('{:52} {:}'.format('Input structure:',
                                 'Scaled structure:'))
        for k, (input_atom, scaled_atom) in enumerate(
                zip(input_structure,
                    scaled_structure)):
            print('{:4}'.format(k), end='')
            print(' {:2}'.format(input_atom.symbol), end='')
            print(' {:12.6f} {:12.6f} {:12.6f}'.format(*input_atom.position),
                  end='')
            print('    -->', end='')
            print(' {:12.6f} {:12.6f} {:12.6f}'.format(*scaled_atom.position),
                  end='')
            print('')
        print('')

        print('{:52} {}'.format('Ideal supercell:',
                                'Scaled structure:'))
        for ideal_atom, k, dr in zip(ideal_supercell, mapped, drs):
            print(' {:2}'.format(ideal_atom.symbol), end='')
            print((3 * '  {:12.6f}').format(*ideal_atom.position), end='')
            print('    -->', end='')
            print(' {:4}'.format(k), end='')
            if k >= 0:
                scaled_pos = scaled_structure[k].position
                print((3 * ' {:12.6f}').format(*scaled_pos), end='')
                print('    --> {:.4}'.format(dr), end='')
            print('')

    return ideal_supercell, dr_max, dr_avg


def _get_scaled_cell(input_structure, reference_structure, vacancy_type=None,
                     inert_species=None):
    '''
    The input structure needs to be scaled in order to match the lattice
    structure of the reference structure. The reference structure can be a
    primitive cell, in which case the input structure would usually be a
    supercell thereof. Also, we need an ideal supercell that matches the input
    structure.

    Parameters
    ----------
    input_structure : ASE Atoms object
        relaxed input structure
    reference_structure : ASE Atoms object
        reference structure, which can but need not represent the primitive
        cell
    vacancy_type : str
        If not None, the cell is scaled if and only if `inert_species` is not
        None.
    inert_species : list of str
        List of chemical symbols (e.g., `['Au', 'Pd']`) that are never
        substituted for a vacancy. Needless if `vacancy_type` is `None`.
    '''
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
            # We can not use the number of atoms since there may be vacancies
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


def _get_transformation_matrix(input_cell, reference_cell, tolerance_cell=0.05):
    '''
    Obtain the (in general non-integer) transformation matrix connecting the
    input structure to the reference structure L = L_p.P --> P = L_p^-1.L

    Parameters
    ----------
    input_cell : NumPy array (3, 3)
        Cell metric of input structure (possibly scaled)
    reference_cell : NumPy array (3, 3)
        Cell metric of reference structure
    tolerance_cell : float
        Tolerance for how much the elements of P are allowed to deviate from
        nearest integer before they are rounded.

    Returns
    -------
    NumPy array (3, 3)
        Transformation matrix P of integers.
    '''
    P = np.dot(input_cell, np.linalg.inv(reference_cell))

    # assert that the transformation matrix does not deviate too
    # strongly from the nearest integer matrix
    if np.linalg.norm(P - np.around(P)) / 9 > tolerance_cell:
        s = 'Failed to map structure to reference'
        s += 'structure (tolerance_cell exceeded).\n'
        s += 'reference:\n {}\n'.format(reference_cell)
        s += 'input:\n {}\n'.format(input_cell)
        s += 'P:\n {}\n'.format(P)
        s += 'P_round:\n {}\n'.format(np.around(P))
        s += 'Deviation: {}\n'.format(np.linalg.norm(P - np.around(P)) / 9)
        s += 'If there are vacancies, you can try specifying `inert_species`.'
        s += ' Else, you can try raising `tolerance_cell`.'
        raise Exception(s)

    # reduce the (real) transformation matrix to the nearest integer one
    P = np.around(P)
    return P


def _rescale_structures(input_structure, reference_structure, P,
                        tolerance_positions=0.01):
    '''
    Rescale `input_structure` with `P` so that it matches
    `reference_structure`, and make a supercell of `reference_structure` using
    `P`
    
    Parameters
    ----------
    input_structure : ASE Atoms object
        relaxed input structure
    reference_structure : ASE Atoms object
        reference structure, which can but need not represent the primitive
        cell
    P : NumPy array (3, 3)
        Transformation matrix of integers.
    tolerance_positions : float
        tolerance factor applied when scanning for overlapping positions in
        Angstrom (forwarded to `ase.build.cut`).

    Returns
    -------
    ASE Atoms object
        Scaled version of `input_structure`
    ASE Atoms object
        Supercell of `reference_structure` matching cell metric of
        `scaled_structure`
    '''
    scaled_structure = input_structure.copy()
    scaled_structure.set_cell(np.dot(P, reference_structure.cell),
                              scale_atoms=True)

    # generate supercell of (presumably primitive) reference structure
    ideal_supercell = cut(reference_structure,
                          P[0], P[1], P[2],
                          tolerance=tolerance_positions)
    assert(len(ideal_supercell) ==
           int(np.round(len(reference_structure) * np.linalg.det(P))))

    return scaled_structure, ideal_supercell
