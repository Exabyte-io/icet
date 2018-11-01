import numpy as np
from ase.neighborlist import NeighborList
from icet.core_py.lattice_site import LatticeSite
import spglib
from icet.tools.geometry import (
    get_fractional_positions_from_ase_neighbor_list,
    find_lattice_site_by_position,
    get_primitive_structure,
    fractional_to_cartesian,
    ase_atoms_to_spglib_cell)

from icet.io.logging import logger
logger = logger.getChild('permutation_matrix')


class PermutationMatrix(object):
    '''
    Permutation matrix object.
    This object can build and store a permutation matrix
    in a couple of different formats. The most important is
    the LatticeSite format.

    The permutation matrix is built up by taking all unique
    fractional positions for the basis atoms and all positions
    of their respective neighbors. Only neigbhors within the
    cutoff will be considered.
    Next, the matrix will be built up column by column by
    taking the fractional position and applying one of the
    allowed crystal symmetry operations for the system.
    The rows of final matrix will then all consist of
    equivalent points in the lattice.
    One can then use this matrix to consider rows of this
    matrix:
    >>> row1, row2 = pm.pm_lattice_sites[i],pm.pm_lattice_sites[j]
    then for each column ,`k`, you can create equivalent pairs:
    >>> equivalent_pairs = []
    >>> for site1, site2 in zip(row1, row2):
    >>>     equivalent_sites.append([site1,site2])
    This grouping together of equivalent sites is what
    makes up an orbit.

    parameters
    ----------
    atoms : ASE Atoms object
    cutoff : float
        this sets the radius for the sites
        to include in the permutation matrix.
    find_prim : bool (default True)
        if True the incoming atoms object will
        be used to construct a primitive structure
        via spglib. This primitive structure will
        then be used to construct the neighbor list,
        be what the lattice sites refer to etc.
    '''

    def __init__(self, atoms, cutoff, find_prim=True):
        atoms = atoms.copy()
        self.cutoff = cutoff

        # set each element to the same since we only care about geometry when
        # taking primitive
        atoms.set_chemical_symbols(len(atoms) * [atoms[0].symbol])

        if find_prim:
            atoms = get_primitive_structure(atoms)

        self.primitive_structure = atoms

        # Get symmetry information and load into a permutation map object
        atoms_as_tuple = ase_atoms_to_spglib_cell(self.primitive_structure)
        symmetry = spglib.get_symmetry(atoms_as_tuple)
        self.translations = symmetry['translations']
        self.rotations = symmetry['rotations']
        self._build()
        self._build_lattice_site_permutation_matrix()
        self._prune_permutation_matrix()
        self.pm_lattice_sites.sort()

    def _build(self):
        """
        First step to build the permutation matrix.
        This is a helper method to initialize permutation matrix.
        """
        # Create neighbor_lists from the different cutoffs
        neighbor_list = NeighborList(len(
            self.primitive_structure) * [self.cutoff / 2.0], skin=1e-8,
            bothways=True, self_interaction=False)
        neighbor_list.update(self.primitive_structure)

        # get fractional positions for neighbor_list
        frac_positions = get_fractional_positions_from_ase_neighbor_list(
            self.primitive_structure, neighbor_list)

        # frac_positions.sort()

        permutation_matrix = []
        for frac_pos in frac_positions:
            # permutation_row = [frac_pos]
            permutation_row = []
            for rotation, translation in zip(
                    self.rotations, self.translations):
                permuted_position = translation + \
                    np.dot(frac_pos, rotation.T)
                permutation_row.append(permuted_position)
            permutation_matrix.append(permutation_row)

        self.permutaded_matrix_frac = permutation_matrix

    def _build_lattice_site_permutation_matrix(self):
        """
        Builds the lattice site version of permutation matrix.

        It will loop through all positions, pos_ij in the matrix,
        convert to lattice sites and construct the matrix,
        lattice_site_ij

        This is a helper method to initialize permutation matrix.
        """
        pm_lattice_sites = []
        for row in self.permutaded_matrix_frac:
            positions = fractional_to_cartesian(self.primitive_structure, row)
            sites = []
            if np.all(self.primitive_structure.pbc):
                sites = [find_lattice_site_by_position(
                    self.primitive_structure, position)
                    for position in positions]
            else:
                for position in positions:
                    try:
                        lattice_site = find_lattice_site_by_position(
                            self.primitive_structure, position)
                        if isinstance(lattice_site, LatticeSite):
                            sites.append(lattice_site)
                    except Exception as e:  # NOQA
                        logger.warning("Skipping exception {}".format(e))
            if len(sites) > 0:
                pm_lattice_sites.append(sites)
            else:
                raise Exception("Lattice sites are empty")

        self.pm_lattice_sites = pm_lattice_sites

    def _prune_permutation_matrix(self):
        """
        Prunes columns in the permutation matrix.
        This is a helper method to initialize permutation matrix.
        """
        for i in range(len(self.pm_lattice_sites)):
            for j in reversed(range(len(self.pm_lattice_sites))):
                if j <= i:
                    continue
                if self.pm_lattice_sites[i][0] == self.pm_lattice_sites[j][0]:
                    self.pm_lattice_sites.pop(j)
                    msg = ['Removing duplicate in permutation matrix']
                    msg += ['i: {} j: {}'.format(i, j)]
                    logger.debug(' '.join(msg))

    @property
    def column1(self):
        """
        Returns column 1 of the lattice site permutation matrix.
        """
        return [row[0] for row in self.pm_lattice_sites]
