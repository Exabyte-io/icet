"""
This module provides the ClusterSpace class.
"""

import copy
import itertools
import pickle
import tarfile
import tempfile
from collections import OrderedDict
from typing import List, Union

import numpy as np

from _icet import ClusterSpace as _ClusterSpace
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from icet.core.orbit_list import OrbitList
from icet.core.structure import Structure
from icet.core.sublattices import Sublattices
from icet.tools.geometry import (add_vacuum_in_non_pbc,
                                 get_decorated_primitive_structure)


class ClusterSpace(_ClusterSpace):
    """
    This class provides functionality for generating and maintaining
    cluster spaces.

    **Note:**
    In icet all :class:`ase.Atoms` objects must have periodic boundary
    conditions. When carrying out cluster-expansions for surfaces and
    nano-particles it is therefore recommended to embed the atoms
    object in a vacuum and use periodic boundary conditions. This can
    be done using e.g., :func:`ase.Atoms.center`.

    Parameters
    ----------
    atoms : ase.Atoms
        atomic configuration
    cutoffs : list(float)
        cutoff radii per order that define the cluster space

        Cutoffs are specified in units of Angstrom and refer to the
        longest distance between two atoms in the cluster. The first
        element refers to pairs, the second to triplets, the third
        to quadruplets, and so on. ``cutoffs=[7.0, 4.5]`` thus implies
        that all pairs distanced 7 A or less will be included,
        as well as all triplets among which the longest distance is no
        longer than 4.5 A.
    chemical_symbols : list(str) or list(list(str))
        list of chemical symbols, each of which must map to an element
        of the periodic table

        If a list of chemical symbols is provided, all sites on the
        lattice will have the same allowed occupations as the input
        list.

        If a list of list of chemical symbols is provided then the
        outer list must be the same length as the atoms object and
        ``chemical_symbols[i]`` will correspond to the allowed species
        on lattice site ``i``.

    Examples
    --------
    The following snippets illustrate several common situations::

        from ase.build import bulk
        from ase.io import read
        from icet import ClusterSpace

        # AgPd alloy with pairs up to 7.0 A and triplets up to 4.5 A
        prim = bulk('Ag')
        cs = ClusterSpace(atoms=prim, cutoffs=[7.0, 4.5],
                          chemical_symbols=[['Ag', 'Pd']])
        print(cs)

        # (Mg,Zn)O alloy on rocksalt lattice with pairs up to 8.0 A
        prim = bulk('MgO', crystalstructure='rocksalt', a=6.0)
        cs = ClusterSpace(atoms=prim, cutoffs=[8.0],
                          chemical_symbols=[['Mg', 'Zn'], ['O']])
        print(cs)

        # (Ga,Al)(As,Sb) alloy with pairs, triplets, and quadruplets
        prim = bulk('GaAs', crystalstructure='zincblende', a=6.5)
        cs = ClusterSpace(atoms=prim, cutoffs=[7.0, 6.0, 5.0],
                          chemical_symbols=[['Ga', 'Al'], ['As', 'Sb']])
        print(cs)

        # PdCuAu alloy with pairs and triplets
        prim = bulk('Pd')
        cs = ClusterSpace(atoms=prim, cutoffs=[7.0, 5.0],
                          chemical_symbols=[['Au', 'Cu', 'Pd']])
        print(cs)


    """

    def __init__(self,
                 atoms: Atoms,
                 cutoffs: List[float],
                 chemical_symbols: Union[List[str], List[List[str]]]) -> None:

        if not isinstance(atoms, Atoms):
            raise TypeError('Input configuration must be an ASE Atoms object'
                            ', not type {}'.format(type(atoms)))
        if not all(atoms.pbc):
            raise ValueError('Input structure must have periodic boundary '
                             'condition')

        self._cutoffs = cutoffs.copy()
        self._input_atoms = atoms.copy()
        self._input_chemical_symbols = copy.deepcopy(chemical_symbols)
        chemical_symbols = self._get_chemical_symbols()

        self._pruning_history = []

        # set up primitive
        decorated_primitive, primitive_chemical_symbols = get_decorated_primitive_structure(
            self._input_atoms, chemical_symbols)
        self._primitive_chemical_symbols = primitive_chemical_symbols
        assert len(decorated_primitive) == len(primitive_chemical_symbols)

        # set up orbit list
        self._orbit_list = OrbitList(decorated_primitive, self._cutoffs)
        self._orbit_list.remove_inactive_orbits(primitive_chemical_symbols)

        # call (base) C++ constructor
        _ClusterSpace.__init__(
            self, primitive_chemical_symbols, self._orbit_list)

    def _get_chemical_symbols(self):
        """ Returns chemical symbols using input atoms and input
        chemical symbols. Carries out multiple sanity checks. """

        # setup chemical symbols as List[List[str]]
        if all(isinstance(i, str) for i in self._input_chemical_symbols):
            chemical_symbols = [
                self._input_chemical_symbols] * len(self._input_atoms)
        elif not all(isinstance(i, list) for i in self._input_chemical_symbols):
            raise TypeError("chemical_symbols must be List[str] or List[List[str]], not {}".format(
                type(self._input_chemical_symbols)))
        elif len(self._input_chemical_symbols) != len(self._input_atoms):
            msg = 'chemical_symbols must have same length as atoms. '
            msg += 'len(chemical_symbols = {}, len(atoms)= {}'.format(
                len(self._input_chemical_symbols), len(self._input_atoms))
            raise ValueError(msg)
        else:
            chemical_symbols = copy.deepcopy(self._input_chemical_symbols)

        for i, symbols in enumerate(chemical_symbols):
            if len(symbols) != len(set(symbols)):
                raise ValueError(
                    'Found duplicates of allowed chemical symbols on site {}.'
                    ' allowed species on  site {}= {}'.format(i, i, symbols))

        if len([tuple(sorted(s)) for s in chemical_symbols if len(s) > 1]) == 0:
            raise ValueError('No active sites found')

        return chemical_symbols

    def _get_chemical_symbol_representation(self):
        """Returns a str version of the chemical symbols that is
        easier on the eyes.
        """
        sublattices = self.get_sublattices(self.primitive_structure)
        nice_str = []
        for sublattice in sublattices.active_sublattices:
            sublattice_symbol = sublattice.symbol

            nice_str.append('{} (sublattice {})'.format(
                list(sublattice.chemical_symbols), sublattice_symbol))
        return ', '.join(nice_str)

    def _get_string_representation(self,
                                   print_threshold: int = None,
                                   print_minimum: int = 10) -> str:
        """
        String representation of the cluster space that provides an overview of
        the orbits (order, radius, multiplicity etc) that constitute the space.

        Parameters
        ----------
        print_threshold
            if the number of orbits exceeds this number print dots
        print_minimum
            number of lines printed from the top and the bottom of the orbit
            list if `print_threshold` is exceeded

        Returns
        -------
        multi-line string
            string representation of the cluster space.
        """

        def repr_orbit(orbit, header=False):
            formats = {'order': '{:2}',
                       'radius': '{:8.4f}',
                       'multiplicity': '{:4}',
                       'index': '{:4}',
                       'orbit_index': '{:4}',
                       'multi_component_vector': '{:}',
                       'sublattices': '{:}'}
            s = []
            for name, value in orbit.items():
                str_repr = formats[name].format(value)
                n = max(len(name), len(str_repr))
                if header:
                    s += ['{s:^{n}}'.format(s=name, n=n)]
                else:
                    s += ['{s:^{n}}'.format(s=str_repr, n=n)]
            return ' | '.join(s)

        # basic information
        # (use largest orbit to obtain maximum line length)
        prototype_orbit = self.orbit_data[-1]
        width = len(repr_orbit(prototype_orbit))
        s = []  # type: List
        s += ['{s:=^{n}}'.format(s=' Cluster Space ', n=width)]
        s += [' chemical species: {}'
              .format(self._get_chemical_symbol_representation())]
        s += [' cutoffs: {}'.format(' '.join(['{:.4f}'.format(co)
                                              for co in self._cutoffs]))]
        s += [' total number of orbits: {}'.format(len(self))]
        t = ['{}= {}'.format(k, c)
             for k, c in self.get_number_of_orbits_by_order().items()]
        s += [' number of orbits by order: {}'.format('  '.join(t))]

        # table header
        s += [''.center(width, '-')]
        s += [repr_orbit(prototype_orbit, header=True)]
        s += [''.center(width, '-')]

        # table body
        index = 0
        orbit_list_info = self.orbit_data
        while index < len(orbit_list_info):
            if (print_threshold is not None and
                    len(self) > print_threshold and
                    index >= print_minimum and
                    index <= len(self) - print_minimum):
                index = len(self) - print_minimum
                s += [' ...']
            s += [repr_orbit(orbit_list_info[index])]
            index += 1
        s += [''.center(width, '=')]

        return '\n'.join(s)

    def __repr__(self) -> str:
        """ String representation. """
        return self._get_string_representation(print_threshold=50)

    def print_overview(self,
                       print_threshold: int = None,
                       print_minimum: int = 10) -> None:
        """
        Print an overview of the cluster space in terms of the orbits (order,
        radius, multiplicity etc).

        Parameters
        ----------
        print_threshold
            if the number of orbits exceeds this number print dots
        print_minimum
            number of lines printed from the top and the bottom of the orbit
            list if `print_threshold` is exceeded
        """
        print(self._get_string_representation(print_threshold=print_threshold,
                                              print_minimum=print_minimum))

    @property
    def orbit_data(self) -> List[dict]:
        """
        list of orbits with information regarding
        order, radius, multiplicity etc
        """
        data = []
        zerolet = OrderedDict([('index', 0),
                               ('order', 0),
                               ('radius', 0),
                               ('multiplicity', 1),
                               ('orbit_index', -1),
                               ('multi_component_vector', '.'),
                               ('sublattices', '.')])
        sublattices = self.get_sublattices(self.primitive_structure)
        data.append(zerolet)
        index = 1
        while index < len(self):
            cluster_space_info = self.get_cluster_space_info(index)
            orbit_index = cluster_space_info[0]
            mc_vector = cluster_space_info[1]
            orbit = self.get_orbit(orbit_index)
            rep_sites = orbit.get_representative_sites()
            orbit_sublattices = '-'.join(
                [sublattices[sublattices.get_sublattice_index(ls.index)].symbol
                 for ls in rep_sites])
            local_Mi = self.get_number_of_allowed_species_by_site(
                self._get_primitive_structure(), orbit.representative_sites)
            mc_vectors = orbit.get_mc_vectors(local_Mi)
            mc_permutations = self.get_multi_component_vector_permutations(
                mc_vectors, orbit_index)
            mc_index = mc_vectors.index(mc_vector)
            mc_permutations_multiplicity = len(mc_permutations[mc_index])
            cluster = self.get_orbit(orbit_index).get_representative_cluster()

            multiplicity = len(self.get_orbit(
                orbit_index).get_equivalent_sites())
            record = OrderedDict([('index', index),
                                  ('order', cluster.order),
                                  ('radius', cluster.radius),
                                  ('multiplicity', multiplicity *
                                   mc_permutations_multiplicity),
                                  ('orbit_index', orbit_index)])
            record['multi_component_vector'] = mc_vector
            record['sublattices'] = orbit_sublattices
            data.append(record)
            index += 1
        return data

    def get_number_of_orbits_by_order(self) -> OrderedDict:
        """
        Returns the number of orbits by order.

        Returns
        -------
        an ordered dictionary where keys and values represent order and number
        of orbits, respectively
        """
        count_orbits = {}  # type: dict[int, int]
        for orbit in self.orbit_data:
            k = orbit['order']
            count_orbits[k] = count_orbits.get(k, 0) + 1
        return OrderedDict(sorted(count_orbits.items()))

    def get_cluster_vector(self, atoms: Atoms) -> np.ndarray:
        """
        Returns the cluster vector for a structure.

        Parameters
        ----------
        atoms
            atomic configuration

        Returns
        -------
        the cluster vector
        """
        if not isinstance(atoms, Atoms):
            raise TypeError('input structure must be an ASE Atoms object')

        try:
            cv = _ClusterSpace.get_cluster_vector(self, Structure.from_atoms(atoms))
        except Exception as e:
            self.assert_structure_compatability(atoms)
            raise(e)
        return cv

    def _prune_orbit_list(self, indices: List[int]) -> None:
        """
        Prunes the internal orbit list

        Parameters
        ----------
        indices
            indices to all orbits to be removed
        """
        size_before = len(self._orbit_list)

        self._prune_orbit_list_cpp(indices)
        for index in sorted(indices, reverse=True):
            self._orbit_list.remove_orbit(index)
        self._precompute_multi_component_vectors()

        size_after = len(self._orbit_list)
        assert size_before - len(indices) == size_after
        self._pruning_history.append(indices)

    @property
    def primitive_structure(self) -> Atoms:
        """
        Primitive structure on which the cluster space is based
        """
        atoms = self._get_primitive_structure().to_atoms()
        # Decorate with the "real" symbols (instead of H, He, Li etc)
        for atom, symbols in zip(atoms, self._primitive_chemical_symbols):
            atom.symbol = min(symbols)
        return atoms

    @property
    def chemical_symbols(self) -> List[List[str]]:
        """
        Chemical species considered
        """
        return self._primitive_chemical_symbols.copy()

    @property
    def cutoffs(self) -> List[float]:
        """
        Cutoffs for the different n-body clusters. Each cutoff radii
        (in Angstroms) defines the largest inter-atomic distance in each
        cluster
        """
        return self._cutoffs

    @property
    def orbit_list(self):
        """Orbit list that defines the cluster in the cluster space"""
        return self._orbit_list

    def get_possible_orbit_decorations(self, orbit_index: int) \
            -> List[List[str]]:
        """Returns possible decorations on the orbit

        Parameters
        ----------
        orbit_index
        """
        orbit = self.orbit_list.orbits[orbit_index]

        indices = [
            lattice_site.index for lattice_site in orbit.representative_sites]

        allowed_species = [self.chemical_symbols[index] for index in indices]

        return list(itertools.product(*allowed_species))

    def get_sublattices(self, structure: Atoms) -> Sublattices:
        """ Returns the sublattices of the input structure

        Parameters
        ----------
        structure
            structure the sublattices are based on
        """
        sl = Sublattices(self.chemical_symbols, self.primitive_structure, structure)
        return sl

    def assert_structure_compatability(self, structure: Atoms, vol_tol: float = 1e-5) -> None:
        """ Raises if structure is not compatible with ClusterSpace.

        TODO: Add check for if structure is relaxed

        Parameters
        ----------
        structure
            structure to check if compatible with ClusterSpace
        """
        # check volume
        prim = self.primitive_structure
        vol1 = prim.get_volume() / len(prim)
        vol2 = structure.get_volume() / len(structure)
        if abs(vol1 - vol2) > vol_tol:
            raise ValueError('Volume per atom of structure does not match the volume of'
                             'ClusterSpace.primitive_structure')

        # check occupations
        sublattices = self.get_sublattices(structure)
        sublattices.assert_occupation_is_allowed(structure.get_chemical_symbols())

    def is_supercell_self_correlated(self, atoms: Atoms) -> bool:
        """
        Check whether an atoms object self-interacts via periodic
        boundary conditions.

        Parameters
        ----------
        atoms
            An atoms object to check self-interaction for

        Returns
        -------
        bool
            If True, the atoms object self-interacts via periodic
            boundary conditions, otherwise False.
        """
        ol = self.orbit_list.get_supercell_orbit_list(atoms)
        orbit_indices = set()
        for orbit in ol.orbits:
            for sites in orbit.get_equivalent_sites():
                indices = tuple(sorted([site.index for site in sites]))
                if indices in orbit_indices:
                    return True
                else:
                    orbit_indices.add(indices)
        return False

    def write(self, filename: str) -> None:
        """
        Saves cluster space to a file.

        Parameters
        ---------
        filename
            name of file to which to write
        """

        with tarfile.open(name=filename, mode='w') as tar_file:

            # write items
            items = dict(cutoffs=self._cutoffs, chemical_symbols=self._input_chemical_symbols,
                         pruning_history=self._pruning_history)
            temp_file = tempfile.TemporaryFile()
            pickle.dump(items, temp_file)
            temp_file.seek(0)
            tar_info = tar_file.gettarinfo(arcname='items', fileobj=temp_file)
            tar_file.addfile(tar_info, temp_file)
            temp_file.close()

            # write atoms
            temp_file = tempfile.NamedTemporaryFile()
            ase_write(temp_file.name, self._input_atoms, format='json')
            temp_file.seek(0)
            tar_info = tar_file.gettarinfo(arcname='atoms', fileobj=temp_file)
            tar_file.addfile(tar_info, temp_file)

    @staticmethod
    def read(filename: str):
        """
        Reads cluster space from filename.

        Parameters
        ---------
        filename
            name of file from which to read cluster space
        """
        if isinstance(filename, str):
            tar_file = tarfile.open(mode='r', name=filename)
        else:
            tar_file = tarfile.open(mode='r', fileobj=filename)

        # read items
        items = pickle.load(tar_file.extractfile('items'))

        # read atoms
        temp_file = tempfile.NamedTemporaryFile()
        temp_file.write(tar_file.extractfile('atoms').read())
        temp_file.seek(0)
        atoms = ase_read(temp_file.name, format='json')

        tar_file.close()
        cs = ClusterSpace(
            atoms=atoms, cutoffs=items['cutoffs'], chemical_symbols=items['chemical_symbols'])
        for indices in items['pruning_history']:
            cs._prune_orbit_list(indices)
        return cs

    def copy(self):
        """ Returns copy of ClusterSpace instance. """
        atoms = self._input_atoms
        cutoffs = self._cutoffs
        chemical_symbols = self._input_chemical_symbols
        cs_copy = ClusterSpace(atoms, cutoffs, chemical_symbols)
        for indices in self._pruning_history:
            cs_copy._prune_orbit_list(indices)
        return cs_copy


def get_singlet_info(atoms: Atoms,
                     return_cluster_space: bool = False):
    """
    Retrieves information concerning the singlets in the input structure.

    Parameters
    ----------
    atoms
        atomic configuration
    return_cluster_space
        if True return the cluster space created during the process

    Returns
    -------
    list of dicts
        each dictionary in the list represents one orbit
    ClusterSpace object (optional)
        cluster space created during the process
    """
    assert isinstance(atoms, Atoms), \
        'input configuration must be an ASE Atoms object'

    # create dummy species and cutoffs
    chemical_symbols = ['H', 'He']
    cutoffs = [0.0]

    cs = ClusterSpace(atoms, cutoffs, chemical_symbols)

    singlet_data = []

    for i in range(1, len(cs)):
        cluster_space_info = cs.get_cluster_space_info(i)
        orbit_index = cluster_space_info[0]
        cluster = cs.get_orbit(orbit_index).get_representative_cluster()
        multiplicity = len(cs.get_orbit(orbit_index).get_equivalent_sites())
        assert len(cluster) == 1, \
            'Cluster space contains higher-order terms (beyond singlets)'

        singlet = {}
        singlet['orbit_index'] = orbit_index
        singlet['sites'] = cs.get_orbit(orbit_index).get_equivalent_sites()
        singlet['multiplicity'] = multiplicity
        singlet['representative_site'] = cs.get_orbit(
            orbit_index).get_representative_sites()
        singlet_data.append(singlet)

    if return_cluster_space:
        return singlet_data, cs
    else:
        return singlet_data


def get_singlet_configuration(atoms: Atoms,
                              to_primitive: bool = False) -> Atoms:
    """
    Returns the atomic configuration decorated with a different species for
    each Wyckoff site. This is useful for visualization and analysis.

    Parameters
    ----------
    atoms
        atomic configuration
    to_primitive
        if True the input structure will be reduced to its primitive unit cell
        before processing

    Returns
    -------
    ASE Atoms object
        structure with singlets highlighted by different chemical species
    """
    from ase.data import chemical_symbols
    assert isinstance(atoms, Atoms), \
        'input configuration must be an ASE Atoms object'
    cluster_data, cluster_space = get_singlet_info(atoms,
                                                   return_cluster_space=True)

    if to_primitive:
        singlet_configuration = cluster_space.primitive_structure
        for singlet in cluster_data:
            for site in singlet['sites']:
                symbol = chemical_symbols[singlet['orbit_index'] + 1]
                atom_index = site[0].index
                singlet_configuration[atom_index].symbol = symbol
    else:
        singlet_configuration = atoms.copy()
        singlet_configuration = add_vacuum_in_non_pbc(singlet_configuration)
        orbit_list_supercell\
            = cluster_space._orbit_list.get_supercell_orbit_list(
                singlet_configuration)
        for singlet in cluster_data:
            for site in singlet['sites']:
                symbol = chemical_symbols[singlet['orbit_index'] + 1]
                sites = orbit_list_supercell.get_orbit(
                    singlet['orbit_index']).get_equivalent_sites()
                for lattice_site in sites:
                    k = lattice_site[0].index
                    singlet_configuration[k].symbol = symbol

    return singlet_configuration


def view_singlets(atoms: Atoms, to_primitive: bool = False):
    """
    Visualize singlets in a structure using the ASE graphical user interface.

    Parameters
    ----------
    atoms
        atomic configuration
    to_primitive
        if True the input structure will be reduced to its primitive unit cell
        before processing
    """
    from ase.visualize import view
    assert isinstance(atoms, Atoms), \
        'input configuration must be an ASE Atoms object'
    singlet_configuration = get_singlet_configuration(
        atoms, to_primitive=to_primitive)
    view(singlet_configuration)
