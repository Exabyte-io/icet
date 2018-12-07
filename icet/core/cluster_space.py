import pickle
from collections import OrderedDict
from typing import List, Union

import numpy as np

from _icet import ClusterSpace as _ClusterSpace
from ase import Atoms
from icet.core.orbit_list import OrbitList
from icet.core.structure import Structure
from icet.tools.geometry import (add_vacuum_in_non_pbc,
                                 get_decorated_primitive_structure)


class ClusterSpace(_ClusterSpace):
    """
    This class provides functionality for generating and maintaining cluster
    spaces.

    Parameters
    ----------
    atoms : ase.Atoms
        atomic configuration
    cutoffs : list(float)
        cutoff radii per order that define the cluster space
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
    """

    def __init__(self, atoms: Atoms, cutoffs: List[float],
                 chemical_symbols: Union[List[str], List[List[str]]]) -> None:

        if not isinstance(atoms, Atoms):
            raise TypeError('Input configuration must be an ASE Atoms object'
                            ', not type {}'.format(type(atoms)))

        self._atoms = atoms.copy()
        self._cutoffs = cutoffs
        self._chemical_symbols = chemical_symbols.copy()
        self._input_chemical_symbols = chemical_symbols.copy()

        # handle occupations
        if all(isinstance(i, str) for i in self._chemical_symbols):
            self._chemical_symbols = [self._chemical_symbols]*len(self._atoms)

        elif not all(isinstance(i, list) for i in self._chemical_symbols):
            raise TypeError("chemical_symbols must be"
                            " List[str] or List[List[str]],"
                            " not {}".format(type(self._chemical_symbols)))

        decorated_primitive, primitive_chemical_symbols = \
            get_decorated_primitive_structure(
                self._atoms, self._chemical_symbols)

        self._primitive_chemical_symbols = primitive_chemical_symbols

        assert len(decorated_primitive) == len(primitive_chemical_symbols)

        # set up orbit list
        self._orbit_list = OrbitList(decorated_primitive, self._cutoffs)
        self._orbit_list.remove_inactive_orbits(primitive_chemical_symbols)

        # call (base) C++ constructor
        _ClusterSpace.__init__(
            self, primitive_chemical_symbols, self._orbit_list)

    def _get_chemical_symbol_representation(self):
        """Returns a str version of the chemical symbols that is
        easier on the eyes.
        """
        nice_str = ''
        if len(self.chemical_symbols) > 4:
            last_symbol = self.chemical_symbols[0]
            count = 1
            for i in range(1, len(self.chemical_symbols)):
                if self.chemical_symbols[i] == last_symbol:
                    count += 1
                    if i == len(self.chemical_symbols)-1:
                        if count == 1:
                            nice_str += '{} '.format(last_symbol)
                        else:
                            nice_str += '{}*{} '.format(count, last_symbol)
                else:
                    if count == 1:
                        nice_str += '{} '.format(last_symbol)
                    else:
                        nice_str += '{}*{} '.format(count, last_symbol)
                    count = 1
                    last_symbol = self.chemical_symbols[i]
        else:
            for s in self.chemical_symbols:
                nice_str += '{} '.format(s)
        return nice_str

    def _get_string_representation(self, print_threshold: int = None,
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
                       'multi_component_vector': '{:}'}
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
                       print_threshold: int = None, print_minimum: int = 10):
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
    def orbit_data(self):
        """
        list of dicts : list of orbits ith information regarding
        order, radius, multiplicity etc
        """
        data = []
        zerolet = OrderedDict([('index', 0),
                               ('order', 0),
                               ('radius', 0),
                               ('multiplicity', 1),
                               ('orbit_index', -1),
                               ('multi_component_vector', '.')])

        data.append(zerolet)
        index = 1
        while index < len(self):
            cluster_space_info = self.get_cluster_space_info(index)
            orbit_index = cluster_space_info[0]
            mc_vector = cluster_space_info[1]
            orbit = self.get_orbit(orbit_index)
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
            data.append(record)
            index += 1
        return data

    def get_number_of_orbits_by_order(self) -> OrderedDict:
        """
        Return the number of orbits by order.

        Returns
        -------
        dictionary (ordered)
            the key represents the order, the value represents the number of
            orbits
        """
        count_orbits = {}  # type: Dict[int, int]
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
        assert isinstance(atoms, Atoms), \
            'input configuration must be an ASE Atoms object'
        if not atoms.pbc.all():
            add_vacuum_in_non_pbc(atoms)
        return _ClusterSpace.get_cluster_vector(self,
                                                Structure.from_atoms(atoms))

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

    @property
    def primitive_structure(self) -> Atoms:
        """
        Primitive structure on which the cluster space is based
        """
        return self._get_primitive_structure().to_atoms()

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

    def write(self, filename: str):
        """
        Saves cluster space to a file.

        Parameters
        ---------
        filename
            name of file to which to write
        """

        parameters = {'atoms': self._atoms.copy(),
                      'cutoffs': self._cutoffs,
                      'chemical_symbols': self._input_chemical_symbols}
        with open(filename, 'wb') as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
            with open(filename, 'rb') as handle:
                parameters = pickle.load(handle)
        else:
            parameters = pickle.load(filename)

        return ClusterSpace(parameters['atoms'],
                            parameters['cutoffs'],
                            parameters['chemical_symbols'])


def get_singlet_info(atoms: Atoms, return_cluster_space: bool = False):
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
