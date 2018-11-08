import pickle
from collections import OrderedDict
from typing import List, Union

import numpy as np

from _icet import ClusterSpace as _ClusterSpace
from ase import Atoms
from icet.core.orbit_list import OrbitList
from icet.core.structure import Structure
from icet.tools.geometry import add_vacuum_in_non_pbc


class ClusterSpace(_ClusterSpace):
    """
    This class provides functionality for generating and maintaining cluster
    spaces.

    Parameters
    ----------
    atoms : ASE Atoms object / icet Structure object (bi-optional)
        atomic configuration
    cutoffs : list of floats
        cutoff radii per order that define the cluster space
    chemical_symbols : list of strings
        list of chemical symbols, each of which must map to an element of
        the periodic table
    Mi : list / dictionary / int
        * if a list is provided, it must contain as many elements as there
          are sites and each element represents the number of allowed
          components on the respective site
        * if a dictionary is provided the key represent the site index and
          the value the number of allowed components
        * if a single `int` is provided each site the number of allowed
          components will be set to `Mi` for sites in the structure
    """

    def __init__(self, atoms: Union[Atoms, Structure], cutoffs: List[float],
                 chemical_symbols: List[str],
                 Mi: Union[list, dict, int]=None) -> None:

        assert isinstance(atoms, Atoms), \
            'input configuration must be an ASE Atoms object'

        self._atoms = atoms
        self._cutoffs = cutoffs
        self._chemical_symbols = chemical_symbols
        self._mi = Mi

        # set up orbit list
        self._orbit_list = OrbitList(self._atoms, self._cutoffs)

        # handle occupations
        if Mi is None:
            Mi = len(chemical_symbols)
        if isinstance(Mi, dict):
            Mi = self._get_Mi_from_dict(
                Mi, self._orbit_list.get_primitive_structure())
        if not isinstance(Mi, list):
            if isinstance(Mi, int):
                Mi = [Mi] * len(self._orbit_list.get_primitive_structure())
            else:
                raise Exception('Mi has wrong type (ClusterSpace)')
        assert len(Mi) == len(self._orbit_list.get_primitive_structure()), \
            'len(Mi) does not equal the number of sites' \
            ' in the primitive structure'

        self._orbit_list.remove_inactive_orbits(Mi)

        # call (base) C++ constructor
        _ClusterSpace.__init__(self, Mi, chemical_symbols, self._orbit_list)

    @staticmethod
    def _get_Mi_from_dict(Mi: dict, atoms: Union[Atoms, Structure]):
        """
        Mi maps the orbit index to the number of allowed components. This
        function maps a dictionary onto the list format that is used
        internatlly for representing Mi.

        Parameters
        ----------
        Mi
            each site in the structure should be represented by one entry in
            this dictionary, where the key is the site index and the value is
            the number of components that are allowed on the repsective site
        atoms
            atomic configuration

        Returns
        -------
        list
            number of species that are allowed on each site

        Todo
        ----
        * rename function
        * remove bi-optionality between icet Structure and ASE Atoms input
        """
        assert isinstance(atoms, (Atoms, Structure)), \
            'input configuration must be an ASE Atoms/icet Structure object'
        if isinstance(atoms, Atoms):
            cluster_data = get_singlet_info(atoms)
        else:
            cluster_data = get_singlet_info(atoms.to_atoms())
        Mi_ret = [-1] * len(atoms)
        for singlet in cluster_data:
            for site in singlet['sites']:
                if singlet['orbit_index'] not in Mi:
                    raise Exception('Mi for site {} missing from dictionary'
                                    ''.format(singlet['orbit_index']))
                Mi_ret[site[0].index] = Mi[singlet['orbit_index']]

        return Mi_ret

    def _get_string_representation(self, print_threshold: int=None,
                                   print_minimum: int=10) -> str:
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
              .format(' '.join(self.get_chemical_symbols()))]
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

    def print_overview(self, print_threshold: int=None, print_minimum: int=10):
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

    @property
    def primitive_structure(self) -> Atoms:
        """
        Primitive structure on which the cluster space is based
        """
        return self._get_primitive_structure().to_atoms()

    @property
    def chemical_symbols(self) -> List[str]:
        """
        Chemical species considered
        """
        return self._chemical_symbols.copy()

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
                      'chemical_symbols': self._chemical_symbols,
                      'Mi': self._mi}
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
                            parameters['chemical_symbols'],
                            parameters['Mi'])


def get_singlet_info(atoms: Atoms, return_cluster_space: bool=False):
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


def get_singlet_configuration(atoms: Atoms, to_primitive: bool=False) -> Atoms:
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
        orbit_list_supercell \
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


def view_singlets(atoms: Atoms, to_primitive: bool=False):
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
