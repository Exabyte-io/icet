from collections import OrderedDict

import numpy as np
from _icet import ClusterSpace as _ClusterSpace
from ase import Atoms
from icet.tools.geometry import get_primitive_structure

from icet.tools.geometry import add_vacuum_in_non_pbc
from icet.core.orbit_list import create_orbit_list
from icet.core.structure import Structure

import pickle


class ClusterSpace(_ClusterSpace):
    '''
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
    verbosity : int
        verbosity level
    '''

    def __init__(self, atoms, cutoffs, chemical_symbols,
                 Mi=None, verbosity=0):

        # deal with different types of structure objects
        if isinstance(atoms, Atoms):
            self._structure = Structure.from_atoms(atoms)
            self._input_atoms = atoms
        elif isinstance(atoms, Structure):
            self._structure = atoms
            self._input_atoms = atoms.to_atoms()
        else:
            msg = 'Unknown structure format'
            msg += ' {} (ClusterSpace)'.format(type(atoms))
            raise Exception(msg)

        self._cutoffs = cutoffs
        self._chemical_symbols = chemical_symbols
        self._mi = Mi
        self._verbosity = verbosity
        # set up orbit list
        orbit_list = create_orbit_list(self._structure, self._cutoffs,
                                       verbosity=verbosity)
        orbit_list.sort()
        if Mi is None:
            Mi = len(chemical_symbols)
        if isinstance(Mi, dict):
            Mi = self._get_Mi_from_dict(Mi,
                                        orbit_list.get_primitive_structure())
        if not isinstance(Mi, list):
            if isinstance(Mi, int):
                Mi = [Mi] * len(orbit_list.get_primitive_structure())
            else:
                raise Exception('Mi has wrong type (ClusterSpace)')
        msg = ['len(Mi) does not equal len(primitive_structure);']
        msg += ['{} != {}'.format(len(Mi),
                                  len(orbit_list.get_primitive_structure()))]
        msg = ' '.join(msg)
        assert len(Mi) == len(orbit_list.get_primitive_structure()), msg

        # call (base) C++ constructor
        _ClusterSpace.__init__(self, Mi, chemical_symbols, orbit_list)

    @staticmethod
    def _get_Mi_from_dict(Mi, structure):
        '''
        Mi maps the orbit index to the number of allowed components. This
        function maps a dictionary onto the list format that is used
        internatlly for representing Mi.

        Parameters
        ----------
        Mi : dictionary
            each site in the structure should be represented by one entry in
            this dictionary, where the key is the site index and the value is
            the number of components that are allowed on the repsective site
        atoms : ASE Atoms object / icet Structure object (bi-optional)
            atomic configuration

        Returns
        -------
        list
            number of species that are allowed on each site

        Todo
        ----
        * rename function
        '''
        cluster_data = get_singlet_info(structure)
        Mi_ret = [-1] * len(structure)
        for singlet in cluster_data:
            for site in singlet['sites']:
                if singlet['orbit index'] not in Mi:
                    raise Exception('Mi for site {} missing from dictionary'
                                    ''.format(singlet['orbit index']))
                Mi_ret[site[0].index] = Mi[singlet['orbit index']]

        return Mi_ret

    def _get_string_representation(self, print_threshold=None,
                                   print_minimum=10):
        '''
        String representation of the cluster space that provides an overview of
        the orbits (order, radius, multiplicity etc) that constitute the space.

        Parameters
        ----------
        print_threshold : int
            if the number of orbits exceeds this number print dots
        print_minimum : int
            number of lines printed from the top and the bottom of the orbit
            list if `print_threshold` is exceeded

        Returns
        -------
        multi-line string
            string representation of the cluster space.
        '''

        def repr_orbit(orbit, header=False):
            formats = {'order': '{:2}',
                       'size': '{:8.4f}',
                       'multiplicity': '{:4}',
                       'index': '{:4}',
                       'orbit index': '{:4}',
                       'MC vector': '{:}'}
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
        prototype_orbit = self.get_orbit_list_info()[-1]
        width = len(repr_orbit(prototype_orbit))
        s = []
        s += ['{s:=^{n}}'.format(s=' Cluster Space ', n=width)]
        s += [' subelements: {}'.format(' '.join(self.get_atomic_numbers()))]
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
        orbit_list_info = self.get_orbit_list_info()
        while index < len(orbit_list_info):
            if (print_threshold is not None and
                    len(self) > print_threshold and
                    index >= print_minimum and
                    index <= len(self) - print_minimum):
                index = len(self) - print_minimum
                s += [' ...']
            s += [repr_orbit(orbit_list_info[index]).rstrip()]
            index += 1
        s += [''.center(width, '=')]

        return '\n'.join(s)

    def __repr__(self):
        ''' String representation. '''
        return self._get_string_representation(print_threshold=50)

    def print_overview(self, print_threshold=None, print_minimum=10):
        '''
        Print an overview of the cluster space in terms of the orbits (order,
        radius, multiplicity etc).

        Parameters
        ----------
        print_threshold : int
            if the number of orbits exceeds this number print dots
        print_minimum : int
            number of lines printed from the top and the bottom of the orbit
            list if `print_threshold` is exceeded
        '''
        print(self._get_string_representation(print_threshold=print_threshold,
                                              print_minimum=print_minimum))

    def get_orbit_list_info(self):
        '''
        Return list of orbits that provides information concerning their order,
        radius, multiplicity etc).

        Returns
        -------
        list of dictionaries
            information about the orbits that constitute the cluster space.
        '''
        data = []
        zerolet = OrderedDict([('index', 0),
                               ('order', 0),
                               ('size', 0),
                               ('multiplicity', 1),
                               ('orbit index', -1)])

        data.append(zerolet)
        index = 1
        while index < len(self):
            cluster_space_info = self.get_cluster_space_info(index)
            orbit_index = cluster_space_info[0]
            mc_vector = cluster_space_info[1]
            orbit = self.get_orbit(orbit_index)
            local_Mi = self.get_allowed_occupations(
                self._get_primitive_structure(), orbit.representative_sites)
            mc_vectors = orbit.get_mc_vectors(local_Mi)
            mc_permutations = self.get_mc_vector_permutations(
                mc_vectors, orbit_index)
            mc_index = mc_vectors.index(mc_vector)
            mc_permutations_multiplicity = len(mc_permutations[mc_index])
            cluster = self.get_orbit(orbit_index).get_representative_cluster()
            multiplicity = len(self.get_orbit(
                               orbit_index).get_equivalent_sites())
            record = OrderedDict([('index', index),
                                  ('order', cluster.order),
                                  ('size', cluster.geometrical_size),
                                  ('multiplicity', multiplicity *
                                   mc_permutations_multiplicity),
                                  ('orbit index', orbit_index)])
            record['MC vector'] = mc_vector
            data.append(record)
            index += 1
        return data

    def get_number_of_orbits_by_order(self):
        '''
        Return the number of orbits by order.

        Returns
        -------
        dictionary (ordered)
            the key represents the order, the value represents the number of
            orbits
        '''
        count_orbits = {}
        for orbit in self.get_orbit_list_info():
            k = orbit['order']
            count_orbits[k] = count_orbits.get(k, 0) + 1
        return OrderedDict(sorted(count_orbits.items()))

    def get_cluster_vector(self, atoms):
        '''
        Returns the cluster vector for a structure.

        Parameters
        ----------
        atoms : ASE Atoms object / icet Structure object (bi-optional)
            atomic configuration

        Returns
        -------
        NumPy array
            the cluster vector
        '''
        if isinstance(atoms, Atoms):
            structure = Structure.from_atoms(atoms)
        elif isinstance(atoms, Structure):
            structure = atoms
        else:
            msg = 'Unknown structure format'
            msg += ' {} (ClusterSpace.get_cluster_vector)'.format(type(atoms))
            raise Exception(msg)
        # if pbc is not true one needs to massage the structure a bit
        if not np.array(structure.get_pbc()).all():
            atoms = structure.to_atoms()
            add_vacuum_in_non_pbc(atoms)
            structure = Structure.from_atoms(atoms)
        else:
            atoms = structure.to_atoms()
            try:
                atoms = get_primitive_structure(atoms)
            except Exception:
                raise "Failed getting primitive "
                "structure in get_primitive_structure"
            structure = Structure.from_atoms(atoms)
        return _ClusterSpace.get_cluster_vector(self, structure)

    @property
    def structure(self):
        '''
        icet Structure object : structure used for initializing the cluster
        space
        '''
        return self._structure

    @property
    def primitive_structure(self):
        '''
        ASE Atoms obejct : primitive structure on which the cluster space
        is built
        '''
        return self._get_primitive_structure().to_atoms()

    @property
    def cutoffs(self):
        ''' list : cutoffs used for initializing the cluster space '''
        return self._cutoffs

    def write(self, filename):
        """
        Save cluster space to a file.

        Parameters
        ---------
        filename : str
        filename for file
        """

        parameters = {'atoms': self._input_atoms.copy(),
                      'cutoffs': self._cutoffs,
                      'chemical_symbols': self._chemical_symbols,
                      "Mi": self._mi,
                      'verbosity': self._verbosity}
        with open(filename, 'wb') as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def read(filename):
        """
        Read cluster space from filename.

        Parameters
        ---------
        filename : str with filename to saved
        cluster space.
        """
        if isinstance(filename, str):
            with open(filename, 'rb') as handle:
                parameters = pickle.load(handle)
        else:
            parameters = pickle.load(filename)

        return ClusterSpace(parameters['atoms'],
                            parameters['cutoffs'],
                            parameters['chemical_symbols'],
                            parameters['Mi'],
                            parameters['verbosity'])


def get_singlet_info(atoms, return_cluster_space=False):
    '''
    Retrieve information concerning the singlets in the input structure.

    Parameters
    ----------
    atoms : ASE Stoms object / icet Structure object (bi-optional)
        atomic configuration
    return_cluster_space : boolean
        return the cluster space created during the process

    Returns
    -------
    list of dicts
        each dictionary in the list represents one orbit
    ClusterSpace object (optional)
        cluster space created during the process
    '''

    # create dummy elements and cutoffs
    subelements = ['H', 'He']
    cutoffs = [0.0]

    cs = ClusterSpace(atoms, cutoffs, subelements)

    singlet_data = []

    for i in range(1, len(cs)):
        cluster_space_info = cs.get_cluster_space_info(i)
        orbit_index = cluster_space_info[0]
        cluster = cs.get_orbit(orbit_index).get_representative_cluster()
        multiplicity = len(cs.get_orbit(orbit_index).get_equivalent_sites())
        assert len(cluster) == 1, \
            'Cluster space contains higher-order terms (beyond singlets)'

        singlet = {}
        singlet['orbit index'] = orbit_index
        singlet['sites'] = cs.get_orbit(orbit_index).get_equivalent_sites()
        singlet['multiplicity'] = multiplicity
        singlet['representative site'] = cs.get_orbit(
            orbit_index).get_representative_sites()
        singlet_data.append(singlet)

    if return_cluster_space:
        return singlet_data, cs
    else:
        return singlet_data


def get_singlet_configuration(atoms, to_primitive=False):
    '''
    Return atomic configuration decorated with a different element for each
    Wyckoff site. This is useful for visualization and analysis.

    Parameters
    ----------
    atoms : ASE Atoms object / icet Structure object (bi-optional)
        atomic configuration
    to_primitive : boolean
        if True the input structure will be reduced to its primitive unit cell
        before processing

    Returns
    -------
    ASE Atoms object
        structure with singlets highlighted by different elements
    '''

    from ase.data import chemical_symbols
    cluster_data, cluster_space = get_singlet_info(atoms,
                                                   return_cluster_space=True)

    if to_primitive:
        singlet_configuration = cluster_space.primitive_structure
        for singlet in cluster_data:
            for site in singlet['sites']:
                element = chemical_symbols[singlet['orbit index'] + 1]
                atom_index = site[0].index
                singlet_configuration[atom_index].symbol = element
    else:
        singlet_configuration = atoms.copy()
        singlet_configuration = add_vacuum_in_non_pbc(singlet_configuration)
        orbit_list = cluster_space.get_orbit_list()
        orbit_list_supercell \
            = orbit_list.get_supercell_orbit_list(singlet_configuration)
        for singlet in cluster_data:
            for site in singlet['sites']:
                element = chemical_symbols[singlet['orbit index'] + 1]
                sites = orbit_list_supercell.get_orbit(
                    singlet['orbit index']).get_equivalent_sites()
                for lattice_site in sites:
                    k = lattice_site[0].index
                    singlet_configuration[k].symbol = element

    return singlet_configuration


def view_singlets(atoms, to_primitive=False):
    '''
    Visualize singlets in a structure using the ASE graphical user interface.

    Parameters
    ----------
    atoms : ASE Atoms object / icet Structure object (bi-optional)
        atomic configuration
    to_primitive : boolean
        if True the input structure will be reduced to its primitive unit cell
        before processing
    '''
    from ase.visualize import view
    singlet_configuration = get_singlet_configuration(
        atoms, to_primitive=to_primitive)
    view(singlet_configuration)
