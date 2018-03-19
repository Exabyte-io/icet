from collections import OrderedDict
from icet.core_py.orbit_list import OrbitList
from icet.tools.geometry import add_vacuum_in_non_pbc


class ClusterSpace(object):
    '''
    This class provides functionality for generating and maintaining cluster
    spaces.

    Parameters
    ----------
    atoms : ASE Atoms object
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

    TODO
    ----
    * Add a method to retrieve a cluster vector
    * When orbitlist is complete:
        * Store OrbitList as a property/attribute
        * Add all the print orbit stuff
        * Fix Mi (allowed components on each site)
        * Write all empty methods that currently are
          only implemented with `pass`
    '''

    def __init__(self, atoms, cutoffs, chemical_symbols,
                 Mi=None, verbosity=0):

        self._structure = atoms
        self._cutoffs = cutoffs
        self._chemical_symbols = chemical_symbols

        # set up orbit list
        orbit_list = OrbitList(self._structure, self._cutoffs,
                               verbosity=verbosity)
        orbit_list.sort()
        self._orbit_list = orbit_list
        self._mi = Mi

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
        n = len(repr_orbit(prototype_orbit))
        s = []
        s += ['{s:-^{n}}'.format(s=' Cluster Space ', n=n)]
        s += [' subelements: {}'.format(' '.join(self.get_atomic_numbers()))]
        s += [' cutoffs: {}'.format(' '.join(['{:.4f}'.format(co)
                                              for co in self._cutoffs]))]
        s += [' total number of orbits: {}'.format(len(self))]
        t = ['{}= {}'.format(k, c)
             for k, c in self.get_number_of_orbits_by_order().items()]
        s += [' number of orbits by order: {}'.format('  '.join(t))]

        # table header
        horizontal_line = '{s:-^{n}}'.format(s='', n=n)
        s += [horizontal_line]
        s += [repr_orbit(prototype_orbit, header=True)]
        s += [horizontal_line]

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
        s += [horizontal_line]

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
            cluster = \
                self.orbit_list.orbits[orbit_index].get_representative_cluster(
                )
            multiplicity = len(self.get_orbit(
                               orbit_index).get_equivalent_sites())
            record = OrderedDict([('index', index),
                                  ('order', cluster.order),
                                  ('size', cluster.geometrical_size),
                                  ('multiplicity', multiplicity),
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
        pass

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
        pass

    @property
    def structure(self):
        '''
        icet Structure object : structure used for initializing the cluster
        space
        '''
        return self._structure

    @property
    def cutoffs(self):
        ''' list : cutoffs used for initializing the cluster space '''
        return self._cutoffs

    @property
    def chemical_symbols(self):
        ''' list of sub elements considered in the cluster space '''
        return self._chemical_symbols

    @property
    def orbit_list(self):
        """
        Return orbit list object
        """
        return self._orbit_list

    def get_orbit(self, index):
        """
        Return orbit with index from
        orbit list

        parameters
        ----------
        index : int
        """
        return self.orbit_list.orbits[index]

    def get_cluster_space_info(self, index):
        pass

    def __len__(self):
        pass


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
