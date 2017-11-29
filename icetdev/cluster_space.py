from _icetdev import ClusterSpace as _ClusterSpace
from icetdev import Structure
from icetdev.orbit_list import create_orbit_list
from icetdev.permutation_map import vacuum_on_non_pbc
from ase import Atoms
import numpy


class ClusterSpace(_ClusterSpace):

    def __init__(self, atoms, cutoffs, chemical_symbols,
                 Mi=None, verbosity=0):
        '''
        Initializes a clusterspace object.

        Parameters
        ----------
        atoms : ASE atoms object / icet structure object (bi-optional)
            atomic configuration
        cutoffs : list of floats
            cutoff radii per order that define the clusterspace
        chemical_symbols : list of strings
            list of chemical symbols, each of which must map to an element of
            the periodic table
        verbosity : int
            verbosity level

        Todo
        ----
        * document `Mi`
        '''

        # deal with different types of structure objects
        if isinstance(atoms, Atoms):
            structure = Structure.from_atoms(atoms)
        elif isinstance(atoms, Structure):
            structure = atoms
        else:
            msg = 'Unknown structure format'
            msg += ' {} (ClusterSpace)'.format(type(atoms))
            raise Exception(msg)

        # set up orbit list
        orbit_list = create_orbit_list(structure, cutoffs, verbosity=verbosity)
        orbit_list.sort()
        if Mi is None:
            Mi = len(chemical_symbols)
        if isinstance(Mi, dict):
            Mi = get_Mi_from_dict(Mi, orbit_list.get_primitive_structure())
        if not isinstance(Mi, list):
            if not isinstance(Mi, int):
                raise Exception('Mi has wrong type (ClusterSpace)')
            else:
                Mi = [Mi] * len(orbit_list.get_primitive_structure())
        msg = ['len(Mi) does not equal len(primitive_structure);']
        msg += ['{} != {}'.format(len(Mi),
                                  len(orbit_list.get_primitive_structure()))]
        msg = ' '.join(msg)
        assert len(Mi) == len(orbit_list.get_primitive_structure()), msg

        # call (base) C++ constructor
        _ClusterSpace.__init__(self, Mi, chemical_symbols, orbit_list)
        self.cutoffs = cutoffs

    def __repr__(self):
        '''
        String representation of the clusterspace.
        '''

        def repr_cluster(index, cluster, multiplicity=0,
                         orbit_index=0, mc_vector=[0] * 5):
            from collections import OrderedDict
            fields = OrderedDict([
                ('order',   '{:2}'.format(cluster.order)),
                ('radius',  '{:9.4f}'.format(cluster.geometrical_size)),
                ('multiplicity', '{:4}'.format(multiplicity)),
                ('index',     '{:4}'.format(index)),
                ('orbit',     '{:4}'.format(orbit_index)),
                ('MC vector', '{:}'.format(mc_vector))])
            s = []
            for name, value in fields.items():
                n = max(len(name), len(value))
                if index < 0:
                    s += ['{s:^{n}}'.format(s=name, n=n)]
                else:
                    s += ['{s:^{n}}'.format(s=value, n=n)]
            return ' | '.join(s)

        # basic information
        cluster = self.get_orbit(0).get_representative_cluster()
        n = len(repr_cluster(-1, cluster))
        s = []
        s += ['{s:-^{n}}'.format(s=' Cluster Space ', n=n)]
        s += [' subelements: {}'.format(' '.join(self.get_atomic_numbers()))]
        s += [' cutoffs: {}'.format(' '.join(['{}'.format(co)
                                              for co in self.cutoffs]))]
        s += [' number of orbits: {}'.format(len(self))]

        # table header
        horizontal_line = '{s:-^{n}}'.format(s='', n=n)
        s += [horizontal_line]
        s += [repr_cluster(-1, cluster)]
        s += [horizontal_line]

        # table body
        index = 0
        print_threshold = 50
        while index < len(self):
            if (len(self) > print_threshold
                    and index > 10 and index < len(self) - 10):
                index = len(self) - 10
                s += [' ...']

            clusterspace_info = self.get_clusterspace_info(index)
            orbit_index = clusterspace_info[0]
            mc_vector = clusterspace_info[1]
            cluster = self.get_orbit(orbit_index).get_representative_cluster()
            multiplicity = len(self.get_orbit(
                               orbit_index).get_equivalent_sites())
            s += [repr_cluster(index, cluster, multiplicity,
                               orbit_index, mc_vector)]

            index += 1

        s += [horizontal_line]

        return '\n'.join(s)

    def get_clustervector(self, atoms):
        '''
        Returns the cluster vector for a structure.

        Parameters
        ----------
        atoms : ASE atoms object / icet structure object (bi-optional)
            atomic configuration

        Returns
        -------
        NumPy array
        '''
        if isinstance(atoms, Atoms):
            structure = Structure.from_atoms(atoms)
        elif isinstance(atoms, Structure):
            structure = atoms
        else:
            msg = 'Unknown structure format'
            msg += ' {} (ClusterSpace.get_clustervector)'.format(type(atoms))
            raise Exception(msg)

        # if pbc is not true one needs to massage the structure a bit
        if not numpy.array(structure.get_pbc()).all() == True:
            atoms = structure.to_atoms()
            vacuum_on_non_pbc(atoms)
            structure = Structure.from_atoms(atoms)
        else:
            atoms = structure.to_atoms()
            atoms.wrap()
            structure = Structure.from_atoms(atoms)
        return _ClusterSpace.get_clustervector(self, structure)


def get_singlet_info(atoms, return_clusterspace=False):
    '''
    Get information concerning the singlets in this structure.

    Parameters
    ----------
    atoms : ASE atoms object / icet structure object (bi-optional)
        atomic configuration
    return_clusterspace : boolean
        return the clusterspace created in the process

    Returns
    -------
    list of dictionaries or tuple of list of dicts and ClusterSpace object
        each dictionary represents one orbit
    '''

    # create dummy elements and cutoffs
    subelements = ['H', 'He']
    cutoffs = [0.0]

    clusterspace = ClusterSpace(atoms, cutoffs, subelements)

    singlet_data = []

    for i in range(len(clusterspace)):
        clusterspace_info = clusterspace.get_clusterspace_info(i)
        orbit_index = clusterspace_info[0]
        cluster = clusterspace.get_orbit(
            orbit_index).get_representative_cluster()
        multiplicity = len(clusterspace.get_orbit(
            orbit_index).get_equivalent_sites())
        assert len(cluster) == 1, \
            'Cluster in singlet only clusterspace has non-singlets'

        singlet = {}
        singlet['orbit_index'] = orbit_index
        singlet['sites'] = clusterspace.get_orbit(
            orbit_index).get_equivalent_sites()
        singlet['multiplicity'] = multiplicity
        singlet['representative_site'] = clusterspace.get_orbit(
            orbit_index).get_representative_sites()
        singlet_data.append(singlet)

    if return_clusterspace:
        return singlet_data, clusterspace
    else:
        return singlet_data


def get_singlet_configuration(atoms, to_primitive=False):
    '''
    Returns ase atoms object decorate with a element for each wyckoff site

    Parameters
    ----------
    atoms : ASE atoms object / icet structure object (bi-optional)
        atomic configuration
    to_primitive : bool
        use primitive of atoms or not

    Returns
    -------
    atoms_singlet
    '''

    from ase.data import chemical_symbols
    cluster_data, clusterspace = get_singlet_info(atoms,
                                                  return_clusterspace=True)

    if to_primitive:
        singlet_configuration = clusterspace.get_primitive_structure().to_atoms()
        for singlet in cluster_data:
            for site in singlet['sites']:
                element = chemical_symbols[singlet['orbit_index'] + 1]
                atom_index = site[0].index
                singlet_configuration[atom_index].symbol = element
    else:
        singlet_configuration = atoms.copy()
        singlet_configuration = vacuum_on_non_pbc(singlet_configuration)
        orbitlist = clusterspace.get_orbit_list()
        orbitlist_supercell = orbitlist.get_supercell_orbitlist(singlet_configuration)        
        for singlet in cluster_data:
            for site in singlet['sites']:
                element = chemical_symbols[singlet['orbit_index'] + 1]
                sites = orbitlist_supercell.get_orbit(
                    singlet['orbit_index']).get_equivalent_sites()
                for lattice_site in sites:
                    singlet_configuration[lattice_site[0].index].symbol = element

    return singlet_configuration


def view_singlets(atoms, to_primitive=False):
    '''
    Visualize singlets in a structure using the ASE gui.

    Parameters
    ----------
    atoms : ASE atoms object / icet structure object (bi-optional)
        atomic configuration
    to_primitive : bool
        use primitive of atoms or not
    '''

    from ase.visualize import view
    singlet_configuration = get_singlet_configuration(
        atoms, to_primitive=to_primitive)
    view(singlet_configuration)


def get_Mi_from_dict(Mi, structure):
    '''
    Mi maps orbit index to allowed components
    this function will return a list, where
    Mi_ret[i] will be the allowed components on site index i

    Parameters
    ----------
    Mi : xx
        xx
    atoms : ASE atoms object / icet structure object (bi-optional)
        atomic configuration

    Todo
    ----
    * rename function, remove `Mi`
    * complete docstring
    '''
    cluster_data = get_singlet_info(structure)
    Mi_ret = [-1] * len(structure)
    for singlet in cluster_data:
        for site in singlet['sites']:
            Mi_ret[site[0].index] = Mi[singlet['orbit_index']]

    for all_comp in Mi_ret:
        if all_comp == -1:
            raise Exception(
                'Error: the calculated Mi from dict did not cover all sites on input structure. \n Were all sites in primitive mapped?')

    return Mi_ret

