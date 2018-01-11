import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError


class StructureContainer(object):

    def __init__(self, cluster_space,
                 list_of_atoms=None,
                 list_of_properties=None):

        '''
        Initializes a StructureContainer object

        This class serves as a container for structure objects, their fit
        properties and their cluster vectors.

        Attributes:
        -----------
        cluster_space : ClusterSpace object
            the cluster space used for evaluating the cluster vectors

        list_of_atoms : list / list of tuples (bi-optional)
            list of structures (ASE Atoms objects) or list of pairs
            of structures and user tag (list of [ASE Atoms object, string]
            tuples)

        list_of_properties : list of dicts
            list of properties, which are provided in dicts
        '''

        self._cluster_space = cluster_space

        # Add atomic structures
        if list_of_atoms is not None:
            assert isinstance(list_of_atoms, (list, tuple)), \
                'list_of_atoms must be list or tuple or None'

            assert len(list_of_atoms) > 0, \
                'list_of_atoms must contain at least one element'

            if list_of_properties is not None:
                msg = 'len(list_of_properties) not equal len(list_of_atoms)'
                assert(len(list_of_properties) == len(list_of_atoms)), msg
            else:
                list_of_properties = [None]*len(list_of_atoms)

            # transform list to tuple
            if isinstance(list_of_atoms[0], Atoms):
                list_of_atoms = [(atoms, None) for atoms in list_of_atoms]

            self._structure_list = []
            for (atoms, user_tag), properties in zip(list_of_atoms,
                                                     list_of_properties):
                try:
                    self.add_structure(atoms=atoms, user_tag=user_tag,
                                       properties=properties)
                except ValueError:
                    print('Skipping list_of_atoms object')

    def __len__(self):
        return len(self._structure_list)

    def __getitem__(self, ind):
        return self._structure_list[ind]

    def get_structure_indices(self, user_tag=None):
        '''
        Get structure indices via user_tag

        Parameters
        ----------
        user_tag : str
            user_tag used for selecting structures

        Returns
        -------
        list of integers
            List of structure's indices
        '''
        return [i for i, s in enumerate(self)
                if user_tag is None or s.user_tag == user_tag]

    def _get_string_representation(self, print_threshold=None,
                                   print_minimum=10):
        '''
        String representation of the structure container that provides an
        overview of the structures in the container.

        Parameters
        ----------
        print_threshold : int
            if the number of structures exceeds this number print dots
        print_minimum : int
            number of lines printed from the top and the bottom of the
            structure list if `print_threshold` is exceeded

        Returns
        -------
        multi-line string
            string representation of the structure container
        '''

        def repr_structure(structure, index=-1, header=False):
            from collections import OrderedDict
            fields = OrderedDict([
                ('index',     '{:4}'.format(index)),
                ('user_tag',  '{:28}'.format(structure.user_tag)),
                ('natoms',    '{:5}'.format(len(structure)))])
            fields.update(sorted(structure.properties.items()))
            for key, value in fields.items():
                if isinstance(value, float):
                    fields[key] = '{:.3f}'.format(value)
            s = []
            for name, value in fields.items():
                n = max(len(name), len(value))
                if header:
                    s += ['{s:^{n}}'.format(s=name, n=n)]
                else:
                    s += ['{s:^{n}}'.format(s=value, n=n)]
            return ' | '.join(s)

        if len(self._structure_list) == 0:
            return 'empty structure container'

        # basic information
        # (use last structure in list to obtain maximum line length)
        dummy = self._structure_list[-1]
        n = len(repr_structure(dummy))
        horizontal_line = '{s:-^{n}}'.format(s='', n=n)

        # table header
        s = []
        s += ['{s:-^{n}}'.format(s=' Structure Container ', n=n)]
        s += ['Total number of structures: {}'.format(len(self))]
        s += [horizontal_line]
        s += [repr_structure(dummy, header=True).rstrip()]
        s += [horizontal_line]

        # table body
        index = 0
        while index < len(self):
            if (print_threshold is not None and
                    len(self) > print_threshold and
                    index >= print_minimum and
                    index <= len(self) - print_minimum):
                index = len(self) - 10
                s += [' ...']
            s += [repr_structure(self._structure_list[index], index=index)]
            index += 1
        s += [horizontal_line]

        return '\n'.join(s)

    def __repr__(self):
        ''' String representation. '''
        return self._get_string_representation(print_threshold=50)

    def print_overview(self, print_threshold=None, print_minimum=10):
        '''
        Print a list of structures in the structure container.

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

    def add_structure(self, atoms, user_tag=None,
                      properties=None):
        '''
        Add a structure to the structure list.

        Parameters
        ----------
        atoms : ASE Atoms object
            the structure to be added
        user_tag : str
            custom user tag to label structure
        properties : dict
            scalar properties. If properties are not specified the atoms
            object are required to have an attached ASE calculator object
            with a calculated potential energy

        '''
        assert isinstance(atoms, Atoms), 'atoms has not ASE Atoms format'

        if user_tag is not None:
            assert isinstance(user_tag, str), 'user_tag has wrong type (str)'

        atoms_copy = atoms.copy()
        if properties is None:
            properties = {}
            assert atoms.calc is not None, \
                'Neither property or calculator could be found'
            msg = 'Not relaxed structure, calculation required'
            assert len(atoms.calc.check_state(atoms)) == 0, msg
            try:
                energy = atoms.get_potential_energy()
            except PropertyNotImplementedError:
                pass
            else:
                properties['energy'] = energy/len(atoms)

        assert properties, 'Calculator does not have energy as a property'

        structure = FitStructure(atoms_copy, user_tag)

        structure.set_properties(properties)

        cv = self._cluster_space.get_cluster_vector(atoms_copy)
        structure.set_cluster_vector(cv)

        self._structure_list.append(structure)

    def get_fit_data(self, structure_indices=None, key='energy'):
        '''
        Return fit data for all structures. The cluster vectors and
        target properties for all structures are stacked into NumPy arrays.

        Parameters
        ----------
        structure_indices : list of integers
            list of structure indices. By default (``None``) the
            method will return all fit data available.

        key : str
            key of properties dictionary

        Returns
        -------
        NumPy array, NumPy array
            cluster vectors and target properties for desired structures

        '''
        if structure_indices is None:
            cv_list = [s.cluster_vector
                       for s in self._structure_list]
            prop_list = [s.properties[key]
                         for s in self._structure_list]
        else:
            cv_list, prop_list = [], []
            for i in structure_indices:
                cv_list.append(self._structure_list[i].cluster_vector)
                prop_list.append(self._structure_list[i].properties[key])

        if len(cv_list) == 0:
            raise Exception('No available fit data for'
                            ' {}'.format(structure_indices))

        return np.array(cv_list), np.array(prop_list)

    def add_properties(self, structure_indices=None, properties=None):
        '''
        This method allows you to add properties and/or modify
        the values of existing properties

        Parameters
        ----------
        structure_indices: list of integers
            list of structure indices. By default (``None``) the
            method will add the properties to all structures.

        properties: list of dict
            list of properties
        '''

        if structure_indices is None:
            msg = 'len of properties does not equal len of fit structures'
            assert len(properties) == len(self), msg
            for s, prop in zip(self._structure_list, properties):
                s.set_properties(prop)
        else:
            for i, prop in zip(structure_indices, properties):
                self._structure_list[i].set_properties(prop)

    def get_properties(self, structure_indices=None, key='energy'):
        '''
        Return a list with the value of properties with key='key'
        for a desired set of structures

        Parameters
        ----------
        structures_indices: list of integers
            list of structure indices. Default to
            None and in that case returns properties of all structures

        key : string
            key of properties dictionary. Default to 'energy'
        '''
        if structure_indices is None:
            prop_list = [s.properties[key]
                         for s in self._structure_list]
        else:
            prop_list = []
            for i in structure_indices:
                prop_list.append(self._structure_list[i].properties[key])

        return prop_list

    def get_structure(self, structure_indices=None):
        '''
        Return a list of structures in the form of ASE Atoms

        Parameters
        ----------
        structure_indices: list of integers
            list of structure indices. By default (``None``) the
            method will return all structures listed in the container

        '''
        if structure_indices is None:
            s_list = [s.atoms for s in self._structure_list]
        else:
            s_list = []
            for i in structure_indices:
                s_list.append(self._structure_list[i].atoms)

        return s_list

    @property
    def cluster_space(self):
        '''
        Returns the icet ClusterSpace object.
        '''
        return self._cluster_space


class FitStructure:
    '''
    This class holds a supercell along with its properties and cluster
    vector.

    Attributes
    ----------
    atoms : ASE Atoms object
        supercell structure
    user_tag : string
        custom user tag
    cvs : list of floats
        calculated cluster vector for actual structure
    properties : dictionary
        the properties dictionary

    '''

    def __init__(self, atoms, user_tag, cv=None, properties=None):
        self._atoms = atoms
        self._user_tag = user_tag
        self._properties = {}
        self.set_cluster_vector(cv)
        self.set_properties(properties)

    @property
    def cluster_vector(self):
        '''NumPy array : the cluster vector'''
        return self._cluster_vector

    @property
    def atoms(self):
        '''ASE Atoms object : supercell structure'''
        return self._atoms

    @property
    def user_tag(self):
        '''string : structure label'''
        return str(self._user_tag)

    @property
    def properties(self):
        '''dict : properties'''
        return self._properties

    def set_cluster_vector(self, cv):
        '''
        Set the cluster vectors of the structure.

        ***Expert function: Use with caution***

        Parameters
        ----------
        cv : list of floats
            cluster vector
        '''
        if cv is not None:
            self._cluster_vector = cv
        else:
            self._cluster_vector = None

    def set_properties(self, properties):
        '''
        Set the properties to structure.

        Parameters
        ----------
        properties : dict
            the properties dictionary

        '''
        if properties is not None:
            self._properties.update(properties)

    def __len__(self):
        return len(self._atoms)
