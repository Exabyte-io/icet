import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

class StructureContainer(object):
    def __init__(self, clusterspace, 
                 list_of_atoms=None,
                 list_of_properties=None):

        '''
        Initializes a StructureContainer object

        This class serves as a container for ase-atoms objects, their fit
        properties and their cluster vectors.

        Attributes:
        -----------
        clusterspace : icet ClusterSpace object
            the cluster space used for evaluating the cluster vectors

        list_of_atoms : list or list of tuples (bi-optional)
            list of input structures (ASE Atom objects) or list of pairs
            structures-user_tag, e.g. [(atom, user_tag)], where user_tag 
            should be a type str

        list_of_properties : list of dict
            list of properties. properties should be given as dict
        '''

        self._clusterspace = clusterspace

       # Add atomic structures
        if list_of_atoms is not None:
            assert isinstance(list_of_atoms, (list, tuple)
                             ), 'list_of_atoms must be list or tuple or None'

            if list_of_properties is not None:
                msg = ['len(list_of_properties) does not equal len(list_of_atoms)']
                assert(len(list_of_properties) == len(list_of_atoms)), msg
            else:
                list_of_properties = [None]*len(list_of_atoms)

            # transform list to tuple
            if isinstance(list_of_atoms, list):
                list_of_atoms = [(atoms, None) for atoms in list_of_atoms]

            self._structure_list = []
            for (atoms, user_tag), properties in zip(list_of_atoms, list_of_properties):
                try:
                    self.add_structure(atoms=atoms, user_tag=user_tag, properties=properties)
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
        return [i for i, s in enumerate(self) if user_tag is None or s.user_tag == user_tag]

    def __repr__(self):
        '''
        Print basic information about each structure in structure list

        '''
        def repr_structure(index, structure):
            from collections import OrderedDict
            fields = OrderedDict([
                ('index',     '{:4}'.format(index)),
                ('user_tag',  '{:12}'.format(structure.user_tag)),
                ('natoms',    '{:5}'.format(len(structure)))])
            fields.update(sorted(structure.properties.items()))
            for key, value in fields.items():
                if isinstance(value, float):
                    fields[key] = '{:.3f}'.format(value)
            s = []
            for name, value in fields.items():
                n = max(len(name), len(value))
                if index < 0:
                    s += ['{s:^{n}}'.format(s=name, n=n)]
                else:
                    s += ['{s:^{n}}'.format(s=value, n=n)]
            return ' | '.join(s)

        dummy = self._structure_list[0]
        n = len(repr_structure(-1, dummy))
        horizontal_line = '{s:-^{n}}'.format(s='', n=n)

        s = []
        s += ['{s:-^{n}}'.format(s=' Structure Container ', n=n)]
        s += ['Total number of structures: {}'.format(len(self))]
        s += [repr_structure(-1, dummy)]
        s += [horizontal_line]
        # table body
        index = 0
        print_threshold = 24
        while index < len(self):
            if (len(self) > print_threshold
                    and index > 10 and index < len(self) - 10):
                index = len(self) - 10
                s += [' ...']

            s += [repr_structure(index, self._structure_list[index])]
            index += 1

        return '\n'.join(s)


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
            assert atoms.calc is not None, 'Neither property or calculator could be found'
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

        cv = self._clusterspace.get_clustervector(atoms_copy)
        structure.set_clustervector(cv)

        self._structure_list.append(structure)

    def get_fit_data(self, structure_indices=None, key='energy'):
        '''
        Return fit data for all structures. The cluster vectors and
        target properties for all structures are stacked into numpy arrays.

        Parameters
        ----------
        structure_indices : list of integers
            list of structure indices. By default (``None``) the
            method will return all fit data available.

        key : str
            key of properties dictionary

        Returns
        -------
        numpy array, numpy array
            cluster vectors and target properties for desired structures

        '''
        if structure_indices is None:
            cv_list = [s.clustervector
                       for s in self._structure_list]
            prop_list = [s.properties[key]
                         for s in self._structure_list]
        else:
            cv_list, prop_list = [], []
            for i in structure_indices:
                cv_list.append(self._structure_list[i].clustervector)
                prop_list.append(self._structure_list[i].properties[key])

        if len(cv_list) == 0:
            raise Exception('No available fit data for {}'.format(structure_indices))

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
        return self._clusterspace



class FitStructure:
    '''
    This class holds a supercell along its property as well as
    the fit clustervector.

    Attributes
    ----------
    atoms : ASE Atoms object
        supercell structure
    user_tag : str
        custom user tag
    cvs : list of floats
        calculated clustervector for actual structure
    properties : dict
        the properties dictionary
    '''

    def __init__(self, atoms, user_tag, cv=None, properties=None):
        self._atoms = atoms
        self._user_tag = user_tag
        self._properties = {}
        self.set_clustervector(cv)
        self.set_properties(properties)

    @property
    def clustervector(self):
        '''numpy array : the fit clustervector'''
        return self._clustervector

    @property
    def atoms(self):
        '''ASE Atoms object : supercell structure'''
        return self._atoms

    @property
    def user_tag(self):
        '''str : structure label'''
        return str(self._user_tag)

    @property
    def properties(self):
        '''dict : properties'''
        return self._properties

    def set_clustervector(self, cv):
        '''
        Set the clustervectors to structure.

        Parameters
        ----------
        cv : list of float
            clustervector

        '''
        if cv is not None:
            self._clustervector = cv
        else:
            self._clustervector = None

    def set_properties(self, properties):
        '''
        Set the properties to structure.

        Parameters
        __________
        properties: dict
            the properties dictionary

        '''
        if properties is not None:
            self._properties.update(properties)

    def __len__(self):
        return len(self._atoms)
