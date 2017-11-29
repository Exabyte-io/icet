import numpy as np

class StructureContainer(object):

    def __init__(self, clusterspace, 
                 list_of_atoms=None):

        '''
        Initializes a StructureContainer object

        This class serves as a container for ase-atoms objects, their fit
        properties and their cluster vectors.

        Attributes:
        -----------
        clusterspace : icet ClusterSpace object
            the cluster space used for evaluating the cluster vectors

        list_of_atoms : list or tuple (bi-optional)
            list of input structures (ASE Atom objects) or tuple
            with input structures, set of properties (dict) and user_tag (string).

        TODO:
        * Overload initialization for use cases where a duple as 
        (structure, properties) or (structure, user_tag) is passed 
        '''

        self._clusterspace = clusterspace

       # Add atomic structures
        if list_of_atoms is not None:
            if not isinstance(list_of_atoms, (list, tuple)):
                raise ValueError('atoms_list must be list or tuple or None.')
            if not isinstance(list_of_atoms, tuple):
                list_of_atoms = [(atoms, None, None) for atoms in list_of_atoms]
            self._structure_list = []
            for atoms, properties, user_tag in list_of_atoms:
                self.add_structure(atoms=atoms,
                                   properties=properties,
                                   user_tag=user_tag)

    def __len__(self):
        return len(self._structure_list)

    def __getitem__(self, ind):
        return self._structure_list[ind]

    def __get_structure_indices(self, user_tag=None):
        '''
        Get structure indices via user_tag

        Parameters
        ----------
        user_tag : str
            user_tag used for selecting structures

        Returns
        -------
        list
            List of indexes of structures with a given user_tag
        '''
        return [i for i, s in enumerate(self) if user_tag is None or s.user_tag == user_tag]

    def get_structure_indices(self, user_tag=None):
        '''
        Returns a lis of indexes of structures with
        a given user_tag
        '''
        return self.__get_structure_indices(user_tag)

    def __repr__(self):
        '''
        Print basic information about each structure in structure list

        '''
        def repr_structure(index, structure):
            from collections import OrderedDict
            fields = OrderedDict([
                ('index',     '{:4}'.format(index)),
                ('user_tag',  '{:12}'.format(structure.user_tag)),
                ('natoms',    '{:5}'.format(len(structure))),
                ('fit_ready', '{:5}'.format(str(structure.fit_ready)))])
            sorted(structure.properties.items())
            fields.update(structure.properties.items())
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
                      properties=None,
                      compute_clustervectors=True):
        '''
        Add a structure to the training set.

        Parameters
        ----------
        atoms : ASE Atoms object
            the structure to be added
        user_tag : str
            custom user tag to identify structure
        compute_clustervector: bool
            if True, clustervector is computed

        '''
        atoms_copy = atoms.copy()

        if user_tag is not None:
            if not isinstance(user_tag, str):
                raise Exception('Skipping atoms object, user_tag has wrong type (str)')
                return

        if properties is None:
            if atoms.calc is not None:
                properties = {'energy': atoms.get_potential_energy()/len(atoms)}
            else:
                raise Exception('Skipping atoms object, no properties found')
                return

        structure = FitStructure(atoms_copy, user_tag)

        structure.set_properties(properties)

        if compute_clustervectors:
            cv = self._clusterspace.get_clustervector(atoms_copy)
            structure.set_clustervector(cv)

        self._structure_list.append(structure)


    def get_fit_data(self, structures=None, key='energy'):
        '''Return fit data for all structures. The cluster vectors and
        target properties for all structures are stacked into numpy arrays.

        Parameters
        ----------
        structures: list, tuple, int
            list of integers corresponding to structure indices. Defaults to
            None and in that case returns all fit data available.

        Returns
        -------
        numpy array, numpy array
            cluster vectors and target properties for all structures

        '''
        if structures is None:
            cv_list = [s.clustervector
                       for s in self._structure_list if s.fit_ready]
            prop_list = [s.properties[key]
                         for s in self._structure_list if s.fit_ready]
        else:
            cv_list, prop_list = [], []
            if isinstance(structures, str):
                structures = self.__get_structure_indices(structures)
            for i in structures:
                if not self._structure_list[i].fit_ready:
                    raise ValueError('Structure {} is not fit ready'.format(i))
                cv_list.append(self._structure_list[i].clustervector)
                prop_list.append(self._structure_list[i].properties[key])

        if len(cv_list) == 0:
            raise Exception('No available fit data for {}'.format(structures))
            return np.array([]), np.array([])
        return np.array(cv_list), np.array(prop_list)

    def load_properties(self, structures=None, properties=None):
        '''
        This method allows you to add properties and/or modify
        the values of existing properties

        Parameters
        ----------
        structures: list, tuple, int
            list of indices of the desired structures

        properties: dict
            properties to add

        '''
        if properties is None:
            raise ValueError('load_properties() requires at least properties in arguments')

        if structures is None:
            for s, prop in zip(self._structure_list, properties):
                s.set_properties(prop)
        else:
            for i, prop in zip(structures, properties):
                if not self._structure_list[i].fit_ready:
                    raise ValueError('Structure {} is not fit ready'.format(i))
                self._structure_list[i].set_properties(prop)


    def get_properties(self, structures=None, key='energy'):
        '''
        Return a list of properties of a list
        of structures.

        Parameters
        ----------
        structures: list, tuple, int
            list of indices of the desired structures

        key : string
            key label of property


        '''
        if structures is None:
            prop_list = [s.properties[key]
                         for s in self._structure_list] #if fit_ready
        else:
            prop_list = []
            if isinstance(structures, str):
                structures = self.__get_structure_indices(structures)
            for i in structures:
                if not self._structure_list[i].fit_ready:
                    raise ValueError('Structure {} is not fit ready'.format(i))
                prop_list.append(self._structure_list[i].properties[key])

        return prop_list


    def get_structure(self, structures=None):
        '''
        Return a list of structures in the form of ASE Atoms

        Parameters
        ----------
        structures: list, tuple, int
            list of indices of the desired structures

        '''
        if structures is None:
            s_list = [s.atoms
                      for s in self._structure_list]
        else:
            s_list = []
            if isinstance(structures, str):
                structures = self.__get_structure_indices(structures)
            for i in structures:
                if not self._structure_list[i].fit_ready:
                    raise ValueError('Structure {} is not fit ready'.format(i))
                s_list.append(self._structure_list[i].atoms)

        return s_list

    @property
    def get_cluster_space(self):
        '''
        Returns a copy of the icet ClusterSpace object. 
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
    '''

    def __init__(self, atoms, user_tag, cv=None, properties=None):
        self._atoms = atoms
        self._user_tag = user_tag
        self._fit_ready = False
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
        return str(self._user_tag)

    @property
    def properties(self):
        '''dict : properties'''
        return self._properties

    @property
    def fit_ready(self):
        '''boolean : True if the structure is prepared for fitting, i.e. the
        clustervector is available'''
        return self._fit_ready


    def set_clustervector(self, cv):
        '''
        Set the clustervectors to structure.

        Parameters
        ----------
        cvs : list of float
            clustervector for this structure

        '''
        if cv is not None:
            self._clustervector = cv
            self._fit_ready = True
        else:
            self._clustervector = None

    def set_properties(self, properties):
        '''
        Set properties to structure.

        Parameters
        __________
        properties: dict (key,value)
            list of properties for this structures

        '''
        if properties is not None:
            self._properties.update(properties)

    def __len__(self):
        return len(self._atoms)
