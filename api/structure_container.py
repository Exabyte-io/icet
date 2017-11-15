
class StructureContainer(object):

    def __init__(self, clusterspace, 
                 atoms_list=None, 
                 properties=None):

        '''
        Initializes a structure_container object
    
        This class serves as a container for ase-atoms objects, their fit
        properties and their clustervectors. 

        Attributes:
        -----------
        clusterspace : icet ClusterSpace object
            ClusterSpace object which containes a list of clusters 
            
        atoms_list : list of ASE Atoms objects
            list of structures to be added to structure container

        TODO:
            - Open a discussion about fit_ready and user_tag
            - Pass a copy of clusterspace instead
            - More structured unit test
            - Logging 

        '''

        self._clusterspace = clusterspace #.copy()

        # Add atomic structures
        if atoms_list is not None:
            if not isinstance(atoms_list, list):
                raise ValueError('atoms_list must be list or None.')
            self._structure_list = []
            for atoms, properties in zip(atoms_list, properties):
                self.add_structure(atoms=atoms, properties=properties)

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
        list
            List of structures with corresponding user_tag
        '''
        return [i for i, s in enumerate(self) if user_tag is None or s.user_tag == user_tag]

    def __repr__(self):
        ''' 
        Print basic information about each structure in structure list 
        
        TODO: print properties (with default units) in more fancy way
        '''
        def repr_structure(index, structure):
            from collections import OrderedDict
            fields = OrderedDict([
                ('index',     '{:4}'.format(index)),
                ('user_tag',  '{:12}'.format(structure.user_tag)),
                ('properties', '{:}'.format('   '.join(['{:}:{:9.6f}'.format(k ,v)
                                                       for k, v in structure.properties.items()]))),
                ('numatoms',  '{:5}'.format(len(structure))),
                ('fit_ready', '{:5}'.format(str(structure.fit_ready)))])
            s = []
            for name, value in fields.items():
                n = max(len(name), len(value))
                if index < 0:
                    s += ['{s:^{n}}'.format(s=name, n=n)]
                else:
                    s += ['{s:^{n}}'.format(s=value, n=n)]
            return ' | '.join(s)

        # basic information
        s = ['Structure_container']
        s += [' total size of structure_container: {}'.format(len(self))]

        dummy = self._structure_list[0]
        n = len(repr_structure(-1, dummy))
        horizontal_line = '{s:-^{n}}'.format(s='', n=n)

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

        s += [horizontal_line]

        return '\n'.join(s)           


    def add_structure(self, atoms, user_tag=None,
                      properties =  None,
                      compute_cvs=False,
                      map_to_prototype=False):
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
        import logging
        atoms_copy = atoms.copy()

        if user_tag is None:
            user_tag = atoms_copy.get_chemical_formula(mode='hill')

        structure = FitStructure(atoms_copy, user_tag)

        if properties is not None:
            structure.set_properties(properties)            
        elif atoms.calc is not None:
            structure.set_properties({'energy': atoms.get_potential_energy()/len(atoms)}) 
        else:
            logging.warning('Skipping atoms object, no properties found')
            return

        if compute_cvs:
            cv = self.clusterspace.get_clustervector(atoms_copy)
            structure.set_cvs(cv)

        self._structure_list.append(structure)


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
            p_list = [s.properties[key]
                      for s in self._structure_list] #if fit_ready
        else:
            p_list = []
            for i in structures:
                if not self._structure_list[i].fit_ready:
                    raise ValueError('Structure {} is not fit ready'.format(i))
                p_list.append(self._structure_list[i].properties[key])

        return p_list


    def get_structures(self, structures=None):
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
        return self._clusterspace #.copy()



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

    def __init__(self, atoms, user_tag, cvs=None, properties=None):
        self._atoms = atoms
        self._user_tag = user_tag
        self._fit_ready = False
        self._properties = {}
        self.set_cvs(cvs)
        self.set_properties(properties)

    @property
    def cvs(self):
        '''numpy array : the fit cvs'''
        return self._cvs

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


    def set_cvs(self, cvs):
        '''
        Set the clustervectors to structure.

        Parameters
        ----------
        cvs : list of float 
            clustervector for this structure

        '''
        if cvs is not None:
            self._cvs = cvs
        else:
            self._cvs = None

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
