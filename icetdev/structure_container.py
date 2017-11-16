import logging
from collections import OrderedDict


class StructureContainer(object):

    def __init__(self, clusterspace, atoms_list=None):
        '''
        Initializes a StructureContainer object.

        This class serves as a container for ASE Atoms objects, their fit
        properties and their cluster vectors.

        Attributes:
        -----------
        clusterspace : icet ClusterSpace object
            the cluster space used for evaluating the cluster vectors

        atoms_list : list of ASE-Atoms objects
            input structures
        '''

        self._clusterspace = clusterspace

        # Add atomic structures
        if atoms_list is not None:
            if not isinstance(atoms_list, list):
                raise ValueError('atoms_list must be list or None.')
            self._structure_list = []
            for atoms in atoms_list:
                self.add_structure(atoms)

    def __len__(self):
        return len(self._structure_list)

    def __getitem__(self, index):
        return self._structure_list[index]

    def get_number_of_structures(self):
        '''
        Return the total number of structures in the container.
        '''
        return len(self._structure_list)

    def get_structures(self, user_tag):
        '''
        Return the structure in the form of an ASE Atoms object.

        Parameters
        ----------
        user_tag : str
            tag assigned during structure addition
        '''
        for structure in self._structure_list:
            if structure.user_tag == user_tag:
                return structure.atoms

    def get_clustervector(self, user_tag):
        '''
        Return cluster vector of structure in container.

        Parameters
        ----------
        user_tag : str
            tag assigned during structure addition
        '''
        for structure in self._structure_list:
            if structure.user_tag == user_tag:
                return structure.cvs

    def get_properties(self, key, user_tag):
        '''
        Return a list of avaliable properties

        Parameters
        ----------
        key : str
            name the desired property listed in user keys of structures
        user_tag : str
            custom user tag to identify strcuture
        '''
        for structure in self._structure_list:
            if structure.user_tag == user_tag:
                return structure.properties[key]

    def print_structures(self):
        '''
        Print basic information about each structure in structure list
        '''
        def repr_structure(index, structure):
            fields = OrderedDict([
                ('index',     '{:4}'.format(index)),
                ('user_tag',  '{:24}'.format(structure.user_tag)),
                ('numatoms',  '{:5}'.format(len(structure))),
                ('properties',   '{:7.4f}'.format(structure.properties))])
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
        logging.info('{s:-^{n}}'.format(s=' Structures ', n=n))
        logging.info(repr_structure(-1, dummy))

        for i, structure in enumerate(self._structure_list):
            logging.info(repr_structure(i, structure))

    def add_structure(self, atoms, user_tag=None,
                      compute_cvs=True,
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
            if True clustervector is computed
        '''

        atoms_copy = atoms.copy()
        atoms_copy.keys = atoms.keys
        structure = FitStructure(atoms_copy, user_tag=atoms.tag)

        if compute_cvs:
            cvs = self.clusterspace.get_clustervector(structure.atoms)
            structure.set_cvs(cvs)

        self._structure_list.append(structure)

    @property
    def clusterspace(self):
        '''icet clusterspace object'''
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
        clustervector
    '''

    def __init__(self, atoms, user_tag=None, cvs=None):
        assert isinstance(user_tag, str) or user_tag is None, \
            'user_tag must be a string'
        self._atoms = atoms
        self._user_tag = user_tag
        self.set_cvs(cvs)

    @property
    def cvs(self):
        '''numpy array : the fit cvs'''
        return self._cvs

    @property
    def atoms(self):
        '''ASE Atoms object : supercell structure'''
        return self._atoms

    @property
    def properties(self):
        '''numpy array : property'''
        return self._atoms.keys

    @property
    def user_tag(self):
        return str(self._user_tag)

    def set_cvs(self, cvs):
        '''Set the clustervectors to structure.

        Parameters
        ----------
        cvs : list of float
            clustervector for this structure
        '''
        if cvs is not None:
            self._cvs = cvs
        else:
            self._cvs = None

    def __len__(self):
        return len(self._atoms)
