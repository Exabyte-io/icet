import tarfile
import tempfile
import os

import numpy as np
import ase.db
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from icet import ClusterSpace
from typing import BinaryIO, Dict, List, TextIO, Tuple, Union
from numpy import array as Array


class StructureContainer:

    def __init__(self, cluster_space,
                 list_of_atoms: List[Tuple[Atoms, str]]=None,
                 list_of_properties: List[Dict[str, float]]=None):
        """Initializes a StructureContainer object

        This class serves as a container for structure objects, their fit
        properties and their cluster vectors.

        Attributes:
        -----------
        cluster_space : :class:`ClusterSpace`
            the cluster space used for evaluating the cluster vectors

        list_of_atoms : list / list of tuples (bi-optional)
            list of atoms or list of tuples with atoms and user tag

        list_of_properties : list of dicts
            list of properties, which are provided in dicts

        """

        self._cluster_space = cluster_space
        self._structure_list = []

        # Add atomic structures
        if list_of_atoms is not None:
            if not isinstance(list_of_atoms, list):
                raise TypeError('atoms must be given as a list') 

            if list_of_properties is not None:
                if not len(list_of_properties) == len(list_of_atoms):
                    raise ValueError('list of atoms and list of properties'
                                     ' must have the same length')
            else:
                list_of_properties = [None] * len(list_of_atoms)
 
            if not all(isinstance(x, tuple) for x in list_of_atoms):
                list_of_atoms = [(atoms, None) for atoms in list_of_atoms]

            for (atoms, user_tag), properties in zip(list_of_atoms,
                                                     list_of_properties):
                try:
                    self.add_structure(atoms=atoms, user_tag=user_tag,
                                       properties=properties)
                except AssertionError:
                    raise

    def __len__(self) -> int:
        return len(self._structure_list)

    def __getitem__(self, ind: int) -> Atoms:
        return self._structure_list[ind]

    def get_structure_indices(self, user_tag: str=None) -> List[int]:
        """
        Get structure indices via user_tag

        Parameters
        ----------
        user_tag
            user_tag used for selecting structures

        Returns
        -------
        list of integers
            List of structure's indices
        """
        return [i for i, s in enumerate(self)
                if user_tag is None or s.user_tag == user_tag]

    def _get_string_representation(self, print_threshold: int=None,
                                   print_minimum: int=10) -> str:
        """
        String representation of the structure container that provides an
        overview of the structures in the container.

        Parameters
        ----------
        print_threshold
            if the number of structures exceeds this number print dots
        print_minimum
            number of lines printed from the top and the bottom of the
            structure list if `print_threshold` is exceeded

        Returns
        -------
        multi-line string
            string representation of the structure container
        """

        def repr_structure(structure, index=-1, header=False):
            from collections import OrderedDict
            fields = OrderedDict([
                ('index',     '{:4}'.format(index)),
                ('user_tag',  '{:21}'.format(structure.user_tag)),
                ('natoms',    '{:5}'.format(len(structure))),
                ('chemical formula', structure._atoms.get_chemical_formula())])
            fields.update(sorted(structure.properties.items()))
            for key, value in fields.items():
                if isinstance(value, float):
                    fields[key] = '{:8.3f}'.format(value)
                if isinstance(value, int):
                    fields[key] = '{:8}'.format(value)
            s = []
            for name, value in fields.items():
                n = max(len(name), len(value))
                if header:
                    s += ['{s:^{n}}'.format(s=name, n=n)]
                else:
                    if name == 'user_tag' or name == 'chemical formula':
                        # We want them aligned to the left
                        value = '{:{padding}}'.format(value, padding=n - 1)
                    s += ['{s:^{n}}'.format(s=value, n=n)]
            return ' | '.join(s)

        if len(self) == 0:
            return 'Empty StructureContainer'

        # basic information
        # (use last structure in list to obtain maximum line length)
        dummy = self._structure_list[-1]
        width = len(repr_structure(dummy))

        # table header
        s = []
        s += ['{s:=^{n}}'.format(s=' Structure Container ', n=width)]
        s += ['Total number of structures: {}'.format(len(self))]
        s += [''.center(width, '-')]
        s += [repr_structure(dummy, header=True).rstrip()]
        s += [''.center(width, '-')]

        # table body
        index = 0
        while index < len(self):
            if (print_threshold is not None and
                    len(self) > print_threshold and
                    index >= print_minimum and
                    index <= len(self) - print_minimum):
                index = len(self) - print_minimum
                s += [' ...']
            s += [repr_structure(self._structure_list[index], index=index)]
            index += 1
        s += [''.center(width, '=')]

        return '\n'.join(s)

    def __repr__(self) -> str:
        """ String representation. """
        return self._get_string_representation(print_threshold=50)

    def print_overview(self, print_threshold: int=None, print_minimum: int=10):
        """
        Print a list of structures in the structure container.

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

    def add_structure(self, atoms: Atoms, user_tag: str=None,
                      properties: Dict[str, float]=None,
                      allow_duplicate: bool=True):
        """
        Add a structure to the structure list.

        Parameters
        ----------
        atoms
            the atomic structure to be added
        user_tag
            custom user tag to label structure
        properties
            scalar properties. If properties are not specified the atoms
            object are required to have an attached ASE calculator object
            with a calculated potential energy
        allow_duplicate
             whether or not to add the structure if there already exists a
             structure with identical cluster-vector
        """
        assert isinstance(atoms, Atoms), 'atoms has not ASE Atoms format'

        if user_tag is not None:
            assert isinstance(user_tag, str), 'user_tag has wrong type (str)'

        atoms_copy = atoms.copy()
        if properties is None:
            properties = {}
            # check if there is a calculator
            if atoms.calc:
                if len(atoms.calc.check_state(atoms)) == 0:
                    try:
                        energy = atoms.get_potential_energy()
                    except PropertyNotImplementedError:
                        pass
                    else:
                        properties['energy'] = energy / len(atoms)

        cv = self._cluster_space.get_cluster_vector(atoms_copy)
        if not allow_duplicate:
            for i, fs in enumerate(self):
                if np.allclose(cv, fs.cluster_vector):
                    raise ValueError('Atoms have identical cluster vector with'
                                     ' structure {}'.format(i))

        structure = FitStructure(atoms_copy, user_tag)
        structure.set_properties(properties)
        structure.set_cluster_vector(cv)
        self._structure_list.append(structure)

    def get_fit_data(self, structure_indices: List[int]=None,
                     key: str='energy') -> Tuple[Array, Array]:
        """
        Return fit data for all structures. The cluster vectors and
        target properties for all structures are stacked into NumPy arrays.

        Parameters
        ----------
        structure_indices
            list of structure indices. By default (``None``) the
            method will return all fit data available.

        key : str
            key of properties dictionary

        Returns
        -------
        NumPy array, NumPy array
            cluster vectors and target properties for desired structures
        """
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

    def add_properties(self, structure_indices: List[int]=None,
                       properties: List[Dict[str, float]]=None):
        """
        This method allows you to add properties and/or modify
        the values of existing properties

        Parameters
        ----------
        structure_indices
            list of structure indices. By default (``None``) the
            method will add the properties to all structures.

        properties
            list of scalar properties
        """
        if structure_indices is None:
            msg = 'len of properties does not equal len of fit structures'
            assert len(properties) == len(self), msg
            for s, prop in zip(self._structure_list, properties):
                s.set_properties(prop)
        else:
            for i, prop in zip(structure_indices, properties):
                self._structure_list[i].set_properties(prop)

    def get_properties(self, structure_indices: List[int]=None,
                       key: str='energy') -> List[float]:
        """
        Return a list with the value of properties with key='key'
        for a desired set of structures

        Parameters
        ----------
        structures_indices
            list of structure indices. Default to
            None and in that case returns properties of all structures

        key
            key of properties dictionary. Default to 'energy'
        """
        if structure_indices is None:
            prop_list = [s.properties[key]
                         for s in self._structure_list]
        else:
            prop_list = []
            for i in structure_indices:
                prop_list.append(self._structure_list[i].properties[key])

        return prop_list

    def get_structures(self, structure_indices: List[int]=None) -> List[Atoms]:
        """
        Return a list of structures in the form of ASE Atoms

        Parameters
        ----------
        structure_indices: list of integers
            list of structure indices. By default (``None``) the
            method will return all structures listed in the container
        """
        if structure_indices is None:
            s_list = [s.atoms for s in self._structure_list]
        else:
            s_list = []
            for i in structure_indices:
                s_list.append(self._structure_list[i].atoms)

        return s_list

    def get_user_tags(self, structure_indices: List[int]=None) -> List[str]:
        """
        Return a list of user tags for the structures in the structure
        container

        Parameters
        ----------
        structure_indices: list of integers
            list of structure indices. By default (``None``) the
            method will return all user tags listed in the container
        """
        if structure_indices is None:
            tag_list = [s.user_tag for s in self._structure_list]
        else:
            tag_list = []
            for i in structure_indices:
                tag_list.append(self._structure_list[i].user_tag)

        return tag_list

    @property
    def cluster_space(self) -> ClusterSpace:
        """Returns the cluster space object."""
        return self._cluster_space

    @property
    def fit_structures(self):
        """Returns the fit structures."""
        return self._structure_list

    @property
    def available_properties(self):
        """List of the available properties."""
        return sorted(set([p for fs in self for p in fs.properties.keys()]))

    def write(self, outfile: Union[str, BinaryIO, TextIO]):
        """
        Write structure container to a file.

        Parameters
        ----------
        outfile
            output file name or file object
        """
        # Write cluster space to tempfile
        temp_cs_file = tempfile.NamedTemporaryFile()
        self.cluster_space.write(temp_cs_file.name)

        # Write fit structures as an ASE db in tempfile
        temp_db_file = tempfile.NamedTemporaryFile()
        db = ase.db.connect(temp_db_file.name, type='db', append=False)

        for fit_structure in self.fit_structures:
            data_dict = {'user_tag': fit_structure.user_tag,
                         'properties': fit_structure.properties,
                         'cluster_vector': fit_structure.cluster_vector}
            db.write(fit_structure.atoms, data=data_dict)

        with tarfile.open(outfile, mode='w') as handle:
            handle.add(temp_db_file.name, arcname='database')
            handle.add(temp_cs_file.name,
                       arcname='cluster_space')

    @staticmethod
    def read(infile: Union[str, BinaryIO, TextIO]):
        """
        Reads StructureContainer object from file.

        Parameters
        ----------
        infile
            file from which to read

        Raises
        ------
        FileNotFoundError
            if file is not found (str)
        """
        if isinstance(infile, str):
            filename = infile
            if not os.path.isfile(filename):
                raise FileNotFoundError
        else:
            filename = infile.name

        temp_db_file = tempfile.NamedTemporaryFile()
        with tarfile.open(mode='r', name=filename) as tar_file:
            cs_file = tar_file.extractfile('cluster_space')
            temp_db_file.write(tar_file.extractfile('database').read())
            temp_db_file.seek(0)
            cluster_space = ClusterSpace.read(cs_file)
            database = ase.db.connect(temp_db_file.name, type='db')

            structure_container = StructureContainer(cluster_space)
            fit_structures = []
            for row in database.select():
                data = row.data
                fit_structure = FitStructure(row.toatoms(),
                                             user_tag=data['user_tag'],
                                             cv=data['cluster_vector'],
                                             properties=data['properties'])
                fit_structures.append(fit_structure)
            structure_container._structure_list = fit_structures
        return structure_container


class FitStructure:
    """
    This class holds a supercell along with its properties and cluster
    vector.

    Attributes
    ----------
    atoms : :class:`ase:Atoms`
        supercell structure
    user_tag : str
        custom user tag
    cvs : NumPy array
        calculated cluster vector for actual structure
    properties : dict
        the properties dictionary

    """

    def __init__(self, atoms, user_tag, cv=None, properties=None):
        self._atoms = atoms
        self._user_tag = user_tag
        self._properties = {}
        self.set_cluster_vector(cv)
        self.set_properties(properties)

    @property
    def cluster_vector(self) -> Array:
        """NumPy array : the cluster vector"""
        return self._cluster_vector

    @property
    def atoms(self) -> Atoms:
        """ASE Atoms object : supercell structure"""
        return self._atoms

    @property
    def user_tag(self) -> str:
        """string : structure label"""
        return str(self._user_tag)

    @property
    def properties(self) -> Dict[str, float]:
        """dict : properties"""
        return self._properties

    def __getattr__(self, key) -> float:
        """Accesses properties if possible and returns value"""
        if key not in self.properties.keys():
            return super().__getattribute__(key)
        return self.properties[key]

    def set_cluster_vector(self, cv: Array):
        """
        Set the cluster vectors of the structure.

        ***Expert function: Use with caution***

        Parameters
        ----------
        cv
            cluster vector
        """
        if cv is not None:
            self._cluster_vector = cv
        else:
            self._cluster_vector = None

    def set_properties(self, properties: Dict[str, float]):
        """
        Set the properties to structure.

        Parameters
        ----------
        properties
            the properties dictionary

        """
        if properties is not None:
            self._properties.update(properties)

    def __len__(self) -> int:
        return len(self._atoms)
