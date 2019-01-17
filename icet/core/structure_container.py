import tarfile
import tempfile

from typing import BinaryIO, Dict, List, TextIO, Tuple, Union

import numpy as np
import ase.db
from ase import Atoms

from icet import ClusterSpace
from icet.io.logging import logger
logger = logger.getChild('structure_container')


class StructureContainer:
    """
    This class serves as a container for structure objects as well as their fit
    properties and cluster vectors.

    Parameters
    ----------
    cluster_space : icet.ClusterSpace
        cluster space used for evaluating the cluster vectors

    list_of_atoms : list or tuple or list(tuple)
        list of atoms; if the list contains tuples, the second element of the
        tuple will be used as a tag of the structure

    list_of_properties : list(dict)
        list of properties, which are provided in dicts
    """

    def __init__(self, cluster_space: ClusterSpace,
                 list_of_atoms: Union[list, tuple] = None,
                 list_of_properties: List[dict] = None,
                 allow_duplicate: bool = True):

        self._cluster_space = cluster_space
        self._structure_list = []

        # Add atomic structures
        if list_of_atoms is not None:
            if not isinstance(list_of_atoms, (list, tuple)):
                raise TypeError('atoms must be given as a list or a tuple')

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
                                       properties=properties,
                                       allow_duplicate=allow_duplicate)
                except (TypeError, ValueError):
                    logger.exception('Skipping structure {}; '.format(
                        list_of_atoms.index((atoms, user_tag))))

    def __len__(self) -> int:
        return len(self._structure_list)

    def __getitem__(self, ind: int):
        return self._structure_list[ind]

    def get_structure_indices(self, user_tag: str = None) -> List[int]:
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

    def _get_string_representation(self, print_threshold: int = None,
                                   print_minimum: int = 10) -> str:
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
        s = []  # type: List
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

    def print_overview(self, print_threshold: int = None,
                       print_minimum: int = 10):
        """
        Prints a list of structures in the structure container.

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

    def add_structure(self, atoms: Atoms, user_tag: str = None,
                      properties: dict = None,
                      allow_duplicate: bool = True):
        """
        Adds a structure to the structure container.

        Parameters
        ----------
        atoms
            the atomic structure to be added
        user_tag
            custom user tag to label structure
        properties
            scalar properties. If properties are not specified the atoms
            object will be checked for an attached ASE calculator object
            with a calculated potential energy
        allow_duplicate
             whether or not to add the structure if there already exists a
             structure with identical cluster-vector
        """
        # atoms must have a proper format and label
        if not isinstance(atoms, Atoms):
            raise TypeError('atoms must be an ASE Atoms object.'
                            ' Not {}'.format(type(atoms)))

        if user_tag is not None:
            if not isinstance(user_tag, str):
                raise TypeError('user_tag must be a string.'
                                ' Not {}.'.format(type(user_tag)))

        atoms_copy = atoms.copy()

        # check for properties in attached calculator
        if properties is None and atoms.calc:
            properties = {}
            if not atoms.calc.calculation_required(atoms, ['energy']):
                energy = atoms.get_potential_energy()
                properties['energy'] = energy / len(atoms)

        # check if there exists structures with identical cluster vector
        cv = self._cluster_space.get_cluster_vector(atoms_copy)
        if not allow_duplicate:
            for i, fs in enumerate(self):
                if np.allclose(cv, fs.cluster_vector):
                    msg = "{} have identical cluster vector with {}".format(
                        user_tag if user_tag is not None else 'Input atoms',
                        fs.user_tag if fs.user_tag != 'None' else 'structure')
                    msg += " at index {}".format(i)
                    raise ValueError(msg)

        structure = FitStructure(atoms_copy, user_tag)
        structure.set_properties(properties)
        structure._set_cluster_vector(cv)
        self._structure_list.append(structure)

    def get_condition_number(self, structure_indices: List[int] = None,
                             key: str = 'energy') -> float:
        """ Returns the condition number for the sensing matrix.

        A very large condition number can be a sign of multicollinearity,
        read more here https://en.wikipedia.org/wiki/Condition_number

        Parameters
        ----------
        structure_indices
            list of structure indices. By default (``None``) the
            method will return all fit data available.
        key
            key of properties dictionary

        Returns
        -------
        condition number for the sensing matrix
        """
        return np.linalg.cond(self.get_fit_data(structure_indices, key)[0])

    def get_fit_data(self, structure_indices: List[int] = None,
                     key: str = 'energy') -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns fit data for all structures. The cluster vectors and
        target properties for all structures are stacked into NumPy arrays.

        Parameters
        ----------
        structure_indices
            list of structure indices. By default (``None``) the
            method will return all fit data available.
        key
            key of properties dictionary

        Returns
        -------
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

    def add_properties(self, structure_indices: List[int] = None,
                       properties: List[dict] = None):
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

    def get_properties(self, structure_indices: List[int] = None,
                       key: str = 'energy') -> List[float]:
        """
        Returns a list with the value of properties with key='key'
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

    def get_structures(self,
                       structure_indices: List[int] = None) -> List[Atoms]:
        """
        Returns a list of structures in the form of ASE Atoms

        Parameters
        ----------
        structure_indices
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

    def get_user_tags(self, structure_indices: List[int] = None) -> List[str]:
        """
        Return a list of user tags for the structures in the structure
        container

        Parameters
        ----------
        structure_indices
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
        """Cluster space used to calculate the cluster vectors."""
        return self._cluster_space

    @property
    def fit_structures(self):
        """List of the fit structures."""
        return self._structure_list

    @property
    def available_properties(self) -> List[str]:
        """List of the available properties."""
        return sorted(set([p for fs in self for p in fs.properties.keys()]))

    def write(self, outfile: Union[str, BinaryIO, TextIO]):
        """
        Writes structure container to a file.

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
        if self.fit_structures:
            db = ase.db.connect(temp_db_file.name, type='db', append=False)

        for fit_structure in self.fit_structures:
            data_dict = {'user_tag': fit_structure.user_tag,
                         'properties': fit_structure.properties,
                         'cluster_vector': fit_structure.cluster_vector}
            db.write(fit_structure.atoms, data=data_dict)

        with tarfile.open(outfile, mode='w') as handle:
            handle.add(temp_db_file.name, arcname='database')
            handle.add(temp_cs_file.name, arcname='cluster_space')

    @staticmethod
    def read(infile: Union[str, BinaryIO, TextIO]):
        """
        Reads StructureContainer object from file.

        Parameters
        ----------
        infile
            file from which to read

        """
        if isinstance(infile, str):
            filename = infile
        else:
            filename = infile.name

        if not tarfile.is_tarfile(filename):
            raise TypeError('{} is not a tar file'.format(filename))

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
    atoms : ASE Atoms
        supercell structure
    user_tag : str
        custom user tag
    cvs : NumPy array
        calculated cluster vector for actual structure
    properties : dict
        the properties dictionary
    """

    def __init__(self, atoms: Atoms, user_tag: str,
                 cv: np.ndarray = None, properties: dict = None):
        self._atoms = atoms
        self._user_tag = user_tag
        self._properties = {}
        self._set_cluster_vector(cv)
        self.set_properties(properties)

    @property
    def cluster_vector(self) -> np.ndarray:
        """calculated cluster vector"""
        return self._cluster_vector

    @property
    def atoms(self) -> Atoms:
        """supercell structure"""
        return self._atoms

    @property
    def user_tag(self) -> str:
        """structure label"""
        return str(self._user_tag)

    @property
    def properties(self) -> Dict[str, Union[float, int]]:
        """scalar properties of structure"""
        return self._properties

    def __getattr__(self, key):
        """Accesses properties if possible and returns value"""
        if key not in self.properties.keys():
            return super().__getattribute__(key)
        return self.properties[key]

    def _set_cluster_vector(self, cv: np.ndarray):
        """ Sets the cluster vectors of the structure. """
        if cv is not None:
            self._cluster_vector = cv
        else:
            self._cluster_vector = None

    def set_properties(self, properties: Dict[str, float]):
        """ Sets properties associated with the structure. """
        if properties is not None:
            self._properties.update(properties)

    def __len__(self) -> int:
        """ Number of sites in the structure. """
        return len(self._atoms)
