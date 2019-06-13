"""
This module provides the StructureContainer class.
"""

import tarfile
import tempfile

from typing import BinaryIO, List, TextIO, Tuple, Union

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

    def __init__(self, cluster_space: ClusterSpace):

        if not isinstance(cluster_space, ClusterSpace):
            raise TypeError('cluster_space must be a ClusterSpace object')

        self._cluster_space = cluster_space
        self._structure_list = []

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
        return [i for i, s in enumerate(self) if user_tag is None or s.user_tag == user_tag]

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
            """
            Helper function used to generate a representation string for a
            single structure.
            """
            from collections import OrderedDict
            fields = OrderedDict([
                ('index', '{:4}'.format(index)),
                ('user_tag', '{:21}'.format(structure.user_tag)),
                ('natoms', '{:5}'.format(len(structure))),
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

    def add_structure(self, atoms: Atoms, user_tag: str = None, properties: dict = None,
                      allow_duplicate: bool = True, sanity_check: bool = True):
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
         sanity_check
            whether or not to carry out a sanity check before adding the
            structure. This includes checking occupations and volume.
        """

        # atoms must have a proper format and label
        if not isinstance(atoms, Atoms):
            raise TypeError('atoms must be an ASE Atoms object. Not {}'.format(type(atoms)))

        if user_tag is not None:
            if not isinstance(user_tag, str):
                raise TypeError('user_tag must be a string. Not {}.'.format(type(user_tag)))

        if sanity_check:
            self._cluster_space.assert_structure_compatability(atoms)

        # check for properties in attached calculator
        if properties is None and atoms.calc:
            properties = {}
            if not atoms.calc.calculation_required(atoms, ['energy']):
                energy = atoms.get_potential_energy()
                properties['energy'] = energy / len(atoms)

        # check if there exists structures with identical cluster vector
        atoms_copy = atoms.copy()
        cv = self._cluster_space.get_cluster_vector(atoms_copy)
        if not allow_duplicate:
            for i, fs in enumerate(self):
                if np.allclose(cv, fs.cluster_vector):
                    msg = "{} have identical cluster vector with {}".format(
                        user_tag if user_tag is not None else 'Input atoms',
                        fs.user_tag if fs.user_tag != 'None' else 'structure')
                    msg += " at index {}".format(i)
                    raise ValueError(msg)

        # add structure
        structure = FitStructure(atoms_copy, user_tag, cv, properties)
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
            cv_list = [s.cluster_vector for s in self._structure_list]
            prop_list = [s.properties[key] for s in self._structure_list]
        else:
            cv_list, prop_list = [], []
            for i in structure_indices:
                cv_list.append(self._structure_list[i].cluster_vector)
                prop_list.append(self._structure_list[i].properties[key])

        if cv_list is None:
            raise Exception('No available fit data for {}'
                            .format(structure_indices))

        return np.array(cv_list), np.array(prop_list)

    @property
    def cluster_space(self) -> ClusterSpace:
        """Cluster space used to calculate the cluster vectors."""
        return self._cluster_space

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
        if self._structure_list:
            db = ase.db.connect(temp_db_file.name, type='db', append=False)

        for fit_structure in self._structure_list:
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
                 cv: np.ndarray, properties: dict = {}):
        self._atoms = atoms
        self._user_tag = user_tag
        self._cluster_vector = cv
        self.properties = properties

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

    def __getattr__(self, key):
        """Accesses properties if possible and returns value"""
        if key not in self.properties.keys():
            return super().__getattribute__(key)
        return self.properties[key]

    def __len__(self) -> int:
        """ Number of sites in the structure. """
        return len(self._atoms)
