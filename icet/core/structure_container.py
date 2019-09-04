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

    Example
    -------
    The following snippet illustrates the initialization
    and usage of a StructureContainer object. The construction
    of a structure container is convenient for compiling the
    data needed to train a cluster expansion, i.e., a sensing
    matrix and target energies::

        from ase.build import bulk
        from icet import ClusterSpace, StructureContainer
        from icet.tools import enumerate_structures
        import numpy as np

        # create cluster space
        prim = bulk('Au')
        cs = ClusterSpace(prim, cutoffs=[7.0, 5.0],
                          chemical_symbols=[['Au', 'Pd']])

        sc = StructureContainer(cs)
        for structure in enumerate_structures(prim, range(5), ['Au', 'Pd']):
            sc.add_structure(structure,
                             properties={'my_random_energy': np.random.rand()})
        print(sc)

        sensing_matrix, target_energies = sc.get_fit_data(key='my_random_energy')
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

    def _get_string_representation(self, print_threshold: int = None) -> str:
        """
        String representation of the structure container that provides an
        overview of the structures in the container.

        Parameters
        ----------
        print_threshold
            if the number of structures exceeds this number print dots

        Returns
        -------
        multi-line string
            string representation of the structure container
        """

        if len(self) == 0:
            return 'Empty StructureContainer'

        # Number of structures to print before cutting and printing dots
        if print_threshold is None or print_threshold >= len(self):
            print_threshold = len(self) + 2

        # format specifiers for fields in table
        def get_format(val):
            if isinstance(val, float):
                return '{:9.4f}'
            else:
                return '{}'

        # table headers
        default_headers = ['index', 'user_tag', 'n_atoms', 'chemical formula']
        property_headers = sorted(set(key for fs in self for key in fs.properties))
        headers = default_headers + property_headers

        # collect the table data
        str_table = []
        for i, fs in enumerate(self):
            default_data = [i, fs.user_tag, len(fs), fs.structure.get_chemical_formula()]
            property_data = [fs.properties.get(key, '') for key in property_headers]
            str_row = [get_format(d).format(d) for d in default_data+property_data]
            str_table.append(str_row)
        str_table = np.array(str_table)

        # find maximum widths for each column
        widths = []
        for i in range(str_table.shape[1]):
            data_width = max(len(val) for val in str_table[:, i])
            header_width = len(headers[i])
            widths.append(max([data_width, header_width]))

        total_width = sum(widths) + 3 * len(headers)
        row_format = ' | '.join('{:'+str(width)+'}' for width in widths)

        # Make string representation of table
        s = []
        s += ['{s:=^{n}}'.format(s=' Structure Container ', n=total_width)]
        s += ['Total number of structures: {}'.format(len(self))]
        s += [''.center(total_width, '-')]
        s += [row_format.format(*headers)]
        s += [''.center(total_width, '-')]
        for i, fs_data in enumerate(str_table, start=1):
            s += [row_format.format(*fs_data)]
            if i+1 >= print_threshold:
                s += [' ...']
                s += [row_format.format(*str_table[-1])]
                break
        s += [''.center(total_width, '=')]
        s = '\n'.join(s)

        return s

    def __repr__(self) -> str:
        """ String representation. """
        return self._get_string_representation(print_threshold=50)

    def print_overview(self, print_threshold: int = None):
        """
        Prints a list of structures in the structure container.

        Parameters
        ----------
        print_threshold
            if the number of orbits exceeds this number print dots
        """
        print(self._get_string_representation(print_threshold=print_threshold))

    def add_structure(self, structure: Atoms, user_tag: str = None,
                      properties: dict = None, allow_duplicate: bool = True,
                      sanity_check: bool = True):
        """
        Adds a structure to the structure container.

        Parameters
        ----------
        structure
            the atomic structure to be added
        user_tag
            custom user tag to label structure
        properties
            scalar properties. If properties are not specified the structure
            object will be checked for an attached ASE calculator object
            with a calculated potential energy
        allow_duplicate
             whether or not to add the structure if there already exists a
             structure with identical cluster-vector
        sanity_check
            whether or not to carry out a sanity check before adding the
            structure. This includes checking occupations and volume.
        """

        # structure must have a proper format and label
        if not isinstance(structure, Atoms):
            raise TypeError('structure must be an ASE Atoms object not {}'.format(type(structure)))

        if user_tag is not None:
            if not isinstance(user_tag, str):
                raise TypeError('user_tag must be a string not {}.'.format(type(user_tag)))

        if sanity_check:
            self._cluster_space.assert_structure_compatibility(structure)

        # check for properties in attached calculator
        if properties is None:
            properties = {}
            if structure.calc is not None:
                if not structure.calc.calculation_required(structure, ['energy']):
                    energy = structure.get_potential_energy()
                    properties['energy'] = energy / len(structure)

        # check if there exist structures with identical cluster vectors
        structure_copy = structure.copy()
        cv = self._cluster_space.get_cluster_vector(structure_copy)
        if not allow_duplicate:
            for i, fs in enumerate(self):
                if np.allclose(cv, fs.cluster_vector):
                    msg = '{} and {} have identical cluster vectors'.format(
                        user_tag if user_tag is not None else 'Input structure',
                        fs.user_tag if fs.user_tag != 'None' else 'structure')
                    msg += ' at index {}'.format(i)
                    raise ValueError(msg)

        # add structure
        structure = FitStructure(structure_copy, user_tag, cv, properties)
        self._structure_list.append(structure)

    def get_condition_number(self, structure_indices: List[int] = None,
                             key: str = 'energy') -> float:
        """ Returns the condition number for the sensing matrix.

        A very large condition number can be a sign of multicollinearity,
        read more here https://en.wikipedia.org/wiki/Condition_number

        Parameters
        ----------
        structure_indices
            list of structure indices; by default (``None``) the
            method will return all fit data available.
        key
            key of properties dictionary

        Returns
        -------
        condition number of the sensing matrix
        """
        return np.linalg.cond(self.get_fit_data(structure_indices, key)[0])

    def get_fit_data(self, structure_indices: List[int] = None,
                     key: str = 'energy') -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns fit data for all structures. The cluster vectors and
        target properties for all structures are stacked into numpy arrays.

        Parameters
        ----------
        structure_indices
            list of structure indices; by default (``None``) the
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
            db.write(fit_structure.structure, data=data_dict)

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
    structure : Atoms
        supercell structure
    user_tag : str
        custom user tag
    cvs : np.ndarray
        calculated cluster vector for actual structure
    properties : dict
        dictionary of properties
    """

    def __init__(self, structure: Atoms, user_tag: str,
                 cv: np.ndarray, properties: dict = {}):
        self._structure = structure
        self._user_tag = user_tag
        self._cluster_vector = cv
        self.properties = properties

    @property
    def cluster_vector(self) -> np.ndarray:
        """calculated cluster vector"""
        return self._cluster_vector

    @property
    def structure(self) -> Atoms:
        """atomic structure"""
        return self._structure

    @property
    def user_tag(self) -> str:
        """structure label"""
        return str(self._user_tag)

    def __getattr__(self, key):
        """ Accesses properties if possible and returns value. """
        if key not in self.properties.keys():
            return super().__getattribute__(key)
        return self.properties[key]

    def __len__(self) -> int:
        """ Number of sites in the structure. """
        return len(self._structure)
