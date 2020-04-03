""" Base data container class. """

import getpass
import json
import numbers
import os
import shutil
import socket
import tarfile
import tempfile
import warnings

from collections import OrderedDict
from datetime import datetime
from typing import BinaryIO, Dict, List, TextIO, Tuple, Union

import numpy as np
import pandas as pd

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from icet import __version__ as icet_version
from ..observers.base_observer import BaseObserver


class Int64Encoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class BaseDataContainer:
    """
    Base data container for storing information concerned with
    Monte Carlo simulations performed with mchammer.

    Parameters
    ----------
    structure : ase.Atoms
        reference atomic structure associated with the data container

    ensemble_parameters : dict
        parameters associated with the underlying ensemble

    metadata : dict
        metadata associated with the data container
    """

    def __init__(self, structure: Atoms,
                 ensemble_parameters: dict,
                 metadata: dict = OrderedDict()):
        """
        Initializes a BaseDataContainer object.
        """
        if not isinstance(structure, Atoms):
            raise TypeError('structure is not an ASE Atoms object')

        self.structure = structure.copy()
        self._ensemble_parameters = ensemble_parameters
        self._metadata = metadata
        self._add_default_metadata()
        self._last_state = {}

        self._observables = set()
        self._data_list = []

    def append(self, mctrial: int, record: Dict[str, Union[int, float, list]]):
        """
        Appends data to data container.

        Parameters
        ----------
        mctrial
            current Monte Carlo trial step
        record
            dictionary of tag-value pairs representing observations

        Raises
        ------
        TypeError
            if input parameters have the wrong type

        """
        if not isinstance(mctrial, numbers.Integral):
            raise TypeError('mctrial has the wrong type: {}'.format(type(mctrial)))

        if self._data_list:
            if self._data_list[-1]['mctrial'] > mctrial:
                raise ValueError('mctrial values should be given in ascending'
                                 ' order. This error can for example occur'
                                 ' when trying to append to an existing data'
                                 ' container after having reset the time step.'
                                 ' Note that the latter happens automatically'
                                 ' when initializing a new ensemble.')

        if not isinstance(record, dict):
            raise TypeError('record has the wrong type: {}'.format(type(record)))

        for tag in record.keys():
            self._observables.add(tag)

        row_data = OrderedDict()
        row_data['mctrial'] = mctrial
        row_data.update(record)
        self._data_list.append(row_data)

    def _update_last_state(self, last_step: int, occupations: List[int],
                           accepted_trials: int, random_state: tuple):
        """Updates last state of the simulation: last step, occupation vector
        and number of accepted trial steps.

        Parameters
        ----------
        last_step
            last trial step
        occupations
            occupation vector observed during the last trial step
        accepted_trial
            number of current accepted trial steps
        random_state
            tuple representing the last state of the random generator
        """
        self._last_state['last_step'] = last_step
        self._last_state['occupations'] = occupations
        self._last_state['accepted_trials'] = accepted_trials
        self._last_state['random_state'] = random_state

    def apply_observer(self, observer: BaseObserver):
        """ Adds observer data from observer to data container.

        The observer will only be run for the mctrials for which the
        trajectory have been saved.

        The interval of the observer is ignored.

        Parameters
        ----------
        observer
            observer to be used
        """
        for row_data in self._data_list:
            if 'occupations' in row_data:
                structure = self.structure.copy()
                structure.numbers = row_data['occupations']
                record = dict()
                if observer.return_type is dict:
                    for key, value in observer.get_observable(structure).items():
                        record[key] = value
                        self._observables.add(key)
                else:
                    record[observer.tag] = observer.get_observable(structure)
                    self._observables.add(observer.tag)
                row_data.update(record)

    def get(self,
            *tags: str,
            start: int = 0) \
            -> Union[np.ndarray, List[Atoms], Tuple[np.ndarray, List[Atoms]]]:
        """Returns the accumulated data for the requested observables,
        including configurations stored in the data container. The latter
        can be achieved by including 'trajectory' as one of the tags.

        Parameters
        ----------
        tags
            names of the requested properties
        start
            minimum value of trial step to consider; by default the
            smallest value in the mctrial column will be used.

        Raises
        ------
        ValueError
            if tags is empty
        ValueError
            if observables are requested that are not in data container

        Examples
        --------
        Below the `get` method is illustrated but first we require a data container.

            >>> from ase.build import bulk
            >>> from icet import ClusterExpansion, ClusterSpace
            >>> from mchammer.calculators import ClusterExpansionCalculator
            >>> from mchammer.ensembles import CanonicalEnsemble

            >>> # prepare cluster expansion
            >>> prim = bulk('Au')
            >>> cs = ClusterSpace(prim, cutoffs=[4.3], chemical_symbols=['Ag', 'Au'])
            >>> ce = ClusterExpansion(cs, [0, 0, 0.1, -0.02])

            >>> # prepare initial configuration
            >>> structure = prim.repeat(3)
            >>> for k in range(5):
            ...     structure[k].symbol = 'Ag'

            >>> # set up and run MC simulation
            >>> calc = ClusterExpansionCalculator(structure, ce)
            >>> mc = CanonicalEnsemble(structure=structure, calculator=calc,
            ...                        temperature=600,
            ...                        dc_filename='myrun_canonical.dc')
            >>> mc.run(100)  # carry out 100 trial swaps

        We can now access the data container by reading it from file by using
        the `read` method. For the purpose of this example, however, we access
        the data container associated with the ensemble directly.

            >>> dc = mc.data_container

        The following lines illustrate how to use the `get` method
        for extracting data from the data container.

            >>> # obtain all values of the potential represented by
            >>> # the cluster expansion along the trajectory
            >>> p = dc.get('potential')

            >>> import matplotlib.pyplot as plt
            >>> # as above but this time the MC trial step is included as well
            >>> s, p = dc.get('mctrial', 'potential')
            >>> _ = plt.plot(s, p)
            >>> plt.show()

            >>> # obtain configurations along the trajectory along with
            >>> # their potential
            >>> p, confs = dc.get('potential', 'trajectory')
        """

        if len(tags) == 0:
            raise TypeError('Missing tags argument')

        local_tags = ['occupations' if tag == 'trajectory' else tag for tag in tags]

        for tag in local_tags:
            if tag in 'mctrial':
                continue
            if tag not in self.observables:
                raise ValueError('No observable named {} in data container'.format(tag))

        # collect data
        mctrials = [row_dict['mctrial'] for row_dict in self._data_list]
        data = pd.DataFrame.from_records(self._data_list, index=mctrials, columns=local_tags)
        data = data.loc[start::, local_tags].copy()
        data.dropna(inplace=True)

        # handling of trajectory
        def occupation_to_atoms(occupation):
            structure = self.structure.copy()
            structure.numbers = occupation
            return structure

        data_list = []
        for tag in local_tags:
            if tag == 'occupations':
                traj = [occupation_to_atoms(o) for o in data['occupations']]
                data_list.append(traj)
            else:
                data_list.append(data[tag].values)

        if len(data_list) > 1:
            return tuple(data_list)
        else:
            return data_list[0]

    @property
    def data(self) -> pd.DataFrame:
        """ pandas data frame (see :class:`pandas.DataFrame`) """
        if self._data_list:
            df = pd.DataFrame.from_records(self._data_list, index='mctrial',
                                           exclude=['occupations'])
            df.dropna(axis='index', how='all', inplace=True)
            df.reset_index(inplace=True)
            return df
        else:
            return pd.DataFrame()

    @property
    def ensemble_parameters(self) -> dict:
        """ parameters associated with Monte Carlo simulation """
        return self._ensemble_parameters.copy()

    @property
    def observables(self) -> List[str]:
        """ observable names """
        return list(self._observables)

    @property
    def metadata(self) -> dict:
        """ metadata associated with data container """
        return self._metadata

    def write(self, outfile: Union[str, BinaryIO, TextIO]):
        """
        Writes BaseDataContainer object to file.

        Parameters
        ----------
        outfile
            file to which to write
        """
        self._metadata['date_last_backup'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        # Save reference atomic structure
        reference_structure_file = tempfile.NamedTemporaryFile()
        ase_write(reference_structure_file.name, self.structure, format='json')

        # Save reference data
        data_container_type = str(self.__class__).split('.')[-1].replace("'>", '')
        reference_data = {'parameters': self._ensemble_parameters,
                          'metadata': self._metadata,
                          'last_state': self._last_state,
                          'data_container_type': data_container_type}

        reference_data_file = tempfile.NamedTemporaryFile()
        with open(reference_data_file.name, 'w') as handle:
            json.dump(reference_data, handle, cls=Int64Encoder)

        # Save runtime data
        runtime_data_file = tempfile.NamedTemporaryFile()
        np.savez_compressed(runtime_data_file, self._data_list)

        # Write temporary tar file
        with tempfile.NamedTemporaryFile('wb', delete=False) as f:
            with tarfile.open(fileobj=f, mode='w') as handle:
                handle.add(reference_data_file.name, arcname='reference_data')
                handle.add(reference_structure_file.name, arcname='atoms')
                handle.add(runtime_data_file.name, arcname='runtime_data')

        # Copy to permanent location
        file_name = f.name
        f.close()  # Required for Windows
        shutil.copy(file_name, outfile)
        os.remove(file_name)
        runtime_data_file.close()

    def _add_default_metadata(self):
        """Adds default metadata to metadata dict."""

        self._metadata['date_created'] = \
            datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        self._metadata['username'] = getpass.getuser()
        self._metadata['hostname'] = socket.gethostname()
        self._metadata['icet_version'] = icet_version

    def __str__(self):
        """ string representation of data container """
        width = 80
        s = []  # type: List
        s += ['{s:=^{n}}'.format(s=' Data Container ', n=width)]
        data_container_type = str(self.__class__).split('.')[-1].replace("'>", '')
        s += [' {:22}: {}'.format('data_container_type', data_container_type)]
        for key, value in self._last_state.items():
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                s += [' {:22}: {}'.format(key, value)]
        for key, value in sorted(self._ensemble_parameters.items()):
            s += [' {:22}: {}'.format(key, value)]
        for key, value in sorted(self._metadata.items()):
            s += [' {:22}: {}'.format(key, value)]
        s += [' {:22}: {}'.format('columns_in_data', self.data.columns.tolist())]
        s += [' {:22}: {}'.format('n_rows_in_data', len(self.data))]
        s += [''.center(width, '=')]
        return '\n'.join(s)

    @classmethod
    # todo: cls and the return should be type hinted as BaseDataContainer.
    # Unfortunately, this requires from __future__ import annotations, which
    # in turn requires Python 3.8.
    def read(cls, infile: Union[str, BinaryIO, TextIO], old_format: bool = False):
        """Reads data container from file.

        Parameters
        ----------
        infile
            file from which to read
        old_format
            If true use old json format to read runtime data; default to false

        Raises
        ------
        FileNotFoundError
            if file is not found (str)
        ValueError
            if file is of incorrect type (not a tarball)
        """
        if isinstance(infile, str):
            filename = infile
        else:
            filename = infile.name

        if not tarfile.is_tarfile(filename):
            raise TypeError('{} is not a tar file'.format(filename))

        reference_structure_file = tempfile.NamedTemporaryFile()
        reference_data_file = tempfile.NamedTemporaryFile()
        runtime_data_file = tempfile.NamedTemporaryFile()

        with tarfile.open(mode='r', name=filename) as tar_file:
            # file with structures
            reference_structure_file.write(tar_file.extractfile('atoms').read())

            reference_structure_file.seek(0)
            structure = ase_read(reference_structure_file.name, format='json')

            # file with reference data
            reference_data_file.write(tar_file.extractfile('reference_data').read())
            reference_data_file.seek(0)
            with open(reference_data_file.name, encoding='utf-8') as fd:
                reference_data = json.load(fd)

            # init DataContainer
            dc = cls(structure=structure,
                     ensemble_parameters=reference_data['parameters'])

            # overwrite metadata
            dc._metadata = reference_data['metadata']

            for tag, value in reference_data['last_state'].items():
                if tag == 'random_state':
                    value = tuple(tuple(x) if isinstance(x, list) else x for x in value)
                dc._last_state[tag] = value

            # add runtime data from file
            runtime_data_file.write(tar_file.extractfile('runtime_data').read())
            runtime_data_file.seek(0)
            if old_format:
                runtime_data = pd.read_json(runtime_data_file)
                data = runtime_data.sort_index(ascending=True)
                dc._data_list = data.T.apply(lambda x: x.dropna().to_dict()).tolist()
            else:
                dc._data_list = np.load(runtime_data_file, allow_pickle=True)['arr_0'].tolist()

        dc._observables = set([key for data in dc._data_list for key in data])
        dc._observables = dc._observables - {'mctrial'}

        return dc

    def get_data(self, *args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('get_data is deprecated, use get instead', DeprecationWarning)
        return self.get(*args, **kwargs)

    def get_trajectory(self, *args, **kwargs):
        """ Returns trajectory as a list of ASE Atoms objects."""
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('get_trajectory is deprecated, use get instead', DeprecationWarning)
        return self.get('trajectory', *args, **kwargs)
