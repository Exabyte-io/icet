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
from typing import Any, BinaryIO, Dict, List, Set, TextIO, Tuple, Union

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
    Monte Carlo simulations performed with :program:`mchammer`.

    Parameters
    ----------
    structure
        Reference atomic structure associated with the data container.
    ensemble_parameters
        Parameters associated with the underlying ensemble.
    metadata
        Metadata associated with the data container.
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
        self._last_state: Dict[str, Any] = {}

        self._observables: Set[str] = set()
        self._data_list: List[Dict[str, Any]] = []

    def append(self, mctrial: int, record: Dict[str, Union[int, float, list]]):
        """
        Appends data to data container.

        Parameters
        ----------
        mctrial
            Current Monte Carlo trial step.
        record
            Dictionary of tag-value pairs representing observations.

        Raises
        ------
        TypeError
            If input parameters have the wrong type.

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
                           accepted_trials: int, random_state: Any) -> None:
        """Updates last state of the simulation: last step, occupation vector
        and number of accepted trial steps.

        Parameters
        ----------
        last_step
            Last trial step.
        occupations
            Occupation vector observed during the last trial step.
        accepted_trial
            Number of current accepted trial steps.
        random_state
            Tuple representing the last state of the random generator.
        """
        self._last_state['last_step'] = last_step
        self._last_state['occupations'] = occupations
        self._last_state['accepted_trials'] = accepted_trials
        self._last_state['random_state'] = random_state

    def apply_observer(self, observer: BaseObserver) -> None:
        """ Adds observer data from observer to data container.

        The observer will only be run for the mctrials for which the
        trajectory have been saved.

        The interval of the observer is ignored.

        Parameters
        ----------
        observer
            Observer to be used.
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
        can be achieved by including ``'trajectory'`` as one of the tags.

        Parameters
        ----------
        tags
            Names of the requested properties.
        start
            Minimum value of trial step to consider. By default the
            smallest value in the mctrial column will be used.

        Raises
        ------
        ValueError
            If :attr:`tags` is empty.
        ValueError
            If observables are requested that are not in data container.

        Examples
        --------
        Below the :func:`get` method is illustrated but first we require a data container.

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
        the :func:`read` method. For the purpose of this example, however, we access
        the data container associated with the ensemble directly.

            >>> dc = mc.data_container

        The following lines illustrate how to use the :func:`get` method
        for extracting data from the data container.

            >>> # obtain all values of the potential represented by
            >>> # the cluster expansion along the trajectory
            >>> p = dc.get('potential')

            >>> import matplotlib.pyplot as plt
            >>> # as above but this time the MC trial step is included as well
            >>> s, p = dc.get('mctrial', 'potential')
            >>> _ = plt.plot(s, p)
            >>> plt.show(block=False)

            >>> # obtain configurations along the trajectory along with
            >>> # their potential
            >>> p, confs = dc.get('potential', 'trajectory')
        """

        if len(tags) == 0:
            raise TypeError('Missing tags argument')

        local_tags = ['occupations' if tag == 'trajectory' else tag for tag in tags]

        for tag in local_tags:
            if isinstance(tag, str) and tag in 'mctrial':
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
        """ Data as :class:`pandas.DataFrame`. """
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
        """ Parameters associated with Monte Carlo simulation. """
        return self._ensemble_parameters.copy()

    @property
    def observables(self) -> List[str]:
        """ Observable names. """
        return list(self._observables)

    @property
    def metadata(self) -> dict:
        """ Metadata associated with data container. """
        return self._metadata

    def write(self, outfile: Union[bytes, str]):
        """
        Writes data container to file.

        Parameters
        ----------
        outfile
            File to which to write.
        """
        self._metadata['date_last_backup'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        # Save reference atomic structure
        reference_structure_file = tempfile.NamedTemporaryFile(delete=False)
        ase_write(reference_structure_file.name, self.structure, format='json')

        # Save reference data
        data_container_type = str(self.__class__).split('.')[-1].replace("'>", '')
        reference_data = {'parameters': self._ensemble_parameters,
                          'metadata': self._metadata,
                          'last_state': self._last_state,
                          'data_container_type': data_container_type}

        reference_data_file = tempfile.NamedTemporaryFile(delete=False)
        with open(reference_data_file.name, 'w') as fileobj:
            json.dump(reference_data, fileobj, cls=Int64Encoder)

        # Save runtime data
        runtime_data_file = tempfile.NamedTemporaryFile(delete=False)
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

        reference_data_file.close()
        reference_structure_file.close()
        runtime_data_file.close()
        os.remove(reference_data_file.name)
        os.remove(reference_structure_file.name)
        os.remove(runtime_data_file.name)

    def _add_default_metadata(self):
        """Adds default metadata to metadata dict."""

        self._metadata['date_created'] = \
            datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        self._metadata['username'] = getpass.getuser()
        self._metadata['hostname'] = socket.gethostname()
        self._metadata['icet_version'] = icet_version

    def __str__(self):
        """ String representation of data container. """
        width = 80
        s = []
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

    def _repr_html_(self) -> str:
        """ HTML representation. Used, e.g., in jupyter notebooks. """
        s = ['<h4>Data Container</h4>']
        s += ['<table border="1" class="dataframe">']
        s += ['<thead><tr><th style="text-align: left;">Field</th><th>Value</th></tr></thead>']
        s += ['<tbody>']
        data_container_type = str(self.__class__).split('.')[-1].replace("'>", '')
        s += [f'<tr><td style="text-align: left;">Type</td><td>{data_container_type}</td></tr>']
        for key, value in self._last_state.items():
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                s += [f'<tr><td style="text-align: left;">{key}</td><td>{value}</td></tr>']
        for key, value in sorted(self._ensemble_parameters.items()):
            s += [f'<tr><td style="text-align: left;">{key}</td><td>{value}</td></tr>']
        for key, value in sorted(self._metadata.items()):
            s += [f'<tr><td style="text-align: left;">{key}</td><td>{value}</td></tr>']
        s += ['<tr><td style="text-align: left;">Number of rows in data</td>'
              f'<td>{len(self.data)}</td></tr>']
        s += ['</tbody>']
        s += ['</table>']
        return ''.join(s)

    @classmethod
    def read(cls, infile: Union[str, BinaryIO, TextIO], old_format: bool = False):
        """Reads data container from file.

        Parameters
        ----------
        infile
            File from which to read.
        old_format
            If ``True`` use old json format to read runtime data.

        Raises
        ------
        FileNotFoundError
            If :attr:`infile` is not found.
        ValueError
            If :attr:`file` is of incorrect type (not a tarball).
        """
        if isinstance(infile, str):
            filename = infile
        else:
            filename = infile.name

        if not tarfile.is_tarfile(filename):
            raise TypeError('{} is not a tar file'.format(filename))

        with tarfile.open(mode='r', name=filename) as tf:
            # file with structures
            with tempfile.NamedTemporaryFile(delete=False) as fobj:
                fobj.write(tf.extractfile('atoms').read())
                fobj.flush()
                structure = ase_read(fobj.name, format='json')
            tmpfname = fobj.name
            os.remove(tmpfname)

            # file with reference data
            with tempfile.NamedTemporaryFile(delete=False) as fobj:
                fobj.write(tf.extractfile('reference_data').read())
                fobj.flush()
                with open(fobj.name, encoding='utf-8') as fd:
                    reference_data = json.load(fd)
            tmpfname = fobj.name
            os.remove(tmpfname)

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
            with tempfile.NamedTemporaryFile(delete=False) as fobj:
                fobj.write(tf.extractfile('runtime_data').read())
                fobj.seek(0)
                if old_format:
                    runtime_data = pd.read_json(fobj)
                    data = runtime_data.sort_index(ascending=True)
                    dc._data_list = data.T.apply(lambda x: x.dropna().to_dict()).tolist()
                else:
                    dc._data_list = np.load(fobj, allow_pickle=True)['arr_0'].tolist()
            tmpfname = fobj.name
            os.remove(tmpfname)

        dc._observables = set([key for data in dc._data_list for key in data])
        dc._observables = dc._observables - {'mctrial'}

        return dc

    def get_data(self, *args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('get_data is deprecated, use get instead', DeprecationWarning)
        return self.get(*args, **kwargs)

    def get_trajectory(self, *args, **kwargs) -> List[Atoms]:
        """ Returns trajectory as a list of ASE Atoms objects."""
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('get_trajectory is deprecated, use get instead', DeprecationWarning)
        return self.get('trajectory', *args, **kwargs)
