import pandas as pd
from datetime import datetime
import getpass
import socket
from ase import Atoms
from collections import OrderedDict
from icet.io.logging import logger


logger = logger.getChild('data_container')


class DataContainer:
    """
    Data container class, which serves for storing
    all information concerned with Monte Carlo
    simulations performed with mchammer.

    Parameters
    ----------
    atoms : ASE Atoms object
        reference atomic structure associated with the data container

    name_ensemble : str
        name of associated ensemble

    random_seed : int
        random seed used in random number generator

    Attributes
    ----------
    observables : dict
        dictionary of tag-type pair of added observables.

    parameters : dict
        dictionary of tag-value pair of added parameters.

    metadata : dict
        dictionary of tag-value pair of added metadata.

    data : Pandas data frame object
        Runtime data collected during the Monte Carlo simulation.

    Todo
    ----
    BaseEnsemble pass atoms=None as default. That must be changed after
    a more advanced implementation of the base classes has been done.
    """

    def __init__(self, atoms, ensemble_name: str, random_seed: int):
        """
        Initialize a DataContainer object.
        """

        if atoms is not None:
            assert isinstance(atoms, Atoms), \
             'Structure must be provided as ASE Atoms object'
            self.structure = atoms.copy()

        self._observables = OrderedDict()
        self._parameters = OrderedDict()
        self._metadata = OrderedDict()
        self._data = pd.DataFrame()

        self.add_parameter('random-seed', random_seed)

        self._metadata['ensemble-name'] = ensemble_name
        self._metadata['date-created'] = datetime.now()
        self._metadata['username'] = getpass.getuser()
        self._metadata['hostname'] = socket.gethostname()

    def add_observable(self, tag: str, obs_type):
        """
        Add an observable to the dict with observables.

        Parameters
        ----------
        tag : str
            name of observable.
        obs_type : type
            type of observable parameter.
        """
        assert isinstance(tag, str), \
            'Observable tag has wrong type (str)'
        assert isinstance(obs_type, (type, list)), \
            'Unknown observable type: {}'.format(obs_type)
        self._observables[tag] = obs_type

    def add_parameter(self, tag: str, value):
        """
        Add parameter of the associated ensemble.

        Parameters
        ----------
        tag : str
            parameter name
        value : int or float or list of int or float
            parameter value
        """
        import copy
        assert isinstance(tag, str), \
            'Parameter tag has the wrong type (str).'
        assert isinstance(value, (int, float, list)), \
            'Unknown parameter type: {}'.format(type(value))
        self._parameters[tag] = copy.deepcopy(value)

    def append(self, mctrial: int, record: dict):
        """
        Append data to the data container.

        Parameters
        ----------
        mctrial : int
            current Monte Carlo trial step
        record : dict
            dictionary of tag-value pairs representing observations

        Todo
        ----
        This might be a quite expensive way to add data to the data
        frame. Testing and profiling to be carried out later.
        """
        assert isinstance(mctrial, int), \
            'Monte Carlo trial step has wrong type (int)'
        assert isinstance(record, dict), \
            'Input record has wrong type (dict)'

        row_data = OrderedDict()
        row_data['mctrial'] = mctrial
        row_data.update(record)

        self._data = self._data.append(row_data,
                                       ignore_index=True)

    def get_data(self, tags=None, interval=None, fill_missing=False):
        """
        Returns a list of lists representing the accumulated data for
        the observables specified via tags.

        Parameters
        ----------
        tags : list of str
            tags of the required properties; if None all columns of the data
            frame will be returned

        interval : tuple
            range of trial steps values from which data frame will be returned

        fill_missing : bool
            If True fill missing values forward
        """
        import math

        if tags is None:
            tags = self._data.columns.tolist()
        else:
            for tag in tags:
                assert tag in self._data, \
                    'observable is not part of DataContainer: {}'.format(tag)

        if interval is None:
            data = self._data.loc[:, tags]
        else:
            assert isinstance(interval, tuple), \
                'interval must be a tuple: {}'.format(type(interval))
            assert len(interval) == 2, \
                'interval must contain only a lower and an upper value'
            data = self._data.set_index(self._data.mctrial)
            lower, upper = interval
            data = data.loc[lower:upper, tags]

        if fill_missing:
            data = data.fillna(method='pad')

        data_list = []
        for tag in tags:
            data_column = data.get(tag).tolist()
            data_column = [None if math.isnan(x) else x for x in data_column]
            data_list.append(data_column)

        return data_list

    @property
    def parameters(self):
        """ list : simulation parameters """
        return self._parameters.copy()

    @property
    def observables(self):
        """ dict : observables """
        return self._observables

    @property
    def metadata(self):
        """ dict : metadata associated with data container """
        return self._metadata

    def reset(self):
        """ Reset (clear) data frame of data container. """
        self._data = pd.DataFrame()

    def __len__(self):
        """ Number of rows in data frame """
        return len(self._data)

    def get_average(self, start: int, stop: int, tag: str):
        """
        Return average of an observable over an interval of trial steps.

        Parameters
        ----------
        start : int
            lower limit of trial step interval
        stop : int
            upper limit of trial step interval
        tag : str
            tag of field over which to average
        """
        pass

    def read(self, infile):
        """
        Read DataContainer object from file.

        Parameters
        ----------
        infile : str or FileObj
            file from which to read
        """
        pass

    def write(self, outfile):
        """
        Write DataContainer object to file.

        Parameters
        ----------
        outfile : str or FileObj
            file to which to write
        """
        self._metadata['date-last-backup'] = datetime.now()
        pass
