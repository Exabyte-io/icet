"""Definition of the Wang-Landau data container class."""

from warnings import warn
from collections import Counter, OrderedDict
from typing import Any, BinaryIO, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd

from ase.units import kB
from ase import Atoms
from pandas import DataFrame, concat as pd_concat

from icet import ClusterSpace
from .base_data_container import BaseDataContainer


class WangLandauDataContainer(BaseDataContainer):
    """
    Data container for storing information concerned with :ref:`Wang-Landau
    simulation <wang_landau_ensemble>` performed with :program:`mchammer`.

    Parameters
    ----------
    structure
        Reference atomic structure associated with the data container.
    ensemble_parameters
        Parameters associated with the underlying ensemble.
    metadata
        Metadata associated with the data container.
    """

    def _update_last_state(self,
                           last_step: int,
                           occupations: List[int],
                           accepted_trials: int,
                           random_state: tuple,
                           fill_factor: float,
                           fill_factor_history: Dict[int, float],
                           entropy_history: Dict[int, Dict[int, float]],
                           histogram=Dict[int, int],
                           entropy=Dict[int, float]):
        """Updates last state of the Wang-Landau simulation.

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
        fill_factor
            Fill factor of Wang-Landau algorithm.
        fill_factor_history
            Evolution of the fill factor of Wang-Landau algorithm (key=MC
            trial step, value=fill factor).
        entropy_history
            Evolution of the (relative) entropy accumulated during Wang-Landau
            simulation (key=MC trial step, value=(key=bin, value=entropy)).
        histogram
            Histogram of states visited during Wang-Landau simulation.
        entropy
            (Relative) entropy accumulated during Wang-Landau simulation.
        """
        super()._update_last_state(
            last_step=last_step,
            occupations=occupations,
            accepted_trials=accepted_trials,
            random_state=random_state)
        self._last_state['fill_factor'] = fill_factor
        self._last_state['fill_factor_history'] = fill_factor_history
        self._last_state['entropy_history'] = entropy_history
        self._last_state['histogram'] = histogram
        self._last_state['entropy'] = entropy

    @property
    def fill_factor(self) -> float:
        """ Final value of the fill factor in the Wang-Landau algorithm. """
        return float(self._last_state['fill_factor'])

    @property
    def fill_factor_history(self) -> DataFrame:
        """ Evolution of the fill factor in the Wang-Landau algorithm. """
        return DataFrame({'mctrial': list(self._last_state['fill_factor_history'].keys()),
                          'fill_factor': list(self._last_state['fill_factor_history'].values())})

    def get(self,
            *tags: str,
            fill_factor_limit: float = None) \
            -> Union[np.ndarray, List[Atoms], Tuple[np.ndarray, List[Atoms]]]:
        """Returns the accumulated data for the requested observables,
        including configurations stored in the data container. The latter
        can be achieved by including ``'trajectory'`` as one of the tags.

        Parameters
        ----------
        tags
            Names of the requested properties.
        fill_factor_limit
            Return data recorded up to the point when the specified fill
            factor limit was reached, or ``None`` if the entropy history is
            empty or the last fill factor is above the limit; otherwise
            return all data.

        Raises
        ------
        ValueError
            If :attr:`tags` is empty.
        ValueError
            If observables are requested that are not in the data container.

        Examples
        --------
        Below the :func:`get` method is
        illustrated but first we require a data container.

        >>> from ase import Atoms
        >>> from icet import ClusterExpansion, ClusterSpace
        >>> from mchammer.calculators import ClusterExpansionCalculator
        >>> from mchammer.ensembles import WangLandauEnsemble

        >>> # prepare cluster expansion
        >>> prim = Atoms('Au', positions=[[0, 0, 0]], cell=[1, 1, 10], pbc=True)
        >>> cs = ClusterSpace(prim, cutoffs=[1.1], chemical_symbols=['Ag', 'Au'])
        >>> ce = ClusterExpansion(cs, [0, 0, 2])

        >>> # prepare initial configuration
        >>> structure = prim.repeat((4, 4, 1))
        >>> for k in range(8):
        ...     structure[k].symbol = 'Ag'

        >>> # set up and run Wang-Landau simulation
        >>> calculator = ClusterExpansionCalculator(structure, ce)
        >>> mc = WangLandauEnsemble(structure=structure,
        ...                         calculator=calculator,
        ...                         energy_spacing=1,
        ...                         dc_filename='ising_2d_run.dc',
        ...                         fill_factor_limit=0.3)
        >>> mc.run(number_of_trial_steps=len(structure)*3000)  # in practice one requires more steps

        We can now access the data container by reading it from file
        by using the :func:`read` method. For the purpose of this
        example, however, we access the data container associated with
        the ensemble directly.

            >>> dc = mc.data_container

        The following lines illustrate how to use the :func:`get` method
        for extracting data from the data container.

            >>> # obtain all values of the potential represented by
            >>> # the cluster expansion and the MC trial step along the
            >>> # trajectory
            >>> import matplotlib.pyplot as plt
            >>> s, p = dc.get('mctrial', 'potential')
            >>> _ = plt.plot(s, p)

            >>> # as above but this time only included data recorded up to
            >>> # the point when the fill factor reached below 0.6
            >>> s, p = dc.get('mctrial', 'potential', fill_factor_limit=0.6)
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
            if tag in 'mctrial':
                continue
            if tag not in self.observables:
                raise ValueError('No observable named {} in data container'.format(tag))

        # collect data
        mctrials = [row_dict['mctrial'] for row_dict in self._data_list]
        data = pd.DataFrame.from_records(self._data_list, index=mctrials, columns=local_tags)
        if fill_factor_limit is not None:
            # only include data for fill factors up to the limit
            df_ffh = self.fill_factor_history.astype(
                {'mctrial': np.int64, 'fill_factor': np.float64})
            mctrial_last = df_ffh.loc[
                df_ffh.fill_factor <= fill_factor_limit].mctrial.min()
            data = data.loc[data.index <= mctrial_last]
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

    def get_entropy(self, fill_factor_limit: float = None) -> DataFrame:
        """Returns the (relative) entropy from this data container accumulated
        during a :ref:`Wang-Landau simulation <wang_landau_ensemble>`. Returns
        ``None`` if the data container does not contain the required
        information.

        Parameters
        ----------
        fill_factor_limit
            Return the entropy recorded up to the point when the specified fill
            factor limit was reached, or ``None`` if the entropy history is
            empty or the last fill factor is above the limit. Otherwise
            return the entropy for the last state.
        """

        if 'entropy' not in self._last_state:
            warn('There is no entropy information in the data container.')
            return None
        entropy = self._last_state['entropy']
        if fill_factor_limit is not None:
            if 'entropy_history' not in self._last_state or \
                    len(self._last_state['entropy_history']) == 0:
                warn('The entropy history is empty.')
                return None
            if self._last_state['fill_factor'] > fill_factor_limit:
                warn('The last fill factor {} is higher than the limit'
                     ' {}.'.format(self.fill_factor, fill_factor_limit))
                return None
            for step, fill_factor in self._last_state['fill_factor_history'].items():
                if fill_factor <= fill_factor_limit:
                    entropy = self._last_state['entropy_history'][step]
                    break

        # compile entropy into DataFrame
        energy_spacing = self.ensemble_parameters['energy_spacing']
        df = DataFrame(data={'energy': energy_spacing * np.array(list(entropy.keys())),
                             'entropy': np.array(list(entropy.values()))},
                       index=list(entropy.keys()))
        # shift entropy for numerical stability
        df['entropy'] -= np.min(df['entropy'])

        return df

    def get_histogram(self) -> DataFrame:
        """Returns the histogram from this data container accumulated since the
        last update of the fill factor. Returns ``None`` if the data container
        does not contain the required information.
        """

        if 'histogram' not in self._last_state:
            return None

        # compile histogram into DataFrame
        histogram = self._last_state['histogram']
        energy_spacing = self.ensemble_parameters['energy_spacing']
        df = DataFrame(data={'energy': energy_spacing * np.array(list(histogram.keys())),
                             'histogram': np.array(list(histogram.values()))},
                       index=list(histogram.keys()))

        return df

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
        dc = super(WangLandauDataContainer, cls).read(infile=infile, old_format=old_format)

        for tag, value in dc._last_state.items():
            if tag in ['histogram', 'entropy', 'fill_factor_history', 'entropy_history']:
                # the following accounts for the fact that the keys of dicts
                # are converted to str when writing to json and have to
                # converted back into numerical values
                dc._last_state[tag] = {}
                for key, val in value.items():
                    if isinstance(val, dict):
                        val = {int(k): v for k, v in val.items()}
                    dc._last_state[tag][int(key)] = val

        return dc


def get_density_of_states_wl(dcs: Union[WangLandauDataContainer,
                                        Dict[Any, WangLandauDataContainer]],
                             fill_factor_limit: float = None) \
        -> Tuple[DataFrame, dict]:
    """Returns a pandas DataFrame with the total density of states from a
    :ref:`Wang-Landau simulation <wang_landau_ensemble>`. If a dict of data
    containers is provided the function also returns a dictionary that
    contains the standard deviation between the entropy of neighboring data
    containers in the overlap region. These errors should be small compared to
    the variation of the entropy across each bin.

    The function can handle both a single data container and a dict thereof.
    In the latter case the data containers must cover a contiguous energy
    range and must at least partially overlap.

    Parameters
    ----------
    dcs
        Data container(s), from which to extract the density of states.
    fill_factor_limit
        Calculate the density of states using the entropy recorded up to the
        point when the specified fill factor limit was reached. Otherwise
        return the density of states for the last state.

    Raises
    ------
    TypeError
        If :attr:`dcs` does not correspond to not a single (dictionary) of data
        container(s) from which the entropy can retrieved.
    ValueError
        If the data container does not contain entropy information.
    ValueError
        If a fill factor limit has been provided and the data container either
        does not contain information about the entropy history or if the last
        fill factor is higher than the specified limit.
    ValueError
        If multiple data containers are provided and there are inconsistencies
        with regard to basic simulation parameters such as system size or
        energy spacing.
    ValueError
        If multiple data containers are provided and there is at least
        one energy region without overlap.
    """

    # preparations
    if isinstance(dcs, WangLandauDataContainer):
        # fetch raw entropy data from data container
        df = dcs.get_entropy(fill_factor_limit)
        if df is None:
            raise ValueError('Entropy information could not be retrieved from'
                             ' the data container {}.'.format(dcs))
        errors = None
        if len(dcs.fill_factor_history) == 0 or dcs.fill_factor > 1e-4:
            warn('The data container appears to contain data from an'
                 ' underconverged Wang-Landau simulation.')

    elif isinstance(dcs, dict) and isinstance(dcs[next(iter(dcs))], WangLandauDataContainer):
        # minimal consistency checks
        tags = list(dcs.keys())
        tagref = tags[0]
        dcref = dcs[tagref]
        for tag in tags:
            dc = dcs[tag]
            if len(dc.structure) != len(dcref.structure):
                raise ValueError('Number of atoms differs between data containers ({}: {}, {}: {})'
                                 .format(tagref, dcref.ensemble_parameters['n_atoms'],
                                         tag, dc.ensemble_parameters['n_atoms']))
            for param in ['energy_spacing', 'trial_move']:
                if dc.ensemble_parameters[param] != dcref.ensemble_parameters[param]:
                    raise ValueError('{} differs between data containers ({}: {}, {}: {})'
                                     .format(param,
                                             tagref, dcref.ensemble_parameters['n_atoms'],
                                             tag, dc.ensemble_parameters['n_atoms']))
                if len(dc.fill_factor_history) == 0 or dc.fill_factor > 1e-4:
                    warn('Data container {} appears to contain data from an'
                         ' underconverged Wang-Landau simulation.'.format(tag))

        # fetch raw entropy data from data containers
        entropies = {}
        for tag, dc in dcs.items():
            entropies[tag] = dc.get_entropy(fill_factor_limit)
            if entropies[tag] is None:
                raise ValueError('Entropy information could not be retrieved'
                                 ' from the data container {}.'.format(dc))

        # sort entropies by energy
        entropies = OrderedDict(sorted(entropies.items(), key=lambda row: row[1].energy.iloc[0]))

        # line up entropy data
        errors = {}
        tags = list(entropies.keys())
        for tag1, tag2 in zip(tags[:-1], tags[1:]):
            df1 = entropies[tag1]
            df2 = entropies[tag2]
            if all(df2.energy.isin(df1.energy)):
                warn('Window {} is a subset of {}'.format(tag2, tag1))
            left_lim = np.min(df2.energy)
            right_lim = np.max(df1.energy)
            if left_lim >= right_lim:
                raise ValueError('No overlap in the energy range {}...{}.\n'
                                 .format(right_lim, left_lim) +
                                 ' The closest data containers have tags "{}" and "{}".'
                                 .format(tag1, tag2))
            df1_ = df1[(df1.energy >= left_lim) & (df1.energy <= right_lim)]
            df2_ = df2[(df2.energy >= left_lim) & (df2.energy <= right_lim)]
            offset = (df2_.entropy - df1_.entropy).mean()
            errors['{}-{}'.format(tag1, tag2)] = (df2_.entropy - df1_.entropy).std()
            entropies[tag2].entropy = entropies[tag2].entropy - offset

        # compile entropy over the entire energy range
        data: Dict[float, float] = {}
        indices = {}
        counts = Counter()
        for df in entropies.values():
            for index, en, ent in zip(df.index, df.energy, df.entropy):
                data[en] = data.get(en, 0) + ent
                counts[en] += 1
                indices[en] = index
        for en in data:
            data[en] = data[en] / counts[en]

        # center entropy to prevent possible numerical issues
        entmin = np.min(list(data.values()))
        df = DataFrame(data={'energy': np.array(list(data.keys())),
                             'entropy': np.array(np.array(list(data.values()))) - entmin},
                       index=list(indices.values()))
    else:
        raise TypeError('dcs ({}) must be a data container with entropy data'
                        ' or be a list of data containers'
                        .format(type(dcs)))

    # density of states
    S_max = df.entropy.max()
    df['density'] = np.exp(df.entropy - S_max) / np.sum(np.exp(df.entropy - S_max))

    return df, errors


def _extract_filter_data(dc: BaseDataContainer,
                         columns_to_keep: List[str],
                         fill_factor_limit: float = None) -> DataFrame:
    """ Extract data from a data container and filter the content.

    Parameters
    ----------
    dc
        Data container from which to extract the data.
    columns_to_keep
        List of requested properties.
    fill_factor_limit
        Only include data recorded up to the point when the specified fill
        factor limit was reached when computing averages. Otherwise include
        all data.
    """

    df = dc.data
    if fill_factor_limit is not None:
        # only include data for fill factors up to the limit
        df_ffh = dc.fill_factor_history.astype(
            {'mctrial': np.int64, 'fill_factor': np.float64})
        mctrial_last = df_ffh.loc[
            df_ffh.fill_factor <= fill_factor_limit].mctrial.min()
        df = df.loc[df.mctrial <= mctrial_last]

    return df.filter(columns_to_keep)


def get_average_observables_wl(dcs: Union[WangLandauDataContainer,
                                          Dict[Any, WangLandauDataContainer]],
                               temperatures: List[float],
                               observables: List[str] = None,
                               boltzmann_constant: float = kB,
                               fill_factor_limit: float = None) -> DataFrame:
    """Returns the average and the standard deviation of the energy from a
    :ref:`Wang-Landau simulation <wang_landau_ensemble>` for the temperatures
    specified. If the :attr:`observables` keyword argument is specified
    the function will also return the mean and standard deviation of the
    specified observables.

    Parameters
    ----------
    dcs
        Data container(s) from which to extract density of states
        as well as observables.
    temperatures
        Temperatures at which to compute the averages.
    observables
        Observables for which to compute averages; the observables
        must refer to fields in the data container.
    boltzmann_constant
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e., units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K].
    fill_factor_limit
        Use data recorded up to the point when the specified fill factor limit
        was reached when computing averages. Otherwise use data for the last
        state.

    Raises
    ------
    ValueError
        If the data container(s) do(es) not contain entropy data
        from Wang-Landau simulation.
    ValueError
        If data container(s) do(es) not contain requested observable.
    """

    def check_observables(dc: WangLandauDataContainer, observables: Optional[List[str]]) -> None:
        """ Helper function that checks that observables are available in data frame. """
        if observables is None:
            return
        for obs in observables:
            if obs not in dc.data.columns:
                raise ValueError('Observable ({}) not in data container.\n'
                                 'Available observables: {}'.format(obs, dc.data.columns))

    # preparation of observables
    columns_to_keep = ['potential', 'density']
    if observables is not None:
        columns_to_keep.extend(observables)

    # check that observables are available in data container
    # and prepare comprehensive data frame with relevant information
    if isinstance(dcs, WangLandauDataContainer):
        check_observables(dcs, observables)
        df_combined = _extract_filter_data(dcs, columns_to_keep, fill_factor_limit)
        dcref = dcs
    elif isinstance(dcs, dict) and isinstance(dcs[next(iter(dcs))], WangLandauDataContainer):
        dfs = []
        for dc in dcs.values():
            check_observables(dc, observables)
            dfs.append(_extract_filter_data(dc, columns_to_keep, fill_factor_limit))
        df_combined = pd_concat([df for df in dfs], ignore_index=True)
        dcref = list(dcs.values())[0]
    else:
        raise TypeError('dcs ({}) must be a data container with entropy data'
                        ' or be a list of data containers'
                        .format(type(dcs)))

    # fetch entropy and density of states from data container(s)
    df_density, _ = get_density_of_states_wl(dcs, fill_factor_limit)

    # compute density for each row in data container if observable averages
    # are to be computed
    if observables is not None:
        energy_spacing = dcref.ensemble_parameters['energy_spacing']
        # NOTE: we rely on the indices of the df_density DataFrame to
        # correspond to the energy scale! This is expected to be handled in
        # the get_density_of_states function.
        bins = list(np.array(np.round(df_combined.potential / energy_spacing), dtype=int))
        data_density = [dens / bins.count(k) for k, dens in df_density.density[bins].items()]

    enref = np.min(df_density.energy)
    averages = []
    for temperature in temperatures:

        # mean and standard deviation of energy
        boltz = np.exp(- (df_density.energy - enref) / temperature / boltzmann_constant)
        sumint = np.sum(df_density.density * boltz)
        en_mean = np.sum(df_density.energy * df_density.density * boltz) / sumint
        en_std = np.sum(df_density.energy ** 2 * df_density.density * boltz) / sumint
        en_std = np.sqrt(en_std - en_mean ** 2)
        record = {'temperature': temperature,
                  'potential_mean': en_mean,
                  'potential_std': en_std}

        # mean and standard deviation of other observables
        if observables is not None:
            boltz = np.exp(- (df_combined.potential - enref) / temperature / boltzmann_constant)
            sumint = np.sum(data_density * boltz)
            for obs in observables:
                obs_mean = np.sum(data_density * boltz * df_combined[obs]) / sumint
                obs_std = np.sum(data_density * boltz * df_combined[obs] ** 2) / sumint
                obs_std = np.sqrt(obs_std - obs_mean ** 2)
                record['{}_mean'.format(obs)] = obs_mean
                record['{}_std'.format(obs)] = obs_std

        averages.append(record)

    return DataFrame.from_dict(averages)


def get_average_cluster_vectors_wl(dcs: Union[WangLandauDataContainer, dict],
                                   cluster_space: ClusterSpace,
                                   temperatures: List[float],
                                   boltzmann_constant: float = kB,
                                   fill_factor_limit: float = None) -> DataFrame:
    """Returns the average cluster vectors from a :ref:`Wang-Landau simulation
    <wang_landau_ensemble>` for the temperatures specified.

    Parameters
    ----------
    dcs
        Data container(s), from which to extract density of states
        as well as observables.
    cluster_space
        Cluster space to use for calculation of cluster vectors.
    temperatures
        Temperatures at which to compute the averages.
    boltzmann_constant
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e., units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K].
    fill_factor_limit
        Use data recorded up to the point when the specified fill factor limit
        was reached when computing the average cluster vectors. Otherwise use
        data for the last state.

    Raises
    ------
    ValueError
        If the data container(s) do(es) not contain entropy data
        from Wang-Landau simulation.
    """

    # fetch potential and structures
    if isinstance(dcs, WangLandauDataContainer):
        potential, trajectory = dcs.get('potential', 'trajectory',
                                        fill_factor_limit=fill_factor_limit)
        energy_spacing = dcs.ensemble_parameters['energy_spacing']
    elif isinstance(dcs, dict) and isinstance(dcs[next(iter(dcs))], WangLandauDataContainer):
        potential, trajectory = [], []
        for dc in dcs.values():
            p, t = dc.get('potential', 'trajectory', fill_factor_limit=fill_factor_limit)
            potential.extend(p)
            trajectory.extend(t)
        energy_spacing = list(dcs.values())[0].ensemble_parameters['energy_spacing']
        potential = np.array(potential)
    else:
        raise TypeError('dcs ({}) must be a data container with entropy data'
                        ' or be a list of data containers'
                        .format(type(dcs)))

    # fetch entropy and density of states from data container(s)
    df_density, _ = get_density_of_states_wl(dcs, fill_factor_limit)

    # compute weighted density and cluster vector for each bin in energy
    # range; the weighted density is the total density divided by the number
    # of structures that fall in the respective bin
    # NOTE: the following code relies on the indices of the df_density
    # DataFrame to correspond to the energy scale. This is expected to be
    # handled in the get_density_of_states function.
    cvs = []
    weighted_density = []
    bins = list(np.array(np.round(potential / energy_spacing), dtype=int))
    for k, structure in zip(bins, trajectory):
        cvs.append(cluster_space.get_cluster_vector(structure))
        weighted_density.append(df_density.density[k] / bins.count(k))

    # compute mean and standard deviation (std) of temperature weighted
    # cluster vector
    averages = []
    enref = np.min(potential)
    for temperature in temperatures:
        boltz = np.exp(- (potential - enref) / temperature / boltzmann_constant)
        sumint = np.sum(weighted_density * boltz)
        cv_mean = np.array([np.sum(weighted_density * boltz * cv) / sumint
                            for cv in np.transpose(cvs)])
        cv_std = np.array([np.sum(weighted_density * boltz * cv ** 2) / sumint
                           for cv in np.transpose(cvs)])
        cv_std = np.sqrt(cv_std - cv_mean ** 2)
        record = {'temperature': temperature,
                  'cv_mean': cv_mean,
                  'cv_std': cv_std}
        averages.append(record)

    return DataFrame.from_dict(averages)
