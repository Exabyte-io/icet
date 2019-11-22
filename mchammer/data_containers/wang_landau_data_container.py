"""Definition of the Wang-Landau data container class."""

from warnings import warn
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple, Union

import numpy as np

from ase.units import kB
from pandas import DataFrame, concat as pd_concat

from icet import ClusterSpace
from .base_data_container import BaseDataContainer


class WangLandauDataContainer(BaseDataContainer):
    """
    Data container for storing information concerned with :ref:`Wang-Landau
    simulation <wang_landau_ensemble>` performed with mchammer.

    Parameters
    ----------
    structure : ase.Atoms
        reference atomic structure associated with the data container

    ensemble_parameters : dict
        parameters associated with the underlying ensemble

    metadata : dict
        metadata associated with the data container
    """

    def _update_last_state(self,
                           last_step: int,
                           occupations: List[int],
                           accepted_trials: int,
                           random_state: tuple,
                           fill_factor: float,
                           fill_factor_history: Dict[int, float],
                           histogram=Dict[int, int],
                           entropy=Dict[int, float]):
        """Updates last state of the Wang-Landau simulation.

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
        fill_factor
            fill factor of Wang-Landau algorithm
        fill_factor_history
            evolution of the fill factor of Wang-Landau algorithm (key=MC
            trial step, value=fill factor)
        histogram
            histogram of states visited during Wang-Landau simulation
        entropy
            (relative) entropy accumulated during Wang-Landau simulation
        """
        super()._update_last_state(
            last_step=last_step,
            occupations=occupations,
            accepted_trials=accepted_trials,
            random_state=random_state)
        self._last_state['fill_factor'] = fill_factor
        self._last_state['fill_factor_history'] = fill_factor_history
        self._last_state['histogram'] = histogram
        self._last_state['entropy'] = entropy

    @property
    def fill_factor(self) -> float:
        """ final value of the fill factor in the Wang-Landau algorithm """
        return self._last_state['fill_factor']

    @property
    def fill_factor_history(self) -> DataFrame:
        """ evolution of the fill factor in the Wang-Landau algorithm """
        return DataFrame({'mctrial': list(self._last_state['fill_factor_history'].keys()),
                          'fill_factor': list(self._last_state['fill_factor_history'].values())})

    def get_entropy(self) -> DataFrame:
        """Returns the (relative) entropy from this data container accumulated
        during a :ref:`Wang-Landau simulation <wang_landau_ensemble>`. Returns
        ``None`` if the data container does not contain the required information.
        """

        if 'entropy' not in self._last_state:
            return None

        # compile entropy into DataFrame
        entropy = self._last_state['entropy']
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


def get_density_of_states_wl(dcs: Union[BaseDataContainer, dict]) -> Tuple[DataFrame, dict]:
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
        data container(s), from which to extract the density of states

    Raises
    ------
    ValueError
        if multiple data containers are provided and there are inconsistencies
        with regard to basic simulation parameters such as system size or
        energy spacing
    ValueError
        if multiple data containers are provided and there is at least
        one energy region without overlap
    """

    # preparations
    if hasattr(dcs, 'get_entropy'):
        # fetch raw entropy data from data container
        df = dcs.get_entropy()
        errors = None
        if len(dcs.fill_factor_history) == 0 or dcs.fill_factor > 1e-4:
            warn('The data container appears to contain data from an'
                 ' underconverged Wang-Landau simulation.')

    elif isinstance(dcs, dict) and isinstance(dcs[next(iter(dcs))], BaseDataContainer):
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
            entropies[tag] = dc.get_entropy()

        # sort entropies by energy
        entropies = OrderedDict(sorted(entropies.items(), key=lambda row: row[1].energy.iloc[0]))

        # line up entropy data
        errors = {}
        tags = list(entropies.keys())
        for tag1, tag2 in zip(tags[:-1], tags[1:]):
            df1 = entropies[tag1]
            df2 = entropies[tag2]
            left_lim = np.min(df2.energy)
            right_lim = np.max(df1.energy)
            if left_lim >= right_lim:
                raise ValueError('No overlap in the energy range {}...{}.\n'
                                 .format(right_lim, left_lim) +
                                 ' The closest data containers have tags "{}" and "{}".'
                                 .format(tag1, tag2))
            df1_ = df1[(df1.energy >= left_lim) & (df1.energy <= right_lim)]
            df2_ = df2[(df2.energy >= left_lim) & (df2.energy <= right_lim)]
            offset = np.average(df2_.entropy - df1_.entropy)
            errors['{}-{}'.format(tag1, tag2)] = np.std(df2_.entropy - df1_.entropy)
            entropies[tag2].entropy = entropies[tag2].entropy - offset

        # compile entropy over the entire energy range
        data = {}
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
    df['density'] = np.exp(df.entropy) / np.sum(np.exp(df.entropy))

    return df, errors


def get_average_observables_wl(dcs: Union[BaseDataContainer, dict],
                               temperatures: List[float],
                               observables: List[str] = None,
                               boltzmann_constant: float = kB) -> DataFrame:
    """Returns the average and the standard deviation of the energy from a
    :ref:`Wang-Landau simulation <wang_landau_ensemble>` for the temperatures
    specified. If the ``observables`` keyword argument is specified
    the function will also return the mean and standard deviation of the
    specified observables.

    Parameters
    ----------
    dcs
        data container(s), from which to extract density of states
        as well as observables
    temperatures
        temperatures, at which to compute the averages
    observables
        observables, for which to compute averages; the observables
        must refer to fields in the data container
    boltzmann_constant
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e. units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K]

    Raises
    ------
    ValueError
        if the data container(s) do(es) not contain entropy data
        from Wang-Landau simulation
    ValueError
        if data container(s) do(es) not contain requested observable
    """

    def check_observables(dc: BaseDataContainer, observables: List[str]) -> None:
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
    if hasattr(dcs, 'get_entropy'):
        check_observables(dcs, observables)
        df_combined = dcs.data.filter(columns_to_keep)
        dcref = dcs
    elif isinstance(dcs, dict):
        for dc in dcs.values():
            check_observables(dc, observables)
        df_combined = pd_concat([dc.data for dc in dcs.values()],
                                ignore_index=True).filter(columns_to_keep)
        dcref = list(dcs.values())[0]
    else:
        raise TypeError('dcs ({}) must be a data container with entropy data'
                        ' or be a list of data containers'
                        .format(type(dcs)))

    # fetch entropy and density of states from data container(s)
    df_density, _ = get_density_of_states_wl(dcs)

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
    n_atoms = dcref.ensemble_parameters['n_atoms']
    for temperature in temperatures:

        # mean and standard deviation of energy
        boltz = np.exp(- (df_density.energy - enref) / temperature / boltzmann_constant)
        sumint = np.sum(df_density.density * boltz)
        en_mean = np.sum(df_density.energy / n_atoms * df_density.density * boltz) / sumint
        en_std = np.sum((df_density.energy / n_atoms) ** 2 * df_density.density * boltz) / sumint
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


def get_average_cluster_vectors_wl(dcs: Union[BaseDataContainer, dict],
                                   cluster_space: ClusterSpace,
                                   temperatures: List[float],
                                   boltzmann_constant: float = kB) -> DataFrame:
    """Returns the average cluster vectors from a :ref:`Wang-Landau simulation
    <wang_landau_ensemble>` for the temperatures specified.

    Parameters
    ----------
    dcs
        data container(s), from which to extract density of states
        as well as observables
    cluster_space
        cluster space to use for calculation of cluster vectors
    temperatures
        temperatures, at which to compute the averages
    boltzmann_constant
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e. units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K]
    """

    # fetch potential and structures
    if hasattr(dcs, 'get_entropy'):
        potential, trajectory = dcs.get_data('potential', 'trajectory')
        energy_spacing = dcs.ensemble_parameters['energy_spacing']
    elif isinstance(dcs, dict):
        potential, trajectory = [], []
        for dc in dcs.values():
            p, t = dc.get_data('potential', 'trajectory')
            potential.extend(p)
            trajectory.extend(t)
        energy_spacing = list(dcs.values())[0].ensemble_parameters['energy_spacing']
        potential = np.array(potential)
    else:
        raise TypeError('dcs ({}) must be a data container with entropy data'
                        ' or be a list of data containers'
                        .format(type(dcs)))

    # fetch entropy and density of states from data container(s)
    df_density, _ = get_density_of_states_wl(dcs)

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
