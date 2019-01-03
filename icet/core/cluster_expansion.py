import pandas as pd
import numpy as np
import pickle
import re

from icet import ClusterSpace
from icet.core.structure import Structure
from typing import List, Union
from ase import Atoms


class ClusterExpansion:
    """Cluster expansions are obtained by combining a cluster space with
    a set of effective cluster interactions (ECIs). Instances of this
    class allow one to predict the property of interest for a given
    structure.

    Attributes
    ----------
    cluster_space : icet.ClusterSpace
        cluster space that was used for constructing the cluster expansion
    parameters : np.ndarray
        effective cluster interactions (ECIs)

    Example
    -------
    The following snippet illustrates the initialization and usage of
    a ClusterExpansion object. Here, the ECIs are taken to be a list
    of ones. Usually, they would be obtained by training with
    respect to a set of reference data::

        from ase.build import bulk
        from icet import ClusterSpace, ClusterExpansion

        # create cluster expansion
        prim = bulk('Au')
        cs = ClusterSpace(prim, cutoffs=[7.0, 5.0],
                          chemical_symbols=[['Au', 'Pd']])
        ecis = 14 * [1.0]
        ce = ClusterExpansion(cs, ecis)

        # make prediction for supercell
        sc = prim.repeat(3)
        for k in [1, 4, 7]:
            sc[k].symbol = 'Pd'
        print(ce.predict(sc))
    """

    def __init__(self, cluster_space: ClusterSpace,
                 parameters: np.array) -> None:
        """
        Initializes a ClusterExpansion object.

        Parameters
        ----------
        cluster_space
            cluster space to be used for constructing the cluster expansion
        parameters
            effective cluster interactions (ECIs)

        Raises
        ------
        ValueError
            if cluster space and parameters differ in length
        """
        if len(cluster_space) != len(parameters):
            raise ValueError('cluster_space ({}) and parameters ({}) must have'
                             ' the same length'.format(len(cluster_space),
                                                       len(parameters)))
        self._cluster_space = cluster_space
        if isinstance(parameters, list):
            parameters = np.array(parameters)
        self._parameters = parameters
        self._original_parameters = parameters.copy()
        self._pruning_history = []

    def predict(self, structure: Union[Atoms, Structure]) -> float:
        """
        Predicts the property of interest (e.g., the energy) for the input
        structure using the cluster expansion.

        Parameters
        ----------
        structure
            atomic configuration

        Returns
        -------
        float
            property value of predicted by the cluster expansion
        """
        cluster_vector = self.cluster_space.get_cluster_vector(structure)
        prop = np.dot(cluster_vector, self.parameters)
        return prop

    @property
    def parameters_as_dataframe(self) -> pd.DataFrame:
        """ dataframe containing orbit data and ECIs """
        rows = self.cluster_space.orbit_data
        for row, eci in zip(rows, self.parameters):
            row['eci'] = eci
        return pd.DataFrame(rows)

    @property
    def orders(self) -> List[int]:
        """ orders included in cluster expansion """
        return list(range(len(self.cluster_space.cutoffs)+2))

    @property
    def cluster_space(self) -> ClusterSpace:
        """ cluster space on which cluster expansion is based """
        return self._cluster_space

    @property
    def parameters(self) -> List[float]:
        """ effective cluster interactions (ECIs) """
        return self._parameters

    def __len__(self) -> int:
        return len(self._parameters)

    def _get_string_representation(self, print_threshold: int = None,
                                   print_minimum: int = 10):
        """ String representation of the cluster expansion. """
        cluster_space_repr = self._cluster_space._get_string_representation(
            print_threshold, print_minimum).split('\n')
        # rescale width
        eci_col_width = max(
            len('{:9.3g}'.format(max(self._parameters, key=abs))), len('ECI'))
        width = len(cluster_space_repr[0]) + len(' | ') + eci_col_width

        s = []  # type: List
        s += ['{s:=^{n}}'.format(s=' Cluster Expansion ', n=width)]
        s += [t for t in cluster_space_repr if re.search(':', t)]

        # table header
        s += [''.center(width, '-')]
        t = [t for t in cluster_space_repr if 'index' in t]
        t += ['{s:^{n}}'.format(s='ECI', n=eci_col_width)]
        s += [' | '.join(t)]
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
            pattern = r'^{:4}'.format(index)
            t = [t for t in cluster_space_repr if re.match(pattern, t)]
            eci_value = '{:9.3g}'.format(self._parameters[index])
            t += ['{s:^{n}}'.format(s=eci_value, n=eci_col_width)]
            s += [' | '.join(t)]
            index += 1
        s += [''.center(width, '=')]

        return '\n'.join(s)

    def __repr__(self) -> str:
        """ string representation """
        return self._get_string_representation(print_threshold=50)

    def prune(self, indices: List[int] = None, tol: float = 0):
        """
        Removes orbits from the cluster expansion (CE), for which the effective
        cluster interactions (ECIs; parameters) are zero or close to zero.
        This commonly reduces the computational cost for evaluating the CE and
        is therefore recommended prior to using it in production. If the method
        is called without arguments orbits will be pruned, for which the ECIs
        are strictly zero. Less restrictive pruning can be achived by setting
        the `tol` keyword.

        Parameters
        ----------
        indices
            indices to parameters to remove in the cluster expansion.
        tol
            orbits for which the absolute ECIs is/are within this
            value will be pruned
        """
        self._pruning_history.append({'indices': indices, 'tol': tol})

        if indices is None:
            indices = [i for i, param in enumerate(
                self.parameters) if np.abs(param) <= tol and i > 0]
        df = self.parameters_as_dataframe
        indices = list(set(indices))

        if 0 in indices:
            raise ValueError('Orbit index cannot be 0 since'
                             ' the zerolet may not be pruned.')
        orbit_candidates_for_removal = \
            df.orbit_index[np.array(indices)].tolist()
        safe_to_remove_orbits, safe_to_remove_params = [], []
        for oi in set(orbit_candidates_for_removal):
            if oi == -1:
                continue
            orbit_count = df.orbit_index.tolist().count(oi)
            oi_remove_count = orbit_candidates_for_removal.count(oi)
            if orbit_count <= oi_remove_count:
                safe_to_remove_orbits.append(oi)
                safe_to_remove_params += df.index[df['orbit_index']
                                                  == oi].tolist()

        self._cluster_space._prune_orbit_list(indices=safe_to_remove_orbits)
        self._parameters = self._parameters[np.setdiff1d(
            np.arange(len(self._parameters)), safe_to_remove_params)]

    def write(self, filename: str):
        """
        Writes ClusterExpansion object to file.

        Parameters
        ---------
        filename
            name of file to which to write
        """
        self.cluster_space.write(filename)

        with open(filename, 'rb') as handle:
            data = pickle.load(handle)

        data['parameters'] = self.parameters
        data['original_parameters'] = self._original_parameters
        data['pruning_history'] = self._pruning_history

        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def read(filename: str):
        """
        Reads ClusterExpansion object from file.

        Parameters
        ---------
        filename
            file from which to read
        """
        cs = ClusterSpace.read(filename)
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        parameters = data['parameters']
        pruning_history = data['pruning_history']
        original_parameters = data['original_parameters']
        ce = ClusterExpansion(cs, original_parameters)
        for arg in pruning_history:
            ce.prune(indices=arg['indices'], tol=arg['tol'])

        assert list(parameters) == list(ce.parameters)
        return ce
