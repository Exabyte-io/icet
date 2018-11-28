import pandas as pd
import numpy as np
import pickle
import re

from icet import ClusterSpace
from icet.core.structure import Structure
from typing import List, Union
from ase import Atoms


class ClusterExpansion:
    """
    Cluster expansion model

    Attributes
    ----------
    cluster_space : ClusterSpace object
        cluster space that was used for constructing the cluster expansion
    parameters : list of floats
        effective cluster interactions (ECIs)
    """

    def __init__(self, cluster_space: ClusterSpace,
                 parameters: List[float]) -> None:
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
        self._parameters = parameters

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
        """ pandas dataframe containing orbit data and ECIs """
        rows = self.cluster_space.orbit_data
        for row, eci in zip(rows, self.parameters):
            row['eci'] = eci
        return pd.DataFrame(rows)

    @property
    def orders(self) -> List:
        return list(range(len(self.cluster_space.cutoffs)+2))

    @property
    def cluster_space(self) -> ClusterSpace:
        """ cluster space on which the cluster expansion is based """
        return self._cluster_space

    @property
    def parameters(self) -> List[float]:
        """ effective cluster interactions (ECIs) """
        return self._parameters

    def __len__(self) -> int:
        return len(self._parameters)

    def _get_string_representation(self, print_threshold: int = None,
                                   print_minimum: int = 10):
        """
        String representation of the cluster expansion.
        """
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

        with open(filename, "wb") as handle:
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

        return ClusterExpansion(cs, parameters)
