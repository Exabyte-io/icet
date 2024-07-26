from typing import Dict, List, Tuple
from collections.abc import Iterable

import pandas as pd

from ase import Atoms
from icet import ClusterSpace
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure
from icet.tools.geometry import chemical_symbols_to_numbers
from mchammer.observers.base_observer import BaseObserver


class ClusterCountObserver(BaseObserver):
    """This class represents a cluster count observer.

    A cluster count observer enables one to keep track of the
    occupation of clusters along the trajectory sampled by a Monte
    Carlo (MC) simulation. For example, given several canonical MC
    simulations representing different temperatures, this observer can
    be used to access the temperature dependence of the number of
    nearest neigbhors of a particular species.

    The outputted cluster counts come in a dictionary format which keys like
    ``0_Al``, ``0_Cu``, ``1_Al_Al``, ``1_Al_Cu``, ``1_Cu_Al``, ``1_Cu_Cu``, etc,
    where the number indicates the orbit index and the symbols indicate the elements for this orbit.
    The values of the dictionary contains the number of the such clusters found in the structure.

    Parameters
    ----------
    cluster_space
        Cluster space to define the clusters to be counted.
    structure
        Defines the lattice that the observer will work on.
    orbit_indices
        Only include orbits up to the orbit with this index.
        By default all orbits are included.
    interval
        Observation interval. Defaults to ``None`` meaning that if the
        observer is used in a Monte Carlo simulations, then the :class:`Ensemble` object
        will determine the interval.
    """

    def __init__(self, cluster_space: ClusterSpace,
                 structure: Atoms,
                 interval: int = None,
                 orbit_indices: List[int] = None) -> None:
        super().__init__(interval=interval, return_type=dict, tag='ClusterCountObserver')

        self._cluster_space = cluster_space
        local_orbit_list_generator = LocalOrbitListGenerator(
            orbit_list=cluster_space.orbit_list,
            structure=Structure.from_atoms(structure),
            fractional_position_tolerance=cluster_space.fractional_position_tolerance)

        self._full_orbit_list = local_orbit_list_generator.generate_full_orbit_list()

        if orbit_indices is None:
            self._orbit_indices = list(range(len(self._full_orbit_list)))
        elif not isinstance(orbit_indices, Iterable):
            raise ValueError('Argument orbit_indices should be a list of integers, '
                             f'not {type(orbit_indices)}')
        else:
            self._orbit_indices = orbit_indices

        self._possible_occupations = self._get_possible_occupations()

    def _get_possible_occupations(self) -> Dict[int, List[Tuple[str]]]:
        """ Returns a dictionary containing the possible occupations for each orbit. """
        possible_occupations = {}
        for i in self._orbit_indices:
            possible_occupations_orbit = self._cluster_space.get_possible_orbit_occupations(i)
            order = self._full_orbit_list.get_orbit(i).order
            assert order == len(possible_occupations_orbit[0]), \
                f'Order (n={order}) does not match possible occupations' \
                f' (n={len(possible_occupations[0])}, {possible_occupations}).'
            possible_occupations[i] = possible_occupations_orbit
        return possible_occupations

    def get_cluster_counts(self, structure: Atoms) -> pd.DataFrame:
        """Counts the number of times different clusters appear in the structure
        and returns this information as a pandas dataframe.

        Parameters
        ----------
        structure
            input atomic structure.
        """
        rows = []
        structure_icet = Structure.from_atoms(structure)
        for i in self._orbit_indices:
            orbit = self._full_orbit_list.get_orbit(i)
            cluster_counts = orbit.get_cluster_counts(structure_icet)
            for chemical_symbols in self._possible_occupations[i]:
                count = cluster_counts.get(tuple(chemical_symbols_to_numbers(chemical_symbols)), 0)
                row = {}
                row['dc_tag'] = '{}_{}'.format(i, '_'.join(chemical_symbols))
                row['occupation'] = chemical_symbols
                row['cluster_count'] = count
                row['orbit_index'] = i
                row['order'] = orbit.order
                rows.append(row)
        return pd.DataFrame(rows)

    def get_observable(self, structure: Atoms) -> dict:
        """
        Returns the value of the property from a cluster expansion model
        for a given atomic configuration.

        Parameters
        ----------
        structure
            input atomic structure
        """
        counts = self.get_cluster_counts(structure)
        count_dict = {row['dc_tag']: row['cluster_count']
                      for i, row in counts.iterrows()}
        return count_dict
