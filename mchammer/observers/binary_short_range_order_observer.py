from collections import namedtuple
from typing import Dict

import numpy as np

from ase import Atoms
from icet import ClusterSpace
from mchammer.observers import ClusterCountObserver
from mchammer.observers.base_observer import BaseObserver

ClusterCountInfo = namedtuple('ClusterCountInfo', ['counts', 'dc_tags'])


class BinaryShortRangeOrderObserver(BaseObserver):
    """
    This class represents a short range order (SRO) observer for a
    binary system.


    Parameters
    ----------
    cluster_space : icet.ClusterSpace
        cluster space usedfor initialization
    structure : ase.Atoms
        defines the lattice which the observer will work on
    interval : int
        observation interval during the Monte Carlo simulation
    radius : float
        the maximum radius  for the neigbhor shells considered

    Attributes
    ----------
    tag : str
        human readable observer name (`BinaryShortRangeOrderObserver`)
    interval : int
        observation interval
    """

    def __init__(self, cluster_space, structure: Atoms,
                 interval: int, radius: float) -> None:
        super().__init__(interval=interval, return_type=dict,
                         tag='BinaryShortRangeOrderObserver')

        self._structure = structure

        self._cluster_space = ClusterSpace(
            atoms=cluster_space.primitive_structure,
            cutoffs=[radius],
            chemical_symbols=cluster_space.chemical_symbols)
        self._cluster_count_observer = ClusterCountObserver(
            cluster_space=self._cluster_space, atoms=structure,
            interval=interval)

        self._sublattices = self._cluster_space.get_sublattices(structure)
        binary_sublattice_counts = 0
        for symbols in self._sublattices.allowed_species:
            if len(symbols) == 2:
                binary_sublattice_counts += 1
                self._symbols = sorted(symbols)
            elif len(symbols) > 2:
                raise ValueError('Cluster space has more than two allowed'
                                 ' species on a sublattice. '
                                 'Allowed species: {}'.format(symbols))
        if binary_sublattice_counts != 1:
            raise ValueError('Number of binary sublattices must equal one,'
                             ' not {}'.format(binary_sublattice_counts))

    def get_observable(self, atoms: Atoms) -> Dict[str, float]:
        """Returns the value of the property from a cluster expansion
        model for a given atomic configurations.

        Parameters
        ----------
        atoms
            input atomic structure
        """

        self._cluster_count_observer._generate_counts(atoms)
        df = self._cluster_count_observer.count_frame

        symbol_counts = self._get_atom_count(atoms)
        conc_B = self._get_concentrations(atoms)[self._symbols[0]]

        pair_orbit_indices = set(
            df.loc[df['order'] == 2]['orbit_index'].tolist())
        N = symbol_counts[self._symbols[0]] + symbol_counts[self._symbols[1]]
        sro_parameters = {}
        for k, orbit_index in enumerate(sorted(pair_orbit_indices)):
            orbit_df = df.loc[df['orbit_index'] == orbit_index]
            A_B_pair_count = 0
            total_count = 0
            total_A_count = 0
            for i, row in orbit_df.iterrows():
                total_count += row.cluster_count
                if self._symbols[0] in row.decoration:
                    total_A_count += row.cluster_count
                if self._symbols[0] in row.decoration and \
                        self._symbols[1] in row.decoration:
                    A_B_pair_count += row.cluster_count

            key = 'sro_{}_{}'.format(self._symbols[0], k+1)
            Z_tot = symbol_counts[self._symbols[0]] * 2 * total_count / N
            if conc_B == 1 or Z_tot == 0:
                value = 0
            else:
                value = 1 - A_B_pair_count/(Z_tot * (1-conc_B))
            sro_parameters[key] = value

        return sro_parameters

    def _get_concentrations(self, structure: Atoms) -> Dict[str, float]:
        """Returns concentrations for each species relative its
        sublattice.

        Parameters
        ----------
        structure
            the configuration that will be analyzed
        """
        decoration = np.array(structure.get_chemical_symbols())
        concentrations = {}
        for sublattice in self._sublattices:
            if len(sublattice.chemical_symbols) == 1:
                continue
            for symbol in sublattice.chemical_symbols:
                symbol_count = decoration[sublattice.indices].tolist().count(
                    symbol)
                concentration = symbol_count / len(sublattice.indices)
                concentrations[symbol] = concentration
        return concentrations

    def _get_atom_count(self, structure: Atoms) -> Dict[str, float]:
        """Returns atom counts for each species relative its
        sublattice.

        Parameters
        ----------
        structure
            the configuration that will be analyzed
        """
        decoration = np.array(structure.get_chemical_symbols())
        counts = {}
        for sublattice in self._sublattices:
            if len(sublattice.chemical_symbols) == 1:
                continue
            for symbol in sublattice.chemical_symbols:
                symbol_count = decoration[sublattice.indices].tolist().count(
                    symbol)
                counts[symbol] = symbol_count
        return counts