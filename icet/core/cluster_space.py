"""
This module provides the :class:`ClusterSpace` class.
"""

import os
import copy
import itertools
import pickle
import tarfile
import tempfile
from collections.abc import Iterable
from math import log10, floor
from typing import Dict, List, Union, Tuple

import numpy as np
import spglib

from _icet import ClusterSpace as _ClusterSpace
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from icet.core.orbit_list import OrbitList
from icet.core.structure import Structure
from icet.core.sublattices import Sublattices
from icet.tools.geometry import (ase_atoms_to_spglib_cell,
                                 get_occupied_primitive_structure)
from pandas import DataFrame


class ClusterSpace(_ClusterSpace):
    """This class provides functionality for generating and maintaining
    cluster spaces.

    Note
    ----
    In :program:`icet` all :class:`Atoms <ase.Atoms>` objects must have
    periodic boundary conditions. When constructing cluster expansions
    for surfaces and nanoparticles it is therefore recommended to
    surround the structure with vacuum and use periodic boundary
    conditions. This can be achieved by using :func:`Atoms.center <ase.Atoms.center>`.

    Parameters
    ----------
    structure
        Atomic structure.
    cutoffs
        Cutoff radii per order that define the cluster space.

        Cutoffs are specified in units of Ångstrom and refer to the
        longest distance between two atoms in the cluster. The first
        element refers to pairs, the second to triplets, the third
        to quadruplets, and so on. :attr:`cutoffs=[7.0, 4.5]` thus implies
        that all pairs distanced 7 Å or less will be included,
        as well as all triplets among which the longest distance is no
        longer than 4.5 Å.
    chemical_symbols
        List of chemical symbols, each of which must map to an element
        of the periodic table.

        If a list of chemical symbols is provided, all sites on the
        lattice will have the same allowed occupations as the input
        list.

        If a list of list of chemical symbols is provided then the
        outer list must be the same length as the :attr:`structure` object and
        :attr:`chemical_symbols[i]` will correspond to the allowed species
        on lattice site ``i``.
    symprec
        Tolerance imposed when analyzing the symmetry using spglib.
    position_tolerance
        Tolerance applied when comparing positions in Cartesian coordinates.

    Examples
    --------
    The following snippets illustrate several common situations::

        >>> from ase.build import bulk
        >>> from ase.io import read
        >>> from icet import ClusterSpace

        >>> # AgPd alloy with pairs up to 7.0 A and triplets up to 4.5 A
        >>> prim = bulk('Ag')
        >>> cs = ClusterSpace(structure=prim, cutoffs=[7.0, 4.5],
        ...                   chemical_symbols=[['Ag', 'Pd']])
        >>> print(cs)

        >>> # (Mg,Zn)O alloy on rocksalt lattice with pairs up to 8.0 A
        >>> prim = bulk('MgO', crystalstructure='rocksalt', a=6.0)
        >>> cs = ClusterSpace(structure=prim, cutoffs=[8.0],
        ...                   chemical_symbols=[['Mg', 'Zn'], ['O']])
        >>> print(cs)

        >>> # (Ga,Al)(As,Sb) alloy with pairs, triplets, and quadruplets
        >>> prim = bulk('GaAs', crystalstructure='zincblende', a=6.5)
        >>> cs = ClusterSpace(structure=prim, cutoffs=[7.0, 6.0, 5.0],
        ...                   chemical_symbols=[['Ga', 'Al'], ['As', 'Sb']])
        >>> print(cs)

        >>> # PdCuAu alloy with pairs and triplets
        >>> prim = bulk('Pd')
        >>> cs = ClusterSpace(structure=prim, cutoffs=[7.0, 5.0],
        ...                   chemical_symbols=[['Au', 'Cu', 'Pd']])
        >>> print(cs)

    """

    def __init__(self,
                 structure: Atoms,
                 cutoffs: List[float],
                 chemical_symbols: Union[List[str], List[List[str]]],
                 symprec: float = 1e-5,
                 position_tolerance: float = None) -> None:

        if not isinstance(structure, Atoms):
            raise TypeError('Input configuration must be an ASE Atoms object'
                            f', not type {type(structure)}.')
        if not all(structure.pbc):
            raise ValueError('Input structure must be periodic.')
        if symprec <= 0:
            raise ValueError('symprec must be a positive number.')

        self._config = {'symprec': symprec}
        self._cutoffs = cutoffs.copy()
        self._input_structure = structure.copy()
        self._input_chemical_symbols = copy.deepcopy(chemical_symbols)
        chemical_symbols = self._get_chemical_symbols()

        self._pruning_history: List[tuple] = []

        # set up primitive
        occupied_primitive, primitive_chemical_symbols = get_occupied_primitive_structure(
            self._input_structure, chemical_symbols, symprec=self.symprec)
        self._primitive_chemical_symbols = primitive_chemical_symbols
        assert len(occupied_primitive) == len(primitive_chemical_symbols)

        # derived tolerances
        if position_tolerance is None:
            self._config['position_tolerance'] = symprec
        else:
            if position_tolerance <= 0:
                raise ValueError('position_tolerance must be a positive number')
            self._config['position_tolerance'] = position_tolerance
        effective_box_size = abs(np.linalg.det(occupied_primitive.cell)) ** (1 / 3)
        tol = self.position_tolerance / effective_box_size
        tol = min(tol, self._config['position_tolerance'] / 5)
        self._config['fractional_position_tolerance'] = round(tol, -int(floor(log10(abs(tol)))))

        # set up orbit list
        self._orbit_list = OrbitList(
            structure=occupied_primitive,
            cutoffs=self._cutoffs,
            chemical_symbols=self._primitive_chemical_symbols,
            symprec=self.symprec,
            position_tolerance=self.position_tolerance,
            fractional_position_tolerance=self.fractional_position_tolerance)
        self._orbit_list.remove_orbits_with_inactive_sites()

        # call (base) C++ constructor
        _ClusterSpace.__init__(self,
                               orbit_list=self._orbit_list,
                               position_tolerance=self.position_tolerance,
                               fractional_position_tolerance=self.fractional_position_tolerance)

    def _get_chemical_symbols(self):
        """ Returns chemical symbols using input structure and
        chemical symbols. Carries out multiple sanity checks. """

        # setup chemical symbols as List[List[str]]
        if all(isinstance(i, str) for i in self._input_chemical_symbols):
            chemical_symbols = [self._input_chemical_symbols] * len(self._input_structure)
        # also accept tuples and other iterables but not, e.g., List[List, str]
        # (need to check for str explicitly here because str is an Iterable)
        elif not all(isinstance(i, Iterable) and not isinstance(i, str)
                     for i in self._input_chemical_symbols):
            raise TypeError('chemical_symbols must be List[str] or List[List[str]], not {}'.format(
                type(self._input_chemical_symbols)))
        elif len(self._input_chemical_symbols) != len(self._input_structure):
            msg = 'chemical_symbols must have same length as structure. '
            msg += 'len(chemical_symbols) = {}, len(structure)= {}'.format(
                len(self._input_chemical_symbols), len(self._input_structure))
            raise ValueError(msg)
        else:
            chemical_symbols = copy.deepcopy(self._input_chemical_symbols)

        for i, symbols in enumerate(chemical_symbols):
            if len(symbols) != len(set(symbols)):
                raise ValueError(
                    'Found duplicates of allowed chemical symbols on site {}.'
                    ' allowed species on  site {}= {}'.format(i, i, symbols))

        if len([tuple(sorted(s)) for s in chemical_symbols if len(s) > 1]) == 0:
            raise ValueError('No active sites found')

        return chemical_symbols

    def _get_chemical_symbol_representation(self):
        """Returns a str version of the chemical symbols that is
        easier on the eyes.
        """
        sublattices = self.get_sublattices(self.primitive_structure)
        nice_str = []
        for sublattice in sublattices.active_sublattices:
            sublattice_symbol = sublattice.symbol
            nice_str.append('{} (sublattice {})'.format(
                list(sublattice.chemical_symbols), sublattice_symbol))
        return ', '.join(nice_str)

    def _get_string_representation(self,
                                   print_threshold: int = None,
                                   print_minimum: int = 10) -> str:
        """
        String representation of the cluster space that provides an overview of
        the orbits (order, radius, multiplicity etc) that constitute the space.

        Parameters
        ----------
        print_threshold
            if the number of orbits exceeds this number print dots
        print_minimum
            number of lines printed from the top and the bottom of the orbit
            list if `print_threshold` is exceeded

        Returns
        -------
        multi-line string
            string representation of the cluster space.
        """

        def repr_orbit(orbit, header=False):
            formats = {'order': '{:2}',
                       'radius': '{:8.4f}',
                       'multiplicity': '{:4}',
                       'index': '{:4}',
                       'orbit_index': '{:4}',
                       'multicomponent_vector': '{:}',
                       'sublattices': '{:}'}
            s = []
            for name, value in orbit.items():
                if name == 'sublattices':
                    str_repr = formats[name].format('-'.join(value))
                else:
                    str_repr = formats[name].format(value)
                n = max(len(name), len(str_repr))
                if header:
                    s += ['{s:^{n}}'.format(s=name, n=n)]
                else:
                    s += ['{s:^{n}}'.format(s=str_repr, n=n)]
            return ' | '.join(s)

        # basic information
        # (use largest orbit to obtain maximum line length)
        prototype_orbit = self.as_list[-1]
        width = len(repr_orbit(prototype_orbit))
        s = []
        s += ['{s:=^{n}}'.format(s=' Cluster Space ', n=width)]
        s += [' {:38} : {}'.format('space group', self.space_group)]
        s += [' {:38} : {}'
              .format('chemical species', self._get_chemical_symbol_representation())]
        s += [' {:38} : {}'.format('cutoffs',
                                   ' '.join(['{:.4f}'.format(c) for c in self.cutoffs]))]
        s += [' {:38} : {}'.format('total number of parameters', len(self))]
        t = ['{}= {}'.format(k, c)
             for k, c in self.number_of_orbits_by_order.items()]
        s += [' {:38} : {}'.format('number of parameters by order', '  '.join(t))]
        for key, value in sorted(self._config.items()):
            s += [' {:38} : {}'.format(key, value)]

        # table header
        s += [''.center(width, '-')]
        s += [repr_orbit(prototype_orbit, header=True)]
        s += [''.center(width, '-')]

        # table body
        index = 0
        orbit_list_info = self.as_list
        while index < len(orbit_list_info):
            if (print_threshold is not None and
                    len(self) > print_threshold and
                    index >= print_minimum and
                    index <= len(self) - print_minimum):
                index = len(self) - print_minimum
                s += [' ...']
            s += [repr_orbit(orbit_list_info[index])]
            index += 1
        s += [''.center(width, '=')]

        return '\n'.join(s)

    def __str__(self) -> str:
        """ String representation. """
        return self._get_string_representation(print_threshold=50)

    def _repr_html_(self) -> str:
        """ HTML representation. Used, e.g., in jupyter notebooks. """
        s = ['<h4>Cluster Space</h4>']
        s += ['<table border="1" class="dataframe">']
        s += ['<thead><tr><th style="text-align: left;">Field</th><th>Value</th></tr></thead>']
        s += ['<tbody>']
        s += [f'<tr><td style="text-align: left;">Space group</td><td>{self.space_group}</td></tr>']
        for sl in self.get_sublattices(self.primitive_structure).active_sublattices:
            s += [f'<tr><td style="text-align: left;">Sublattice {sl.symbol}</td>'
                  f'<td>{sl.chemical_symbols}</td></tr>']
        s += [f'<tr><td style="text-align: left;">Cutoffs</td><td>{self.cutoffs}</td></tr>']
        s += ['<tr><td style="text-align: left;">Total number of parameters</td>'
              f'<td>{len(self)}</td></tr>']
        for k, n in self.number_of_orbits_by_order.items():
            s += [f'<tr><td style="text-align: left;">Number of parameters of order {k}</td>'
                  f'<td>{n}</td></tr>']
        for key, value in sorted(self._config.items()):
            s += [f'<tr><td style="text-align: left;">{key}</td><td>{value}</td></tr>']
        s += ['</tbody>']
        s += ['</table>']
        return ''.join(s)

    def __repr__(self) -> str:
        """ Representation. """
        s = type(self).__name__ + '('
        s += f'structure={self.primitive_structure.__repr__()}'
        s += f', cutoffs={self._cutoffs.__repr__()}'
        s += f', chemical_symbols={self._input_chemical_symbols.__repr__()}'
        s += f', position_tolerance={self._config["position_tolerance"]}'
        s += ')'
        return s

    def __getitem__(self, ind: int):
        return self.as_list[ind]

    @property
    def symprec(self) -> float:
        """ Tolerance imposed when analyzing the symmetry using spglib. """
        return self._config['symprec']

    @property
    def position_tolerance(self) -> float:
        """ Tolerance applied when comparing positions in Cartesian coordinates. """
        return self._config['position_tolerance']

    @property
    def fractional_position_tolerance(self) -> float:
        """ Tolerance applied when comparing positions in fractional coordinates. """
        return self._config['fractional_position_tolerance']

    @property
    def space_group(self) -> str:
        """ Space group of the primitive structure in international notion (via spglib). """
        structure_as_tuple = ase_atoms_to_spglib_cell(self.primitive_structure)
        return spglib.get_spacegroup(structure_as_tuple, symprec=self._config['symprec'])

    @property
    def as_list(self) -> List[dict]:
        """Representation of cluster space as list with information regarding
        order, radius, multiplicity etc.
        """
        data = []
        zerolet = dict(
            index=0,
            order=0,
            radius=0,
            multiplicity=1,
            orbit_index=-1,
            multicomponent_vector='.',
            sublattices='.',
        )
        data.append(zerolet)

        sublattices = self.get_sublattices(self.primitive_structure)
        index = 0
        for orbit_index in range(len(self.orbit_list)):
            orbit = self.orbit_list.get_orbit(orbit_index)
            representative_cluster = orbit.representative_cluster
            orbit_sublattices = [
                sublattices[sublattices.get_sublattice_index_from_site_index(ls.index)].symbol
                for ls in representative_cluster.lattice_sites]
            for cv_element in orbit.cluster_vector_elements:
                index += 1
                data.append(dict(
                    index=index,
                    order=representative_cluster.order,
                    radius=representative_cluster.radius,
                    multiplicity=cv_element['multiplicity'],
                    orbit_index=orbit_index,
                    multicomponent_vector=cv_element['multicomponent_vector'],
                    sublattices=orbit_sublattices
                ))
        return data

    def to_dataframe(self) -> DataFrame:
        """ Returns a representation of the cluster space as a DataFrame. """
        df = DataFrame.from_dict(self.as_list)
        del df['index']
        return df

    @property
    def number_of_orbits_by_order(self) -> dict:
        """ Number of orbits by order in the form of a dictionary
        where keys and values represent order and number of orbits,
        respectively.
        """
        count_orbits: Dict[int, int] = {}
        for orbit in self.as_list:
            k = orbit['order']
            count_orbits[k] = count_orbits.get(k, 0) + 1
        return dict(sorted(count_orbits.items()))

    def get_cluster_vector(self, structure: Atoms) -> np.ndarray:
        """
        Returns the cluster vector for a structure.

        Parameters
        ----------
        structure
            Atomic configuration.
        """
        if not isinstance(structure, Atoms):
            raise TypeError('Input structure must be an ASE Atoms object')

        try:
            cv = _ClusterSpace.get_cluster_vector(
                self,
                structure=Structure.from_atoms(structure),
                fractional_position_tolerance=self.fractional_position_tolerance)
        except Exception as e:
            self.assert_structure_compatibility(structure)
            raise Exception(str(e))
        return cv

    def get_coordinates_of_representative_cluster(self, orbit_index: int) -> List[Tuple[float]]:
        """
        Returns the positions of the sites in the representative cluster of the selected orbit.

        Parameters
        ----------
        orbit_index
            Index of the orbit for which to return the positions of the sites.
        """
        # Raise exception if chosen orbit index not in current list of orbit indices
        if orbit_index not in range(len(self._orbit_list)):
            raise ValueError('The input orbit index is not in the list of possible values.')
        return self._orbit_list.get_orbit(orbit_index).representative_cluster.positions

    def _remove_orbits(self, indices: List[int]) -> None:
        """
        Removes orbits.

        Parameters
        ----------
        indices
            Indices to all orbits to be removed.
        """
        size_before = len(self._orbit_list)

        # Since we remove orbits, orbit indices will change,
        # so we run over the orbits in reverse order.
        for ind in reversed(sorted(indices)):
            self._orbit_list.remove_orbit(ind)

        size_after = len(self._orbit_list)
        assert size_before - len(indices) == size_after

    def prune_orbit_list(self, indices: List[int]) -> None:
        """
        Prunes the internal orbit list and maintains the history.

        Parameters
        ----------
        indices
            Indices to all orbits to be removed.
        """
        self._remove_orbits(indices)
        self._pruning_history.append(('prune', indices))

    @property
    def primitive_structure(self) -> Atoms:
        """ Primitive structure on which cluster space is based. """
        structure = self._get_primitive_structure().to_atoms()
        # Decorate with the "real" symbols (instead of H, He, Li etc)
        for atom, symbols in zip(structure, self._primitive_chemical_symbols):
            atom.symbol = min(symbols)
        return structure

    @property
    def chemical_symbols(self) -> List[List[str]]:
        """ Species identified by their chemical symbols. """
        return self._primitive_chemical_symbols.copy()

    @property
    def cutoffs(self) -> List[float]:
        """
        Cutoffs for different n-body clusters. The cutoff radius (in
        Ångstroms) defines the largest interatomic distance in a
        cluster.
        """
        return self._cutoffs

    @property
    def orbit_list(self):
        """ Orbit list that defines the cluster in the cluster space. """
        return self._orbit_list

    def get_possible_orbit_occupations(self, orbit_index: int) -> List[List[str]]:
        """ Returns possible occupations of the orbit.

        Parameters
        ----------
        orbit_index
            Index of orbit of interest.
        """
        orbit = self.orbit_list.orbits[orbit_index]
        indices = [ls.index for ls in orbit.representative_cluster.lattice_sites]
        allowed_species = [self.chemical_symbols[index] for index in indices]
        return list(itertools.product(*allowed_species))

    def get_sublattices(self, structure: Atoms) -> Sublattices:
        """ Returns the sublattices of the input structure.

        Parameters
        ----------
        structure
            Atomic structure the sublattices are based on.
        """
        sl = Sublattices(self.chemical_symbols,
                         self.primitive_structure,
                         structure,
                         fractional_position_tolerance=self.fractional_position_tolerance)
        return sl

    def assert_structure_compatibility(self, structure: Atoms, vol_tol: float = 1e-5) -> None:
        """ Raises error if structure is not compatible with this cluster space.

        Parameters
        ----------
        structure
            Structure to check for compatibility with cluster space.
        vol_tol
            Tolerance imposed when comparing volumes.
        """
        # check volume
        vol1 = self.primitive_structure.get_volume() / len(self.primitive_structure)
        vol2 = structure.get_volume() / len(structure)
        if abs(vol1 - vol2) > vol_tol:
            raise ValueError(f'Volume per atom of structure ({vol1}) does not match the volume of'
                             f' the primitive structure ({vol2}; vol_tol= {vol_tol}).')

        # check occupations
        sublattices = self.get_sublattices(structure)
        sublattices.assert_occupation_is_allowed(structure.get_chemical_symbols())

        # check pbc
        if not all(structure.pbc):
            raise ValueError('Input structure must be periodic.')

    def merge_orbits(self,
                     equivalent_orbits: Dict[int, List[int]],
                     ignore_permutations: bool = False) -> None:
        """ Combines several orbits into one. This allows one to make custom
        cluster spaces by manually declaring the clusters in two or more
        orbits to be equivalent. This is a powerful approach for simplifying
        the cluster spaces of low-dimensional structures such as
        surfaces or nanoparticles.

        The procedure works in principle for any number of components. Note,
        however, that in the case of more than two components the outcome of
        the merging procedure inherits the treatment of the multi-component
        vectors of the orbit chosen as the representative one.

        Parameters
        ----------
        equivalent_orbits
            The keys of this dictionary denote the indices of the orbit into
            which to merge. The values are the indices of the orbits that are
            supposed to be merged into the orbit denoted by the key.
        ignore_permutations
            If ``True`` orbits will be merged even if their multi-component
            vectors and/or site permutations differ. While the object will
            still be functional, the cluster space may not be properly spanned
            by the resulting cluster vectors.

        Note
        ----
        The orbit index should not be confused with the index shown when
        printing the cluster space.

        Examples
        --------
        The following snippet illustrates the use of this method to create a
        cluster space for a (111) FCC surface, in which only the singlets for
        the first and second layer are distinct as well as the in-plane pair
        interaction in the topmost layer. All other singlets and pairs are
        respectively merged into one orbit. After merging there aree only 3
        singlets and 2 pairs left with correspondingly higher multiplicities.

            >>> from icet import ClusterSpace
            >>> from ase.build import fcc111
            >>>
            >>> # Create primitive surface unit cell
            >>> structure = fcc111('Au', size=(1, 1, 8), a=4.1, vacuum=10, periodic=True)
            >>>
            >>> # Set up initial cluster space
            >>> cs = ClusterSpace(structure=structure, cutoffs=[3.8], chemical_symbols=['Au', 'Ag'])
            >>>
            >>> # At this point, one can inspect the orbits in the cluster space by printing the
            >>> # ClusterSpace object and accessing the individial orbits.
            >>> # There will be 4 singlets and 8 pairs.
            >>>
            >>> # Merge singlets for the third and fourth layers as well as all pairs except for
            >>> # the one corresponding to the in-plane interaction in the topmost surface
            >>> # layer.
            >>> cs.merge_orbits({2: [3], 4: [6, 7, 8, 9, 10, 11]})
        """

        self._pruning_history.append(('merge', equivalent_orbits))
        orbits_to_delete = []
        for k1, orbit_indices in equivalent_orbits.items():
            orbit1 = self.orbit_list.get_orbit(k1)

            for k2 in orbit_indices:

                # sanity checks
                if k1 == k2:
                    raise ValueError(f'Cannot merge orbit {k1} with itself.')
                if k2 in orbits_to_delete:
                    raise ValueError(f'Orbit {k2} cannot be merged into orbit {k1}'
                                     ' since it was already merged with another orbit.')
                orbit2 = self.orbit_list.get_orbit(k2)
                if orbit1.order != orbit2.order:
                    raise ValueError(f'The order of orbit {k1} ({orbit1.order}) does not'
                                     f' match the order of orbit {k2} ({orbit2.order}).')

                if not ignore_permutations:
                    # compare site permutations
                    permutations1 = [el['site_permutations']
                                     for el in orbit1.cluster_vector_elements]
                    permutations2 = [el['site_permutations']
                                     for el in orbit2.cluster_vector_elements]
                    for vec_group1, vec_group2 in zip(permutations1, permutations2):
                        if len(vec_group1) != len(vec_group2) or \
                                not np.allclose(np.array(vec_group1), np.array(vec_group2)):
                            raise ValueError(f'Orbit {k1} and orbit {k2} have different '
                                             'site permutations.')

                        # compare multi-component vectors (maybe this is redundant because
                        # site permutations always differ if multi-component vectors differ?)
                    mc_vectors1 = [el['multicomponent_vector']
                                   for el in orbit1.cluster_vector_elements]
                    mc_vectors2 = [el['multicomponent_vector']
                                   for el in orbit2.cluster_vector_elements]
                    if not all(np.allclose(vec1, vec2)
                               for vec1, vec2 in zip(mc_vectors1, mc_vectors2)):
                        raise ValueError(f'Orbit {k1} and orbit {k2} have different '
                                         'multi-component vectors.')

                # merge
                self._merge_orbit(k1, k2)
                orbits_to_delete.append(k2)

        # update merge/prune history
        self._remove_orbits(orbits_to_delete)

    def is_supercell_self_interacting(self, structure: Atoms) -> bool:
        """
        Checks whether a structure has self-interactions via periodic
        boundary conditions.
        Returns ``True`` if the structure contains self-interactions via periodic
        boundary conditions, otherwise ``False``.

        Parameters
        ----------
        structure
            Structure to be tested.
        """
        ol = self.orbit_list.get_supercell_orbit_list(
            structure=structure,
            fractional_position_tolerance=self.fractional_position_tolerance)
        orbit_indices = set()
        for orbit in ol.orbits:
            for cluster in orbit.clusters:
                indices = tuple(sorted([site.index for site in cluster.lattice_sites]))
                if indices in orbit_indices:
                    return True
                else:
                    orbit_indices.add(indices)
        return False

    def get_multiplicities(self) -> List[int]:
        """
        Get multiplicities for each cluster space element as a list.
        """
        return [elem['multiplicity'] for elem in self.as_list]

    def write(self, filename: str) -> None:
        """
        Saves cluster space to a file.

        Parameters
        ---------
        filename
            Name of file to which to write.
        """

        with tarfile.open(name=filename, mode='w') as tar_file:

            # write items
            items = dict(cutoffs=self._cutoffs,
                         chemical_symbols=self._input_chemical_symbols,
                         pruning_history=self._pruning_history,
                         symprec=self.symprec,
                         position_tolerance=self.position_tolerance)
            temp_file = tempfile.TemporaryFile()
            pickle.dump(items, temp_file)
            temp_file.seek(0)
            tar_info = tar_file.gettarinfo(arcname='items', fileobj=temp_file)
            tar_file.addfile(tar_info, temp_file)
            temp_file.close()

            # write structure
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.close()
            ase_write(temp_file.name, self._input_structure, format='json')
            with open(temp_file.name, 'rb') as tt:
                tar_info = tar_file.gettarinfo(arcname='atoms', fileobj=tt)
                tar_file.addfile(tar_info, tt)
            os.remove(temp_file.name)

    @staticmethod
    def read(filename: str):
        """
        Reads cluster space from file and returns :attr:`ClusterSpace` object.

        Parameters
        ---------
        filename
            Name of file from which to read cluster space.
        """
        if isinstance(filename, str):
            tar_file = tarfile.open(mode='r', name=filename)
        else:
            tar_file = tarfile.open(mode='r', fileobj=filename)

        # read items
        items = pickle.load(tar_file.extractfile('items'))

        # read structure
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(tar_file.extractfile('atoms').read())
        temp_file.close()
        structure = ase_read(temp_file.name, format='json')
        os.remove(temp_file.name)

        tar_file.close()

        # ensure backward compatibility
        if 'symprec' not in items:  # pragma: no cover
            items['symprec'] = 1e-5
        if 'position_tolerance' not in items:  # pragma: no cover
            items['position_tolerance'] = items['symprec']

        cs = ClusterSpace(structure=structure,
                          cutoffs=items['cutoffs'],
                          chemical_symbols=items['chemical_symbols'],
                          symprec=items['symprec'],
                          position_tolerance=items['position_tolerance'])
        if len(items['pruning_history']) > 0:
            if isinstance(items['pruning_history'][0], tuple):
                for key, value in items['pruning_history']:
                    if key == 'prune':
                        cs.prune_orbit_list(value)
                    elif key == 'merge':
                        # It is safe to ignore permutations here because otherwise
                        # the orbits could not have been merged in the first place.
                        cs.merge_orbits(value, ignore_permutations=True)
            else:  # for backwards compatibility
                for value in items['pruning_history']:
                    cs.prune_orbit_list(value)

        return cs

    def copy(self):
        """ Returns copy of :class:`ClusterSpace` instance. """
        cs_copy = ClusterSpace(structure=self._input_structure,
                               cutoffs=self.cutoffs,
                               chemical_symbols=self._input_chemical_symbols,
                               symprec=self.symprec,
                               position_tolerance=self.position_tolerance)

        for key, value in self._pruning_history:
            if key == 'prune':
                cs_copy.prune_orbit_list(value)
            elif key == 'merge':
                # It is safe to ignore permutations here because otherwise
                # the orbits could not have been merged in the first place.
                cs_copy.merge_orbits(value, ignore_permutations=True)
        return cs_copy
