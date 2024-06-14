"""
This module provides the Orbit class.
"""

from _icet import Orbit
from numpy.typing import NDArray
__all__ = ['Orbit']


def __str__(self) -> str:
    """ String representation. """
    s = f'Order: {self.order}\n'
    s += f'Multiplicity: {len(self)}\n'
    s += f'Radius: {self.radius:.4f}\n'

    # Representative cluster
    s += 'Representative cluster:\n'
    for site, off, pos in zip(self.sites, self.site_offsets, self.positions):
        offset = ''.join([f'{x:3}' for x in off])
        position = ''.join([f' {x:6.2f}' for x in pos])
        s += f'\tSite: {site},\tOffset: {offset},\tPosition:{position}\n'

    return s


def __repr__(self) -> str:
    """ Representation. """
    s = type(self).__name__ + '('
    s += f'order={self.order}'
    s += f', multiplicity={len(self)}'
    s += f', radius={self.radius:.4f}'
    s += f', sites={self.sites}'
    s += ')'
    return s


def _repr_html_(self) -> str:
    """ HTML representation. Used, e.g., in jupyter notebooks. """
    s = ['<h4>Orbit</h4>']
    s += ['<table border="1" class="dataframe">']
    s += ['<thead><tr><th style="text-align: left;">Field</th><th>Value</th></tr></thead>']
    s += ['<tbody>']
    s += [f'<tr><td style="text-align: left;">Order</td><td>{self.order}</td></tr>']
    s += [f'<tr><td style="text-align: left;">Multiplicity</td><td>{len(self)}</td></tr>']
    s += [f'<tr><td style="text-align: left;">Radius</td><td>{self.radius:.4f}</td></tr>']
    s += [f'<tr><td style="text-align: left;">Sites</td><td>{self.sites}</td></tr>']
    s += ['</tbody>']
    s += ['</table>']
    return ''.join(s)


@property
def distances(self) -> list[float]:
    """ Distances between all sites in the representative cluster. """
    return self.representative_cluster.distances


@property
def sites(self) -> list[int]:
    """ The indices of all sites in the representative cluster."""
    return [site.index for site in self.representative_cluster.lattice_sites]


@property
def site_offsets(self) -> list[NDArray[int]]:
    """ Unit cell offsets of all sites in the representative cluster. """
    return [site.unitcell_offset.astype(int)
            for site in self.representative_cluster.lattice_sites]


@property
def positions(self) -> list[NDArray[float]]:
    """ Positions of all sites in the representative cluster. """
    return self.representative_cluster.positions


@property
def all_distances(self) -> list[list[float]]:
    """ Distances between all sites in all clusters. """
    return [cluster.distances for cluster in self.clusters]


@property
def all_sites(self) -> list[list[int]]:
    """ The site indices of all sites in all clusters."""
    return [[site.index for site in cluster.lattice_sites]
            for cluster in self.clusters]


@property
def all_site_offsets(self) -> list[list[NDArray[int]]]:
    """ Unit cell offsets of all sites in all clusters. """
    return [[site.unitcell_offset.astype(int) for site in cluster.lattice_sites]
            for cluster in self.clusters]


@property
def all_positions(self) -> list[list[NDArray[float]]]:
    """ Positions of all sites in all clusters. """
    return [cluster.positions for cluster in self.clusters]


Orbit.__str__ = __str__
Orbit.__repr__ = __repr__
Orbit._repr_html_ = _repr_html_

Orbit.distances = distances
Orbit.sites = sites
Orbit.site_offsets = site_offsets
Orbit.positions = positions

Orbit.all_distances = all_distances
Orbit.all_sites = all_sites
Orbit.all_site_offsets = all_site_offsets
Orbit.all_positions = all_positions
