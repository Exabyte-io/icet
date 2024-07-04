# -*- coding: utf-8 -*-
"""
Main module of the icet package.
"""

from .core.cluster_space import ClusterSpace
from .core.cluster_expansion import ClusterExpansion
from .core.structure_container import StructureContainer

__project__ = 'icet'
__description__ = 'A Pythonic approach to cluster expansions'
__copyright__ = '2024'
__license__ = 'Mozilla Public License 2.0 (MPL 2.0)'
__version__ = '3.0'
__maintainer__ = 'The icet developers team'
__email__ = 'icet@materialsmodeling.org'
__status__ = 'Stable'
__url__ = 'http://icet.materialsmodeling.org/'

__all__ = [
    'ClusterSpace',
    'ClusterExpansion',
    'StructureContainer',
]
