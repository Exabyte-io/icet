# -*- coding: utf-8 -*-
"""
Main module of the icet package.
"""

from .core.cluster_space import ClusterSpace
from .core.cluster_expansion import ClusterExpansion
from .core.orbit_list import OrbitList
from .core.structure import Structure
from .core.structure_container import StructureContainer
from .fitting import (Optimizer,
                      EnsembleOptimizer,
                      CrossValidationEstimator)

__project__ = 'icet'
__description__ = 'A Pythonic approach to cluster expansions'
__copyright__ = '2020'
__license__ = 'Mozilla Public License 2.0 (MPL 2.0)'
__version__ = '1.2'
__maintainer__ = 'The icet developers team'
__email__ = 'icet@materialsmodeling.org'
__status__ = 'Stable'
__url__ = 'http://icet.materialsmodeling.org/'

__all__ = ['ClusterSpace',
           'ClusterExpansion',
           'Structure',
           'OrbitList',
           'StructureContainer',
           'Optimizer',
           'EnsembleOptimizer',
           'CrossValidationEstimator']
