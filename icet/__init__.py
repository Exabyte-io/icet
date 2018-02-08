# -*- coding: utf-8 -*-
from __future__ import print_function, division

from .core.cluster_space import ClusterSpace, get_singlet_info, view_singlets
from .core.cluster_expansion import ClusterExpansion
from .core.structure import Structure
from .core.structure_container import StructureContainer
from .fitting import (Optimizer,
                     EnsembleOptimizer,
                     CrossValidationEstimator)

'''
icet module.
'''

__description__ = 'A Pythonic approach to cluster expansions'
__authors__ = ['Mattias Ångqvist',
               'William Armando Muñoz',
               'Thomas Holm Rod',
               'Paul Erhart']
__copyright__ = ''
__license__ = ''
__credits__ = ['Mattias Ångqvist',
               'William Armando Muñoz',
               'Thomas Holm Rod',
               'Paul Erhart']
__version__ = '0.1'
__all__ = ['ClusterSpace',
           'ClusterExpansion',
           'Structure',
           'StructureContainer',
           'Optimizer',
           'EnsembleOptimizer',
           'CrossValidationEstimator',
           'get_singlet_info',
           'view_singlets']
__maintainer__ = 'The icet developers team'
__email__ = 'icet@materialsmodeling.org'
__status__ = 'alpha-version'
__url__ = 'http://icet.materialsmodeling.org/'
