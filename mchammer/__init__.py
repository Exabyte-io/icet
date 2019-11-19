# -*- coding: utf-8 -*-

from .configuration_manager import ConfigurationManager
from .data_container import DataContainer

"""
mchammer - Monte Carlo simulation module
"""

__project__ = 'icet-mchammer'
__description__ = 'icet Monte Carlo simulations module'
__authors__ = ['Mattias Ångqvist',
               'William A. Muñoz',
               'J. Magnus Rahm',
               'Erik Fransson',
               'Céline Durniak',
               'Piotr Rozyczko',
               'Thomas Holm Rod',
               'Paul Erhart']
__copyright__ = '2019'
__license__ = 'Mozilla Public License 2.0 (MPL 2.0)'
__version__ = '1.0'
__maintainer__ = 'The icet developers team'
__email__ = 'icet@materialsmodeling.org'
__status__ = 'Stable'
__url__ = 'http://icet.materialsmodeling.org/'

__all__ = ['ConfigurationManager',
           'DataContainer']
