#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../../icet'))
sys.path.insert(0, os.path.abspath('../../../build/src'))
sys.path.insert(0, os.path.abspath('../../../tutorial'))
sys.path.insert(0, os.path.abspath('../../../examples'))

extensions = [
    'breathe',
    'cloud_sptheme.ext.table_styling',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_sitemap',
    ]

# html_show_sourcelink = False

graphviz_output_format = 'svg'
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
todo_include_todos = True

# Collect basic information from main module
with open('../../../icet/__init__.py', encoding='utf-8') as fd:
    lines = '\n'.join(fd.readlines())
version = re.search("__version__ = '(.*)'", lines).group(1)
release = ''
copyright = re.search("__copyright__ = '(.*)'", lines).group(1)
project = re.search("__project__ = '(.*)'", lines).group(1)
author = re.search("__maintainer__ = '(.*)'", lines).group(1)

site_url = 'https://icet.materialsmodeling.org/'
html_logo = "_static/logo.png"
html_favicon = "_static/logo.ico"
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
html_theme_options = {'display_version': False}
html_context = {
    'software':
        [('atomicrex',
          'https://atomicrex.org/',
          'interatomic potential construction'),
         ('dynasor',
          'https://dynasor.materialsmodeling.org/',
          'dynamical structure factors from MD'),
         ('hiphive',
          'https://hiphive.materialsmodeling.org/',
          'anharmonic force constant potentials'),
         ('icet',
          'https://icet.materialsmodeling.org/',
          'cluster expansions'),
         ('libvdwxc',
          'https://libvdwxc.org/',
          'library for van-der-Waals functionals'),
         ('storq',
          'https://storq.materialsmodeling.org/',
          'high-throughput submission system'),
         ('vcsgc-lammps',
          'https://vcsgc-lammps.materialsmodeling.org/',
          'Monte Carlo simulations with lammps'),
         ]}
htmlhelp_basename = 'icetdoc'
intersphinx_mapping = \
    {'ase':    ('https://wiki.fysik.dtu.dk/ase', None),
     'python': ('https://docs.python.org/3.6', None),
     'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None)}

# Options for doxygen incorporation
breathe_projects = {'icet': '../../apidoc/xml/'}
breathe_default_project = 'icet'
breathe_domain_by_extension = {'h': 'cpp'}

# Options for LaTeX output
_PREAMBLE = r"""
\usepackage{amsmath,amssymb}
\renewcommand{\vec}[1]{\boldsymbol{#1}}
\DeclareMathOperator*{\argmin}{\arg\!\min}
\DeclareMathOperator{\argmin}{\arg\!\min}
"""

latex_elements = {
    'preamble': _PREAMBLE,
}
latex_documents = [
    (master_doc, 'icet.tex', 'icet Documentation',
     'The icet developer team', 'manual'),
]


# Options for manual page output
man_pages = [
    (master_doc, 'icet', 'icet Documentation',
     [author], 1)
]


# Options for Texinfo output
texinfo_documents = [
    (master_doc, 'icet', 'icet Documentation',
     author, 'icet', 'A Pythonic approach to cluster expansions',
     'Miscellaneous'),
]
