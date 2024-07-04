.. raw:: html

  <p>
  <a href="https://badge.fury.io/py/icet"><img src="https://badge.fury.io/py/icet.svg" alt="PyPI version" height="18"></a>
  </p>

:program:`icet` — A Pythonic approach to cluster expansions
***********************************************************

:program:`icet` is a tool for the construction and sampling of alloy cluster expansions.
It is written in Python, which enables easy integration with many first-principles codes and analysis tools accessible from Python, and allows for a simple and intuitive user interface.
All computationally demanding parts are, however, written in C++ ensuring performance while maintaining portability.

The following snippet provides a minimal example for its usage:

.. testsetup::

   from ase.build import bulk
   from icet import ClusterExpansion, ClusterSpace, StructureContainer
   from trainstation import Optimizer
   primitive_cell = bulk('Au', a=1.0, crystalstructure='fcc')
   cutoffs = [1.1]
   species = ['Au', 'Ag']

   training_structures, energies = [], []

   s = primitive_cell.repeat(1)
   training_structures.append(s)
   energies.append(-1.0)

   s = primitive_cell.repeat(2)
   s[0].symbol = 'Ag'
   training_structures.append(s)
   energies.append(-4.0)

   s = primitive_cell.repeat(2)
   s[0].symbol = 'Ag'
   s[1].symbol = 'Ag'
   training_structures.append(s)
   energies.append(-5.0)

.. doctest::

   >>> cs = ClusterSpace(primitive_cell, cutoffs, species)
   >>> sc = StructureContainer(cs)
   >>> for structure, energy in zip(training_structures, energies):
   ...     sc.add_structure(structure, properties={'energy': energy})
   >>> opt = Optimizer(sc.get_fit_data())
   >>> opt.train()
   >>> ce = ClusterExpansion(cs, opt.parameters)


:program:`icet` has been developed at the `Department of Physics <https://www.chalmers.se/en/departments/physics/Pages/default.aspx>`_ of `Chalmers University of Technology <https://www.chalmers.se/>`_ (Gothenburg, Sweden) and the `Data and Software Management Center <https://europeanspallationsource.se/data-management-software>`_ at the European Spallation Source (Copenhagen, Denmark).
Please consult the :ref:`credits page <credits>` for information on how to cite :program:`icet`.

For questions and help please use the `icet discussion forum on matsci.org <https://matsci.org/icet>`_.
:program:`icet` and its development are hosted on `gitlab <https://gitlab.com/materials-modeling/icet>`_.

.. toctree::
   :maxdepth: 2
   :caption: Main

   background/index
   installation
   tutorial/index
   faq
   credits

.. toctree::
   :maxdepth: 2
   :caption: Advanced topics

   advanced_topics/structure_generation
   advanced_topics/ce_construction
   advanced_topics/sampling

.. toctree::
   :maxdepth: 2
   :caption: Function reference

   moduleref_icet/index
   moduleref_mchammer/index
   moduleref_core/index

.. toctree::
   :maxdepth: 2
   :caption: Backmatter

   bibliography
   publications
   glossary
   genindex
