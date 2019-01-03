.. raw:: html

  <p>
  <a href="https://gitlab.com/materials-modeling/icet/commits/master"><img alt="pipeline status" src="https://gitlab.com/materials-modeling/icet/badges/master/pipeline.svg" /></a>
  <a href="https://icet.materialsmodeling.org/coverage"><img alt="coverage report" src="https://gitlab.com/materials-modeling/icet/badges/master/coverage.svg" /></a>
  </p>

:program:`icet` — A Pythonic approach to cluster expansions
***********************************************************

:program:`icet` is a tool for the construction and sampling of alloy cluster
expansions. A detailed description of the functionality provided as well as an
extensive tutorial can be found in the `user guide
<https://icet.materialsmodeling.org/>`_

:program:`icet` is written in Python, which allows easy integration with
countless first-principles codes and analysis tools accessible from Python, and
allows for a simple and intuitive user interface. All computationally demanding
parts are, however, written in C++ providing performance while maintaining
portability.

The following snippet provides a minimal example for its usage:

.. code-block:: python

   cs = ClusterSpace(primitive_cell, cutoffs, species)
   sc = StructureContainer(cs, list_of_training_structure)
   opt = Optimizer(sc.get_fit_data())
   opt.train()
   ce = ClusterExpansion(cs, opt.parameters)

:program:`icet` and its development are hosted on `gitlab
<https://gitlab.com/materials-modeling/icet>`_. Bugs and feature
requests are ideally submitted via the `gitlab issue tracker
<https://gitlab.com/materials-modeling/icet/issues>`_. The development
team can also be reached by email via icet@materialsmodeling.org.

.. toctree::
   :maxdepth: 2
   :caption: Main

   overview
   background
   workflow
   installation
   basic/index
   advanced/index

.. toctree::
   :maxdepth: 2
   :caption: Function reference

   moduleref_icet/index
   moduleref_mchammer/index
   coreref/index

.. toctree::
   :maxdepth: 2
   :caption: Backmatter

   bibliography
   glossary
   genindex
