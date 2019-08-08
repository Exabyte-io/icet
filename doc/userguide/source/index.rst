.. raw:: html

  <p>
  <a href="https://gitlab.com/materials-modeling/icet/commits/master"><img alt="pipeline status" src="https://gitlab.com/materials-modeling/icet/badges/master/pipeline.svg" /></a>
  <a href="https://icet.materialsmodeling.org/htmlcov"><img alt="coverage report" src="https://gitlab.com/materials-modeling/icet/badges/master/coverage.svg" /></a>
  <a href="https://badge.fury.io/py/icet"><img src="https://badge.fury.io/py/icet.svg" alt="PyPI version" height="18"></a>
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

:program:`icet` has been developed at the `Department of Physics
<https://www.chalmers.se/en/departments/physics/Pages/default.aspx>`_
of `Chalmers University of Technology <https://www.chalmers.se/>`_ in
Gothenburg, Sweden, and the `Data and Software Management Center
<https://europeanspallationsource.se/data-management-software#data-analysis-modelling>`_
at the European Spallation Source in Copenhagen, Denmark. Please
consult the :ref:`credits page <credits>` for information on how to
cite :program:`icet`.

:program:`icet` and its development are hosted on `gitlab
<https://gitlab.com/materials-modeling/icet>`_. Bugs and feature
requests are ideally submitted via the `gitlab issue tracker
<https://gitlab.com/materials-modeling/icet/issues>`_. The development
team can also be reached by email via icet@materialsmodeling.org.

.. toctree::
   :maxdepth: 2
   :caption: Main

   background/index
   installation
   tutorial/index
   advanced_topics/index
   credits

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
