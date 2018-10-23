.. raw:: html

  <p>
  <a href="https://gitlab.com/materials-modeling/icet/commits/master"><img alt="pipeline status" src="https://gitlab.com/materials-modeling/icet/badges/master/pipeline.svg" /></a>
  <a href="https://icet.materialsmodeling.org/coverage"><img alt="coverage report" src="https://gitlab.com/materials-modeling/icet/badges/master/coverage.svg" /></a>
  </p>

:program:`icet` — A Pythonic approach to cluster expansions
***********************************************************

:program:`icet` is a Python environment for the construction and sampling of
alloy cluster expansions. It features a Python interface that enables seamless
integration with other Python libraries including for example
`SciPy <https://www.scipy.org/>`_ or
`scikit-learn <http://scikit-learn.org/>`_.
Yet, all computationally demanding parts are written in C++ providing
performance while maintaining portability.

:program:`icet` requires Python3 and invokes functionality from
several external libraries including the `atomic simulation
environment <https://wiki.fysik.dtu.dk/ase>`_ and `spglib
<https://atztogo.github.io/spglib/>`_.

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
