icet
====

**icet** is a tool for the construction and sampling of alloy cluster
expansions. A detailed description of the functionality provided as well as an
extensive tutorial can be found in the `user guide
<https://icet.materialsmodeling.org/>`_

**icet** is written in Python, which allows easy integration with countless
first-principles codes and analysis tools accessible from Python, and allows
for a simple and intuitive user interface. All computationally demanding parts
are, however, written in C++ providing performance while maintaining
portability. The following snippet illustrates how one can train a cluster
expansion:

.. code-block:: python

   cs = ClusterSpace(primitive_cell, cutoffs, species)
   sc = StructureContainer(cs, list_of_training_structure)
   opt = Optimizer(sc.get_fit_data())
   opt.train()
   ce = ClusterExpansion(cs, opt.parameters)

Afterwards the cluster expansion can be used, e.g., for finding ground state
structures, or sampled via Monte Carlo simulations.


Installation
------------

In the most simple case, **icet** can be installed using pip as follows::

    pip3 install icet --user

or alternatively::

    python3 -m pip install icet --user

Installation requires a C++11 compliant compiler. Please consult the
`installation section of the user guide
<https://icet.materialsmodeling.org/installation.html>`_ for details.

**icet** is based on Python3 and invokes functionality from other Python
libraries including
`ase <https://wiki.fysik.dtu.dk/ase>`_,
`pandas <https://pandas.pydata.org/>`_,
`numpy <http://www.numpy.org/>`_,
`scipy <https://www.scipy.org/>`_,
`scitkit-learn <http://scikit-learn.org/>`_, and
`spglib <https://atztogo.github.io/spglib/>`_.


Credits
-------

* Mattias Ångqvist
* William A. Muñoz
* Magnus Rahm
* Erik Fransson
* Céline Durniak
* Piotr Rozyczko
* Thomas Holm Rod
* Paul Erhart

**icet** has been developed at Chalmers University of Technology in Gothenburg
(Sweden) in the `Materials and Surface Theory division
<http://www.materialsmodeling.org>`_ at the Department of Physics, in
collaboration with the Data Analysis group at the `Data Management and Software
Center of the European Spallation Source
<https://europeanspallationsource.se/data-management-software#data-analysis-modelling>`_
in Copenhagen (Denmark).

**icet** and its development are hosted on `gitlab <https://gitlab.com
/materials-modeling/icet>`_. Bugs and feature requests are ideally submitted
via the `gitlab issue tracker <https://gitlab.com/materials-
modeling/icet/issues>`_. The development team can also be reached by email via
icet@materialsmodeling.org.
