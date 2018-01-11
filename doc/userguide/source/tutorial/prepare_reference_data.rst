.. _tutorial_prepare_reference_data:
.. highlight:: python
.. index::
   single: Tutorial; Preparing reference data

Preparation of reference data
=============================

Throughout the tutorial we will be using some reference data for training and
comparison. The present section provides a short description of the code to
generate these data.


General preparations
--------------------

Several `ASE <https://wiki.fysik.dtu.dk/ase>`_ functions are required for
generating a database with reference data. Specifically, :func:`ase.db.connect`
and :func:`ase.build.bulk` are needed to initialize the database and the create
the primitive structure, respectively, while :func:`ase.calculators.emt.EMT`
and :func:`ase.optimize.BFGS` will be used relax and optimize structures. To
construct the latter the :program:`icet` function
:func:`enumerate_structures <icet.tools.structure_enumeration.enumerate_structures>`
will be employed.

.. literalinclude:: ../../../../tutorial/prepare_reference_data.py
   :start-after: # import modules
   :end-before: # step 1


Connect to database
-------------------

The first step is to initialize the database, which will be called
``structures.db``, as well as the primitive structure, in the form of an gold
bulk unit cell. Additionally, it is decided that the enumerated structures,
created in the next step, will be randomly populated with gold and silver
atoms.

.. literalinclude:: ../../../../tutorial/prepare_reference_data.py
   :start-after: # step 1
   :end-before: # step 2


Enumeration and relaxation
--------------------------

The second and final step is to generate, relax and then add the structures to
the database. This is achieved by looping over the :class:`ASE Atoms` instances
obtained by calling the :func:`enumerate_structures
<icet.tools.structure_enumeration.enumerate_structures>` function with the
primitive structure and the list of elements specified earlier as well as the
list ``sizes``, which specifies the permissible number of atoms per cell, as
input arguments. Note that the original positions of all atoms are recorded.
Next, a :func:`ase.calculators.emt.EMT` calculator is attached and the
structure is relaxed until all forces are smaller than 0.01 eV/atom. Lastly,
the relaxed structure is added to the database.

.. literalinclude:: ../../../../tutorial/prepare_reference_data.py
   :start-after: # step 2

When running this script, the :func:`ase.optimize.BFGS.run` function prints the
time, energy and maximum forces after each step of the structure relaxation. In
particular, the output should be a long list of entries, similar to the
following::

  ...
        Step     Time          Energy         fmax
  BFGS:    0 17:12:42       -0.024699        0.1726
  BFGS:    1 17:12:42       -0.025870        0.1556
  BFGS:    2 17:12:42       -0.031735        0.0157
  BFGS:    3 17:12:42       -0.031748        0.0107
  BFGS:    4 17:12:42       -0.031755        0.0073
        Step     Time          Energy         fmax
  BFGS:    0 17:12:42       -0.020691        0.0691
  BFGS:    1 17:12:42       -0.020822        0.0636
  BFGS:    2 17:12:42       -0.021549        0.0004
        Step     Time          Energy         fmax
  BFGS:    0 17:12:42       -0.023467        0.0000
  ...


Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``tutorial/prepare_reference_data.py``

    .. literalinclude:: ../../../../tutorial/prepare_reference_data.py
