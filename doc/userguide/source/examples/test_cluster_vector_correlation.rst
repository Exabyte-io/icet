.. _example_test_cluster_vector_correlation:
.. highlight:: python
.. index::
   single: Examples; Cluster vector correlations

Cluster vector correlations
===========================

The purpose of this example is to demonstrate how to test the correlation
between cluster vectors.

Import modules
--------------

In the present case it is necessary to import two :program:`icet` classes,
namely :class:`ClusterSpace <icet.core.cluster_space.ClusterSpace>` and
:class:`Structure <icet.core.structure.Structure>`. The corresponding
objects are used to store information regarding a specific cluster space and
structure, respectively. Additionally, the `ASE
<https://wiki.fysik.dtu.dk/ase>`_ function
`ase.db.connect <https://wiki.fysik.dtu.dk/ase/ase/db/db.html#ase.db.core.connect>`_
will be used to extract structures from a previously prepared database.

.. literalinclude:: ../../../../examples/test_cluster_vector_correlation.py
   :start-after: # Import modules
   :end-before: # Function for generating random

Function: Generate random structure
-----------------------------------

Since the intention is to calculate the correlation between the cluster vectors
for a number of structures in a database, first several functions are defined
that will be called in the loop over strucures described in the last section.
The first function, :func:`generate_random_structure`, takes an :class:`ASE
Atoms` object, ``atoms_prim``, creates a supercell with the dimensions
specified by ``repeat`` and populates it randomly with species from the
``subelements`` list. In the end, the corresponding
:class:`ASE Atoms` object is returned.

.. literalinclude:: ../../../../examples/test_cluster_vector_correlation.py
   :start-after: # Function for generating random
   :end-before: # Function for generating cluster

Function: Generate CV set
-------------------------

The second function, :func:`generate_cv_set`, generates cluster vectors for ``n``
supercells, which have been constructed from the :class:`ASE Atoms` object,
``atoms_prim`` and randomly populated with the elements in the ``subelements``
list, based on the :class:`ClusterSpace
<icet.core.cluster_space.ClusterSpace>` object ``cluster_space``. Specifically,
this is achieved by first calling the :func:`generate_random_structure` function,
defined in the previous section, with the :class:`ASE Atoms` object,
``atoms_prim``, the ``subelements`` list and size of the supercell,
``repeat``, as input arguments. The resulting configuration is subsequently
converted to a :class:`Structure <icet.core.structure.Structure>` object, using
the :meth:`Structure.from_atoms` method. This structure is, thereafter,
provided as an input argument to the :func:`get_cluster_vector <icet.core.cluster_space.ClusterSpace.get_cluster_vector>`
method to extract the cluster vectors.

.. literalinclude:: ../../../../examples/test_cluster_vector_correlation.py
   :start-after: # Function for generating cluster
   :end-before: # Function for calculating

Function: Get column correlation
--------------------------------

The third function, :func:`get_column_correlation`, calculates the correlation
between the columns ``i`` and ``j`` in the matrix ``cv_matrix``, which should
be a :class:`numpy.array` object.

.. literalinclude:: ../../../../examples/test_cluster_vector_correlation.py
   :start-after: # Function for calculating
   :end-before: # Function for asserting

Function: Check correlation
---------------------------

The fourth function, :func:`assert_no_correlation`, checks that the correlations
between the columns in the matrix ``cvs`` are smaller than the tolerance,
``tol``. This is achieved by calculating the correlations between all pairs of
columns with help of the :func:`get_column_correlation`, which was defined
earlier. If this value is not smaller than the tolerance, the number of the
columns and the correlation is printed to the standard output.

.. literalinclude:: ../../../../examples/test_cluster_vector_correlation.py
   :start-after: # Function for asserting
   :end-before: # Create a list

Setup
-----

Before starting the test a list of the relevant subelements, namely Pd, H, and
V, of which the latter is used for representing vacancies, is defined. Also,
the cutoff for pairs is set to 2.0 Å and it is decided that the cluster vectors
be calculated for :math:`8\times8\times8` supercells.

.. literalinclude:: ../../../../examples/test_cluster_vector_correlation.py
   :start-after: # cutoff distance for
   :end-before: # Test the correlation

Test the correlation
--------------------

The functions defined above are now employed to test the cluster vector
correlations for the first ten structures in the database ``PdHVac-fcc.db``,
which should already have been created using the script
``enumerate_structures.py``. At each step of the loop, some basic information
regarding the structure is first extracted and printed. Thereafter, a
:class:`ClusterSpace <icet.core.cluster_space.ClusterSpace>` object,
``cluster_space``, is initiated using the :class:`ASE Atoms` object
``atoms_row`` as well as the ``cutoffs`` list and the ``subelements``
list, specified in the previous section, as input arguments.
Subsequently the previously defined function is called with ``n=20`` and
``atoms_row``, ``subelements`` and ``cluster_space`` as additional input
arguments. The list of cluster vectors thus obtained is then fed into
the :func:`assert_no_correlation` function, that was described earlier, to check
if correlations between the columns are smaller than the tolerance specified
previously.

.. literalinclude:: ../../../../examples/test_cluster_vector_correlation.py
   :start-after: # previously generated database.

The result of running the script should be the following::

  Testing structure: PdH (id=1) with cutoffs [2.0]
  Number of atoms: 1024    Length of cluster vector: 6
  Testing structure: PdV (id=2) with cutoffs [2.0]
  Number of atoms: 1024    Length of cluster vector: 6
  Testing structure: Pd2VH (id=3) with cutoffs [2.0]
  Number of atoms: 2048    Length of cluster vector: 6
  Testing structure: Pd2VH (id=4) with cutoffs [2.0]
  Number of atoms: 2048    Length of cluster vector: 6
  Testing structure: Pd3VH2 (id=5) with cutoffs [2.0]
  Number of atoms: 3072    Length of cluster vector: 6
  Testing structure: Pd3V2H (id=6) with cutoffs [2.0]
  Number of atoms: 3072    Length of cluster vector: 6
  Testing structure: Pd3VH2 (id=7) with cutoffs [2.0]
  Number of atoms: 3072    Length of cluster vector: 6
  Testing structure: Pd3V2H (id=8) with cutoffs [2.0]
  Number of atoms: 3072    Length of cluster vector: 6
  Testing structure: Pd3VH2 (id=9) with cutoffs [2.0]
  Number of atoms: 3072    Length of cluster vector: 6
  Testing structure: Pd3V2H (id=10) with cutoffs [2.0]
  Number of atoms: 3072    Length of cluster vector: 6

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``examples/test_cluster_vector_correlation.py``

    .. literalinclude:: ../../../../examples/test_cluster_vector_correlation.py
