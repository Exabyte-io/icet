.. _example_get_cluster_vector:
.. highlight:: python
.. index::
   single: Examples; Cluster vectors

Cluster vectors
===============

The purpose of this example is to demonstrate how to construct cluster vectors.

Import modules
--------------

First, one needs to import the class :class:`ClusterSpace
<icet.core.cluster_space.ClusterSpace>` class, which is used to store
information regarding a given cluster space. Additionally, the `ASE
<https://wiki.fysik.dtu.dk/ase>`_ function :func:`ase.build.bulk` will be
needed to generate the structures.

.. literalinclude:: ../../../../examples/get_cluster_vector.py
   :start-after: # Import modules
   :end-before: # Create a prototype

Generate prototype Si structure
-------------------------------

The next step is to build a prototype structure, here a bulk silicon unit cell.
It is furthermore decided that the cluster vectors will be created by
populating the sites with either silicon or germanium. Also, the cutoffs for
pairs, triplets and quadruplets are all set to 5 Å.

.. literalinclude:: ../../../../examples/get_cluster_vector.py
   :start-after: # and quadruplets (5.0
   :end-before: # Initiate and print

Initiate the cluster space
--------------------------

The cluster space is created by initiating a :class:`ClusterSpace
<icet.core.cluster_space.ClusterSpace>` object and providing the prototype
structure, cutoffs and list elements defined previously as arguments. Next, the
:meth:`print` method is used to print all relevant information regarding the
cluster space in tabular format.

.. literalinclude:: ../../../../examples/get_cluster_vector.py
   :start-after: # Initiate and print
   :end-before: # Generate and print the cluster vector for a pure Si

Specifically, the final call should produce the following (partial) output::

  ------------------------- Cluster Space -------------------------
  subelements: Si Ge
  cutoffs: 5.0 5.0 5.0
  number of orbits: 21
  -----------------------------------------------------------------
  order |  radius  | multiplicity | index | orbit |    MC vector
  -----------------------------------------------------------------
    1   |   0.0000 |        2     |    0  |    0  |    [0]
    2   |   1.1756 |        4     |    1  |    1  |  [0, 0]
    2   |   1.9198 |       12     |    2  |    2  |  [0, 0]
  ...
    4   |   2.5525 |        8     |   20  |   20  | [0, 0, 0, 0]
  -----------------------------------------------------------------


Cluster vectors for Si supercells
---------------------------------

After building a new structure in the form of a :math:`2\times2\times2`
supercell, the cluster vectors are constructed using the
:meth:`ClusterSpace.get_cluster_vector
<icet.core.cluster_space.ClusterSpace.get_cluster_vector>` method for the
instance of the :class:`ClusterSpace <icet.core.cluster_space.ClusterSpace>`
class that was initiated in the previous section. The cluster vectors are
printed, as a sequence of tables, with help of the :meth:`print` method.

.. literalinclude:: ../../../../examples/get_cluster_vector.py
   :start-after: # Generate and print the cluster vector for a pure Si
   :end-before: # Generate and print the cluster vector for a mixed Si-Ge

These lines ought to yield the following result::

  [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

Cluster vectors for Si/Ge supercells
------------------------------------

Finally, the steps described in the previous section are repeated after
substituting one of the Si atoms in the supercell with Ge.

.. literalinclude:: ../../../../examples/get_cluster_vector.py
   :start-after: # Generate and print the cluster vector for a mixed Si-Ge

In this case the output should be::

  [1.0, -0.875, 0.75, 0.75, 0.75, -0.625, -0.625, -0.625, -0.625, -0.625, -0.625, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``examples/get_cluster_vector.py``

    .. literalinclude:: ../../../../examples/get_cluster_vector.py
