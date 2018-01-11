.. _example_get_cluster_count:
.. highlight:: python
.. index::
   single: Examples; Cluster counts

Cluster counts
===============

The purpose of this example is to demonstrate how to carry out a cluster count
analysis.

Import modules
--------------

First, one needs to import the classes :class:`ClusterCounts
<_icet.ClusterCounts>` and :class:`Structure <_icet.Structure>`, which
are used for storing information regarding cluster counts and structures,
respectively, in addition to the functions :func:`create_orbit_list
<icet.core.orbit_list.create_orbit_list>` and :func:`get_primitive_structure
<icet.tools.geometry.get_primitive_structure>`. In particular, the latter
will be used to generate a primitive structure, for which an orbit list will be
created, using the former. Also, the `ASE <https://wiki.fysik.dtu.dk/ase>`_
function :func:`ase.build.bulk` is needed to generate the prototype structure.

.. literalinclude:: ../../../../examples/get_cluster_count.py
   :start-after: # Start import
   :end-before: # End import

Generate a prototype Ti supercell
---------------------------------

The next step is to build a prototype supercell, in the form of a
:math:`2\times2\times1` BCC cell that is randomly populated with titanium and
tungsten atoms. Moreover, the cutoff for pairs is set to 4 Å.

.. literalinclude:: ../../../../examples/get_cluster_count.py
   :start-after: # Start setup
   :end-before: # End setup

Create a primitive orbit list
-----------------------------

As will be seen later, the orbit list corresponding to the primitive structure
of the :class:`ASE Atoms` object defined earlier is required for calculating
the cluster counts. More precisely, this is achieved by first calling the
:func:`get_primitive_structure <icet.tools.geometry.get_primitive_structure>`
followed by the :meth:`Structure.from_atoms
<icet.core.structure.Structure.from_atoms>` method. Thereafter, the
primitive structure thus obtained is provided as input to
:func:`create_orbit_list <icet.core.orbit_list.create_orbit_list>`, together
with the list of cutoff distances for the clusters.

.. literalinclude:: ../../../../examples/get_cluster_count.py
   :start-after: # Determine the orbit list
   :end-before: # Use the primitive


Count the clusters
------------------

To count the number of clusters a :class:`ClusterSpace
<icet.core.cluster_space.ClusterSpace>` object is initiated first.
Thereafter, a :class:`Structure <_icet.Structure>` is generated from the
supercell that was generated above with help of the :meth:`Structure.from_atoms
<icet.core.structure.Structure.from_atoms>` method. To calculate the number
of clusters, this structure together with the primitive orbit list that was
created in the previous section are given as input arguments to the
:meth:`ClusterCounts.count_clusters <_icet.ClusterCounts.count_orbit_list>`
method.

.. literalinclude:: ../../../../examples/get_cluster_count.py
   :start-after: # Use the primitive
   :end-before: # Print all of

Print the cluster counts
------------------------

Finally, the cluster counts that were calculated by the previous code block are
printed using the :meth:`print` method.

.. literalinclude:: ../../../../examples/get_cluster_count.py
   :start-after: # Print all of

The resulting output should be similar to the following::

  number of atoms 4
  Found 3 clusters
  3.43  :: 0 0 1.715
  ==============
  Ti Ti 2
  Ti W 2
  Total: 4

  2.97047  :: 0 0 1.48523
  ==============
  Ti Ti 4
  Ti W 4
  Total: 8

   :: 0 0
  ==============
  Ti 3
  W 1
  Total: 4

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``examples/get_cluster_count.py``

    .. literalinclude:: ../../../../examples/get_cluster_count.py
