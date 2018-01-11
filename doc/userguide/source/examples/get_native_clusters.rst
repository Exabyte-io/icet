.. _example_get_native_clusters:
.. highlight:: python
.. index::
   single: Examples; Native clusters

Native clusters
===============

The purpose of this example is to demonstrate how to extract native clusters.

Import modules
--------------

In the present case it is necessary to import two :program:`icet` classes,
namely :class:`ClusterSpace <icet.core.cluster_space.ClusterSpace>` and
:class:`Structure <_icet.Structure>`. The respective objects are used to
store information regarding a specific cluster space and structure.
Additionally, the `ASE <https://wiki.fysik.dtu.dk/ase>`_ function
:func:`ase.build.bulk` will be used to generate the structures.

.. literalinclude:: ../../../../examples/get_native_clusters.py
   :start-after: # Import modules
   :end-before: # Create a prototype

Generate prototype Si cell
--------------------------

The next step is to build a prototype, here a unit cell of bulk silicon. It is
furthermore decided that the cluster vectors will be created by populating the
sites with either silicon or germanium. Also the cutoff for pairs is set to 10
Å.

.. literalinclude:: ../../../../examples/get_native_clusters.py
   :start-after: # it with (Si,
   :end-before: # Initiate the cluster

Initiate the cluster space
--------------------------

The cluster space is created by initiating a :class:`ClusterSpace
<icet.core.cluster_space.ClusterSpace>` object and providing the prototype
structure, cutoffs and list of elements that were defined above as arguments.

.. literalinclude:: ../../../../examples/get_native_clusters.py
   :start-after: # Initiate the cluster
   :end-before: # Prepare 2x2x1 supercells

Structure from Si/Ge supercell
------------------------------

First, a :math:`2\times2\times1` supercell is built, after which the sites are
randomly populated with Si and Ge atoms. Thereafter, an :class:`icet Structure
<_icet.Structure>` object structure is created by providing an :class:`ASE
Atoms` object as input to the :meth:`Structure.from_atoms` method.

.. literalinclude:: ../../../../examples/get_native_clusters.py
   :start-after: # Prepare 2x2x1 supercells
   :end-before: # Extract and print

Extract the native clusters
---------------------------

The native clusters are extracted with help of the
:meth:`ClusterSpace.get_native_clusters
<icet.core.cluster_space.ClusterSpace.get_native_clusters>` method, with the
structure defined in the previous section as input argument. Afterwards the
structure itself and the native clusters are printed in tabular format, in the
latter case by using the :func:`print` method.

.. literalinclude:: ../../../../examples/get_native_clusters.py
   :start-after: # Extract and print

The (partial) output produced by this script should similar to the following::

  Cell:
  [[ 0.     5.43   5.43 ]
   [ 5.43   0.     5.43 ]
   [ 2.715  2.715  0.   ]]

  Element and positions:
   Si  [ 0.  0.  0.]
   Ge  [ 1.3575  1.3575  1.3575]
   Si  [ 2.715  0.     2.715]
   Si  [ 4.0725  1.3575  4.0725]
   Si  [ 0.     2.715  2.715]
   Ge  [ 1.3575  4.0725  4.0725]
   Ge  [ 2.715  2.715  5.43 ]
   Si  [ 4.0725  4.0725  6.7875]

  Native cluster counts:
  5.91721  :: 0 0 2.9586
  ==============
  Si Si 3
  Si Ge 1
  Total: 4
  ...
   :: 0 0
  ==============
  Si 5
  Ge 3
  Total: 8

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``examples/get_native_clusters.py``

    .. literalinclude:: ../../../../examples/get_native_clusters.py
