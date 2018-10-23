.. _example_get_neighbor_list:
.. highlight:: python
.. index::
   single: Examples; Neighbor list

Neighbor list
=============

The purpose of this example is to demonstrate how to build a neighbor list.

Import modules
--------------

It is necessary to import the :class:`Structure <_icet.Structure>` class,
which is used to store information regarding a specific structure as well as
for the :func:`get_neighbor_lists
<icet.core.neighbor_list.get_neighbor_lists>` function. Additionally, the
`ASE <https://wiki.fysik.dtu.dk/ase>`_ function :func:`ase.build.bulk` will be
needed to generate the structures.

.. literalinclude:: ../../../../examples/get_neighbor_list.py
   :start-after: # Import modules
   :end-before: # Generate an icet

Generate prototype Al supercell
-------------------------------

The next step is to build a prototype structure, in the form of a aluminum
:math:`2\times2\times2` supercell.

.. literalinclude:: ../../../../examples/get_neighbor_list.py
   :start-after: # Generate an icet
   :end-before: # Construct a list

Obtain neighbor list
--------------------

A list of all neighbors within a cutoff distance of 1.5 Å is obtained by
calling the :func:`get_neighbor_lists
<icet.core.neighbor_list.get_neighbor_lists>` function and providing a
:class:`Structure <_icet.Structure>` object, which was defined above, and a
list of cutoffs as input arguments. One should note that the output from this
functions is a list that contains one :class:`NeighborList
<icet.core.neighbor_list.NeighborList>` object per cutoff.

.. literalinclude:: ../../../../examples/get_neighbor_list.py
   :start-after: # Construct a list
   :end-before: # Loop over all

Print all neighbors
-------------------

All relevant information can be extracted from the :class:`NeighborList
<icet.neighbor_list.NeighborList>` object, obtained earlier using the built-
in methods for this class. Here, we will merely loop over the atoms in the
structure and print the indices, offsets and distances for all the neighbors.

The first step is to use :meth:`NeighborList.get_neighbors
<icet.core.neighbor_list.NeighborList.get_distance>` to obtain the list of
neighbors for the atom with the specified index. While the indices and offsets
are attributes of the objects in the :class:`NeighborList
<icet.core.neighbor_list.NeighborList>` the distances are not. For this
reason, it is calculated using the :meth:`NeighborList.get_distance
<icet.core.neighbor_list.NeighborList.get_distance>` method, which takes the
indices and offsets of the atom and the neighbor as input arguments.

.. literalinclude:: ../../../../examples/get_neighbor_list.py
   :start-after: # Loop over all

These lines should give the following (partial) output::

  Neighbors of atom with index 0
  1 [ 0.  0. -1.] 1.41421
  1 [ 0.  0.  0.] 1.41421
  2 [ 0. -1.  0.] 1.41421
  2 [ 0.  0.  0.] 1.41421
  3 [ 0. -1.  0.] 1.41421
  3 [ 0.  0. -1.] 1.41421
  4 [-1.  0.  0.] 1.41421
  4 [ 0.  0.  0.] 1.41421
  5 [-1.  0.  0.] 1.41421
  5 [ 0.  0. -1.] 1.41421
  6 [-1.  0.  0.] 1.41421
  6 [ 0. -1.  0.] 1.41421
  ...
  Neighbors of atom with index 7
  1 [ 0.  1.  0.] 1.41421
  1 [ 1.  0.  0.] 1.41421
  2 [ 0.  0.  1.] 1.41421
  2 [ 1.  0.  0.] 1.41421
  3 [ 0.  0.  0.] 1.41421
  3 [ 1.  0.  0.] 1.41421
  4 [ 0.  0.  1.] 1.41421
  4 [ 0.  1.  0.] 1.41421
  5 [ 0.  0.  0.] 1.41421
  5 [ 0.  1.  0.] 1.41421
  6 [ 0.  0.  0.] 1.41421
  6 [ 0.  0.  1.] 1.41421

  fcc has 12 nearest neighbors

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``examples/get_neighbor_list.py``

    .. literalinclude:: ../../../../examples/get_neighbor_list.py
