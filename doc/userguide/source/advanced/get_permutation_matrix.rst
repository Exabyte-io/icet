.. _example_get_permutation_matrix:
.. highlight:: python
.. index::
   single: Examples; Permutation matrix

Permutation matrix
==================

The purpose of this example is to demonstrate how to extract a permutation matrix.

Import modules
--------------

To extract permutation matrices, only a single :program:`icet` function is
required, namely :func:`permutation_matrix_from_atoms
<icet.core.permutation_matrix.permutation_matrix_from_atoms>`. Additionally, the
`ASE <https://wiki.fysik.dtu.dk/ase>`_ function :func:`ase.build.bulk` will be
needed to generate a structure.

.. literalinclude:: ../../../../examples/get_permutation_matrix.py
   :start-after: # Import modules
   :end-before: # Create a prototype

Generate prototype Al unit cell
-------------------------------

The next step is to build a prototype :class:`ASE Atoms` object, here an
aluminum fcc unit cell.

.. literalinclude:: ../../../../examples/get_permutation_matrix.py
   :start-after: # Create a prototype
   :end-before: # Generate a permutation

Obtain a permutation matrix
---------------------------

Next, the :func:`permutation_matrix_from_atoms
<icet.core.permutation_matrix.permutation_matrix_from_atoms>` function is called
with the :class:`ASE Atoms` object that was defined above and a neighbor cutoff
of 2.0 Å as arguments. The resulting output is a list that contains
three elements, namely
a :class:`PermutationMatrix <_icet.PermutationMatrix>`,
a primitive :class:`Structure <_icet.Structure>` and
a :class:`NeighborList <_icet.NeighborList>`.

.. literalinclude:: ../../../../examples/get_permutation_matrix.py
   :start-after: # Generate a permutation matrix
   :end-before: # Extract the permuted,

Since the verbosity was set to :math:`\geq 3`, the following extra information
should be printed by :func:`permutation_matrix_from_atoms
<icet.core.permutation_matrix.permutation_matrix_from_atoms>`::

  size of atoms_prim 1
  number of positions: 19

Extract permuted positions
----------------------------

The permuted as well as the indexed and unique positions can be extracted from
the :class:`PermutationMatrix <_icet.PermutationMatrix>` object, generated
earlier, using the methods :meth:`get_permuted_positions
<_icet.PermutationMatrix.get_permuted_positions>` and
:meth:`get_indexed_positions <_icet.PermutationMatrix.get_indexed_positions>`,
respectively.

.. literalinclude:: ../../../../examples/get_permutation_matrix.py
   :start-after: # Extract the permuted,
   :end-before: # Print the permuted,

Print the positions
-------------------

Finally the permuted as well as the indexed and unique positions, obtained in
the previous section, are printed using the following snippet:

.. literalinclude:: ../../../../examples/get_permutation_matrix.py
   :start-after: # Print the permuted,

These lines should give the following (partial) output::

  Permutated fractional coordinates
  [ 0.  0.  0.]
  [ 1.  1. -1.] [-1.  1.  1.] [-1. -1.  1.] [ 1. -1. -1.] [ 1. -1.  1.] [-1.  1. -1.]
  ...
  [ 1.  1. -1.] [-1.  1.  1.] [-1. -1.  1.] [ 1. -1. -1.] [ 1. -1.  1.] [-1.  1. -1.]
  Permutated indices and positions
  0 1 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  1 6 [1, 2, 3, 4, 2, 1, 4, 3, 2, 1, 3, 4, 1, 2, 4, 3, 5, 6, 2, 1, 6, 5, 1, 2, 6, 5, 2, 1, 5, 6, 1, 2, 4, 3, 6, 5, 3, 4, 5, 6, 3, 4, 6, 5, 4, 3, 5, 6]
  ...
  18 6 [2, 1, 4, 3, 1, 2, 3, 4, 1, 2, 4, 3, 2, 1, 3, 4, 6, 5, 1, 2, 5, 6, 2, 1, 5, 6, 1, 2, 6, 5, 2, 1, 3, 4, 5, 6, 4, 3, 6, 5, 4, 3, 5, 6, 3, 4, 6, 5]
  Unique permuted indices and positions
  0 [ 0.  0.  0.]
  1 [-1. -1.  1.]
  ...
  18 [ 0.  1.  0.]

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``examples/get_permutation_matrix.py``

    .. literalinclude:: ../../../../examples/get_permutation_matrix.py
