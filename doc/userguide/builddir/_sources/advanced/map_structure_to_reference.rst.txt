.. _example_map_structure_to_reference:
.. highlight:: python
.. index::
   single: Examples; Structure mapping
   single: Examples; Mapping structures

Mapping structures
==================

A cluster vector calculation requires all atoms to reside on a fixed
lattice. Properties of interest, on the other hand, are typically
calculated for a structure in which cell metric and atoms have
been allowed to relax. Unless the ideal structures have been saved prior to
relaxation, one is therefore faced with the task of mapping back the relaxed
structure onto the ideal one. In some cases, in particular involving vacancies,
relaxation can also lead to atoms moving between sites, in which case
remapping is mandatory.

This is the purpose of the function :func:`map_structure_to_reference
<icet.tools.map_sites.map_structures_to_reference>`. The function is also
useful to analyze whether the relaxation has gone too far for the cluster
expansion to be viable, i.e., whether the ideal structure from which the
relaxation started is not a valid representation of the structure for which
the property has been obtained.

Import modules
--------------

The :func:`map_structure_to_reference
<icet.tools.map_sites.map_structures_to_reference>`
function needs to be imported together with some additional functions from `ASE
<https://wiki.fysik.dtu.dk/ase>`_.

.. literalinclude:: ../../../../examples/map_structure_to_reference.py
   :start-after: # Import modules
   :end-before: # End import

Prepare dummy structures
------------------------

First, for the sake of demonstration, a reference structure defining
the ideal lattice is created, and a supercell thereof is scaled and
rattled to simulate relaxation in an energy minimixation.

.. literalinclude:: ../../../../examples/map_structure_to_reference.py
   :start-after: # simulate a relaxed structure.
   :end-before: # Map the

Map the relaxed structure onto an ideal structure
-------------------------------------------------

The structure can now be mapped onto a structure in which all atoms reside
on ideal lattice sites. The function returns the ideal structure, as well as
the maximum and average displacement.

.. literalinclude:: ../../../../examples/map_structure_to_reference.py
   :start-after: # Map the "relaxed"
   :end-before: # Map a structure

Structures with vacancies
-------------------------

If the structure to be mapped contains vacancies additional keywords
should e provided, (1) ``vacancy_type``, the chemical symbol that
signifies vacancies in the reference structure, and (2)
``inert_species``, a list of elements that are never substituted for a
vacancy. The latter allows the code to rescale the volume of the cell
and can be omitted, but the mapping is then more likely to fail.

In the example below, a Au-Pd-H-vacancy system is created. In this
example, vanadium (``'V'``) represents vacancies. The system of choice
consists of two sublattices, one occupied by Au and Pd and another
occupied by H and vacancies. Since Au and Pd belong to a sublattice
in which we do not allow vacancies, we may set ``inert_species =
['Au', 'Pd']``.

.. literalinclude:: ../../../../examples/map_structure_to_reference.py
   :start-after: # Pd and Au share

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``examples/map_structure_to_reference.py``

    .. literalinclude:: ../../../../examples/map_structure_to_reference.py
