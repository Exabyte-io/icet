.. index:: Glossary

Glossary
********


General
=======
.. glossary::

   BCC
        Several metallic including for example elements from groups 5 (V, Nb,
        Ta) and 6 (Cr, Mo, W) have a `body-centered cubic (BCC)
        <https://en.wikipedia.org/wiki/Cubic_crystal_system>`_ ground state
        structure.

   FCC
        The `face-centered cubic (FCC) lattice
        <https://en.wikipedia.org/wiki/Cubic_crystal_system>`_ is one of the
        most common crystal structures for metallic elements including e.g.,
        the late transition metals from group 10 (Ni, Pd, Pt) and 11 (Cu, Ag,
        Au).

   DFT
        The construction of force constants requires accurate reference data.
        `Density functional theory (DFT)
        <https://en.wikipedia.org/wiki/Density_functional_theory>`_
        calculations are one of the most common source for such data.




Crystal symmetry and clusters
=============================
.. glossary::

   crystal symmetry operation
        A crystal symmetry operation for a specific lattice means that the
        lattice is invariant under this operation. An operation comprises
        translational and rotational components.

   cluster
        A cluster is defined as a set of points on a lattice.

   cluster size
        The size of a cluster (commonly refered to as the cluster radius) is
        defined as the average distance to the geometrical center of the cluster.

   cluster space
        The set of clusters into which a structure can be decomposed.

   cutoff
        Cutoffs define the longest allowed distance between two atoms in a
        cluster for each order.

   orbit
        An orbit is defined as a set of symmetry equivalent clusters.



Cluster expansions
==================
.. glossary::

   cluster expansion
   CE
   CEs
   	     :ref:`Cluster expansions <cluster_expansions>` (CEs) provide a
   	     mapping between a configuration and a property of interest
   	     that can be many orders of magnitude faster than the
   	     underlying reference calculations from e.g., :term:`DFT`.

   DOS
         density of states

   ECI
   ECIs
	       The parameters of a :term:`CE` are usually referred to as
	       :ref:`effective cluster interactions (ECIs) <cluster_expansions>`.

   MC
         Monte Carlo (MC) simulations are an effective method for
         sampling a multi-dimensional space.

   MCS
   MCSs
         A Monte Carlo sweep (MCS) is defined as :math:`N_\mathrm{sites}` MC trial
         steps, where :math:`N_\mathrm{sites}` is the number of sites in the system.

   SQS   
         Special quasirandom structures, alloy supercells that mimic a
         random alloy using few atoms [ZunWeiFer90]_.

   WL
         The `Wang-Landau (WL) algorithm
         <https://en.wikipedia.org/wiki/Wang_and_Landau_algorithm>`_
         allows one to extract the microcanonical :term:`density of states
         (DOS) <DOS>`, from which many other thermodynamic quantities
         can be calculated [WanLan01a]_.
