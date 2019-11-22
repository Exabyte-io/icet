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




Optimization and machine learning
=================================
.. glossary::

   ARDR
        Automatic relevance determination regression (ARDR) is an optimization
        algorithm provided by `scikit-learn
        <https://scikit-learn.org/stable/modules/linear_model.html#automatic-relevance-determination-ard>`_

   Compressive sensing
        `Compressive sensing (CS)
        <https://en.wikipedia.org/wiki/Compressed_sensing>`_, also known as
        compressive sampling, is an efficient method for constructing sparse
        solutions for linear systems.

   CV
   Cross validation
        `Cross validation (CV)
        <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_
        is commonly employed to evaluated the transferability and accuracy of
	linear problems.

   LASSO
        The `least absolute shrinkage and selection operator (LASSO)
        <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_ is a method for
        performing variable selection and regularization in problems in
        statistics and machine learning.

   Kernel ridge regression
        `Kernel ridge regression (KRR) <http://scikit-
        learn.org/stable/modules/kernel_ridge.html>`_ combines `ridge
        regression <https://en.wikipedia.org/wiki/Tikhonov_regularization>`_
        with the `kernel trick <https://en.wikipedia.org/wiki/Kernel_method>`_.

   RFE
        In machine learning `recursive feature elimination (RFE)
        <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html>`_
        is a popular `feature selection
        <https://en.wikipedia.org/wiki/Feature_selection>`_ process in model
        construction.

   Regularization
        `Regularization
        <https://en.wikipedia.org/wiki/Regularization_(mathematics)>`_,
        is commonly used in machine learning to combat overfitting and
        for solving underdetermined systems.


Crystal symmetry and clusters
=============================
.. glossary::

   Crystal symmetry operation
        A crystal symmetry operation for a specific lattice means that the
        lattice is invariant under this operation. An operation comprises
        translational and rotational components.

   Cluster
        A cluster is defined as a set of points on a lattice.

   Cluster size
        The size of a cluster (commonly refered to as the cluster radius) is
        defined as the average distance to the geometrical center of the cluster.

   Cluster space
        The set of clusters into which a structure can be decomposed.

   Cutoff
        Cutoffs define the longest allowed distance between two atoms in a
        cluster for each order.

   Orbit
   Orbits
        An orbit is defined as a set of symmetry equivalent clusters.



Cluster expansions
==================
.. glossary::

   Cluster expansion
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
         A Monte Carlo sweep (MCS) is defined as :math:`N_{sites}` MC trial
         steps, where :math:`N_{sites}` is the number of sites in the system.

   WL
         The `Wang-Landau (WL) algorithm
         <https://en.wikipedia.org/wiki/Wang_and_Landau_algorithm>`_
         allows one to extract the microcanonical :term:`density of states
         (DOS) <DOS>`, from which many other thermodynamic quantities
         can be calculated [WanLan01a]_.
