.. _bayesian_cluster_expansions:
.. highlight:: python
.. index::
   single: Examples; Bayesian cluster expansions

Bayesian cluster expansions
==============================
Bayesian :term:`CE` is a flexible approach to infer physical knowledge about the system by formulating priors for the :term:`ECIs` and including them in an inverse covariance matrix [MueCed09]_. 
Here, we showcase the use of Bayesian :term:`CE` for a low symmetry system, namely a surface slab, to couple similar orbits. We will use the example from :ref:`Customizing cluster spaces<customizing_cluster_spaces>` of a 10-layer surface slab and use priors to couple orbits far from the surface. 
We refer to our :ref:`previous tutorial<customizing_cluster_spaces>` for an introduction on how to inspect a cluster space to figure out which orbits are far from the surface. 

A more comprehensive tutorial on Bayesian cluster expansions can be found `here <https://ce-tutorials.materialsmodeling.org/part-2/low-symmetry-ce.html#Bayesian-CE>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   training_bayesian_cluster_expansions
