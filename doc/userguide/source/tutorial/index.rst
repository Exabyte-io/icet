.. _tutorial:
.. index:: Tutorial

Tutorial
********

This tutorial serves as a hands-on introduction to :program:`icet` and provides
an overview of its key features. The objective is to predict the phase diagram
of the Ag--Pd alloy, which is based on the :term:`FCC` crystal structure. In
this context, the following sections demonstrate how to construct, validate,
and sample cluster expansions as well as to analyze the thus obtained data to
generate a phase diagram.

The scripts and database that are required for this tutorial can be downloaded
as a single zip archive. To this end, run the following command from the
command line::

    curl -O https://icet.materialsmodeling.org/tutorial.zip
    unzip tutorial.zip


Building a cluster expansion
============================

The first part of this tutorial concerns constructing, validating, and
analyzing a cluster expansion.

.. toctree::
   :maxdepth: 1

   construct_cluster_expansion
   compare_to_target_data
   enumerate_structures
   analyze_ecis


Sampling a cluster expansion
============================

The second part of this tutorial addresses sampling the cluster expansion
constructed above using Monte Carlo simulations and the analysis of these
results.

.. toctree::
   :maxdepth: 1

   run_monte_carlo
   analyze_monte_carlo
