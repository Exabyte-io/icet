.. _tutorial_monte_carlo_simulations:
.. highlight:: python
.. index::
   single: Tutorial; Monte Carlo simulations

Monte Carlo simulations
=======================

We are now in a position to carry out a series of Monte Carlo (MC) simulations
to sample the cluster expansion model that was :ref:`constructed
<tutorial_construct_cluster_expansion>` and :ref:`validated
<tutorial_compare_to_target_data>` in the previous steps. To set up the
simulation we first construct a supercell and initialize an associated
calculator by combining :ref:`our cluster expansion model
<tutorial_construct_cluster_expansion>` with the supercell.

.. literalinclude:: ../../../../tutorial/basic/5a_run_monte_carlo_sgc.py
   :start-after: # step 1
   :end-before: # step 2

In this example the sampling will be carried out in the semi-grand canonical
(SGC) ensemble. To this end, we set up a :ref:`SGC ensemble object
<sgc_ensemble>` object and loop over both temperatures and chemical potential
differences.

We carry out a rather long MC run, anticipating that the analysis will only
include the latter part of the simulation after equilibration. After the run
the results are stored on disk in the form of a :ref:`DataContainer
<data_container>` object. The latter will be used in the next step to analyze
the runs. At the end of each iteration we save the last state of the system
and provide it as input to the ensemble object in the next iteration.
Thereby the configuration evolves gradually and the period needed for
equilibration is shortened.

.. literalinclude:: ../../../../tutorial/basic/5a_run_monte_carlo_sgc.py
   :start-after: # step 2

On an Intel i5-6400 CPU the set up of the calculator takes about 1 second,
whereas the Monte Carlo simulation takes about 1 to 2 milliseconds per MC trial
step, but note that for a production simulation, the supercell should be larger
and the sampling both longer and more dense.

The VCSGC ensemble
------------------

A simulation in the VCSGC ensemble can be carried on an analogous fashion.

.. literalinclude:: ../../../../tutorial/basic/5b_run_monte_carlo_vcsgc.py
   :start-after: # step 2

Here, we chose :math:`\kappa=200`, which usually provides a good compromise
between acceptance probability and fluctuations in variance.

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``tutorial/basic/5a_run_monte_carlo_sgc.py`` and
       ``tutorial/basic/5b_run_monte_carlo_vcsgc.py``

    .. literalinclude:: ../../../../tutorial/basic/5a_run_monte_carlo_sgc.py
    .. literalinclude:: ../../../../tutorial/basic/5b_run_monte_carlo_vcsgc.py
