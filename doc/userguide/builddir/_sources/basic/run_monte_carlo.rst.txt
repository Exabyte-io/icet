.. _tutorial_monte_carlo_simulations:
.. highlight:: python
.. index::
   single: Tutorial; Monte Carlo simulations

Monte Carlo simulations
=======================

We are now in a position to carry out a series of Monte Carlo (MC) simulations
to sample the cluster expansion model that was constructed and validated in the
previous steps. To set up the simulation we first construct a supercell and
initialize an associated calculator by combining :ref:`our cluster expansion
model <tutorial_construct_cluster_expansion>` with the supercell.

.. literalinclude:: ../../../../tutorial/basic/5_run_monte_carlo.py
   :start-after: # step 1
   :end-before: # step 2

In this example the sampling will be carried out in the semi-grand canonical
(SGC) ensemble. To this end, we set up a :ref:`SGC ensemble object <sgc>`
object and loop over both temperatures and chemical potential differences.

We carry out a rather long MC run, anticipating that the analysis will
only include the latter part of the simulation after equilibration.
After the run the results are written to file (in the form of a
:ref:`DataContainer <data_container>` object). The latter will be used
in the next step to analyze the runs. Note that the ensemble object is
only initialized once for each temperature. Thereby the configuration
evolves gradually and the period needed for equilibration is
shortened.

.. literalinclude:: ../../../../tutorial/basic/5_run_monte_carlo.py
   :start-after: # step 2

On an Intel i7-950 CPU the set up of the calculator takes about 10 seconds,
whereas the Monte Carlo simulation takes about 0.2 seconds per MC trial step.

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``tutorial/basic/5_run_monte_carlo.py``

    .. literalinclude:: ../../../../tutorial/basic/5_run_monte_carlo.py
