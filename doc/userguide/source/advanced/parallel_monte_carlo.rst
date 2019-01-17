.. _tutorial_parallel_monte_carlo_simulations:
.. highlight:: python
.. index::
   single: Tutorial; Parallel Monte Carlo simulations

Parallel Monte Carlo simulations
================================

Monte Carlo simulations are in general pleasingly parallel in the sense that
no communication is needed between two runs with different sets of parameters.
In :ref:`mchammer <moduleref_mchammer>`, this can be conveniently exploited
with the `multiprocessing package
<https://docs.python.org/3/library/multiprocessing.html>`_, which is included
in Python's standard library. A run script requires very little modification
to be parallelized. Here, the :ref:`Monte Carlo simulation in the basic
tutorial <tutorial_monte_carlo_simulations>` is reproduced. The initialization
is identic:

.. literalinclude:: ../../../../tutorial/advanced/parallel_monte_carlo.py
   :start-after: # step 1
   :end-before: # step 2

A non-parallel simulation would now run in a nested loop over all parameters.
In a parallel simulation, the content of the loop is instead wrapped in a
function:

.. literalinclude:: ../../../../tutorial/advanced/parallel_monte_carlo.py
   :start-after: # step 2
   :end-before: # step 3

Next, all sets of parameters to be run are stored in a list:

.. literalinclude:: ../../../../tutorial/advanced/parallel_monte_carlo.py
   :start-after: # step 3
   :end-before: # step 4

Finally, a `multiprocessing Pool object <https://docs.python.org/3.7/library/m
ultiprocessing.html#multiprocessing.pool.Pool>`_ is created. At this step, the
number of processes are specified. It is typically advisable to use the same
number of processes as available cores. The simulation is started by
mapping the sets of parameters to the run function:

.. literalinclude:: ../../../../tutorial/advanced/parallel_monte_carlo.py
   :start-after: # step 4

Note that in the above example, an ensemble object will always be initialized
with the same supercell, which means that the system needs to be equilibrated
from scratch for every set of parameter. If equilibration is time consuming,
it may be advisable to, for example, parallelize over temperature but not
chemical potential.

HPC environments
----------------

The above desribed parallelization can usually be used in HPC environments as
well. If the system requires a job to be launched with `aprun`, some extra
arguments may be required. The following run command has been used
successfully on at least one such system:

`aprun -n 1 -cc none python3 run-mc.py`

The job should typically run on only one node. Do not forget to adapt the
number of processes to the specified number of cores!

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``tutorial/advanced/parallel_monte_carlo.py``

    .. literalinclude:: ../../../../tutorial/advanced/parallel_monte_carlo.py
