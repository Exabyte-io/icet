.. _tutorial_parallel_monte_carlo_simulations:
.. highlight:: python
.. index::
   single: Tutorial; Parallel Monte Carlo simulations

Parallel Monte Carlo simulations
================================

Monte Carlo simulations are in general `pleasingly parallel
<https://en.wikipedia.org/wiki/Embarrassingly_parallel>`_ in the sense that no
communication is needed between two runs with different sets of parameters. In
:ref:`mchammer <moduleref_mchammer>`, this can be conveniently exploited with
the `multiprocessing package
<https://docs.python.org/3/library/multiprocessing.html>`_, which is included
in the Python standard library. A run script requires very little modification
to be parallelized. Here, the :ref:`Monte Carlo simulation in the basic
tutorial <tutorial_monte_carlo_simulations>` is reproduced. The initialization
is identic:

.. literalinclude:: ../../../../examples/advanced_topics/parallel_monte_carlo.py
   :start-after: # step 1
   :end-before: # step 2

A non-parallel simulation would now run in a nested loop over all parameters.
In a parallel simulation, the content of the loop is instead wrapped in a
function:

.. literalinclude:: ../../../../examples/advanced_topics/parallel_monte_carlo.py
   :start-after: # step 2
   :end-before: # step 3

Next, all sets of parameters to be run are stored in a list:

.. literalinclude:: ../../../../examples/advanced_topics/parallel_monte_carlo.py
   :start-after: # step 3
   :end-before: # step 4

Finally, a `multiprocessing Pool object <https://docs.python.org/3.7/library/m
ultiprocessing.html#multiprocessing.pool.Pool>`_ is created. At this step, the
number of processes is specified. It is typically advisable to use the same
number of processes as available cores. The simulation is started by
mapping the sets of parameters to the run function:

.. literalinclude:: ../../../../examples/advanced_topics/parallel_monte_carlo.py
   :start-after: # step 4

Note that in the above example, an ensemble object will always be initialized
with the same supercell, which means that the system needs to be equilibrated
from scratch for every set of parameter. If equilibration is time consuming,
it may be advisable to, for example, avoid parallelization over chemical
potential (but keeping parallelization over temperature).

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``examples/advanced_topics/parallel_monte_carlo.py``

    .. literalinclude:: ../../../../examples/advanced_topics/parallel_monte_carlo.py
