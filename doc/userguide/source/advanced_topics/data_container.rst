.. _advanced_topics_data_container:
.. highlight:: python
.. index::
   single: Advanced topics; Data container

Data container
==============

The results of a :term:`MC` simulation are stored in the form of a
:class:`DataContainer <mchammer.DataContainer>` object, which can be accessed
via the :func:`data_container
<mchammer.ensembles.CanonicalEnsemble.data_container>` property of the MC
ensemble. If a file name is provided during ensemble initialization via the
`data_container` parameter the data container is also written to file. The
latter can then be easily read at a later time via the :func:`read
<mchammer.DataContainer.read>` function of the :class:`DataContainer
<mchammer.DataContainer>`.

The :class:`DataContainer <mchammer.DataContainer>` class provides ample
functionality for processing data and extracting various observables that are
briefly introduced in this section.


Extracting data
---------------

The raw data as a function of MC trial step can be obtained via the
:func:`get_data <mchammer.DataContainer.get_data>` function, which also allows
slicing data by specifying an initial and final MC step. This is useful e.g.,
for discarding the equilibration part of a simulation::

    energy = dc.get_data('potential', start=5000)

The :func:`get_data <mchammer.DataContainer.get_data>` function also allows
extracting several observables in parallel::

    mctrial, energy, sro = dc.get_data('mctrial', 'potential', 'sro_Ag_1')

The available observables can be checked using the :attr:`observables
<mchammer.DataContainer.observables>` attribute.


Extracting trajectory
---------------------

The atomic configuration can be extracted using the :func:`get_trajectory
<mchammer.DataContainer.get_trajectory>`

.. code-block:: python

    traj = dc.get_trajectory()

Alternatively, the trajectory can be obtained via the :func:`get_data
<mchammer.DataContainer.get_data>` function, which also allows for pairing the
snapshots in the trajectory with observables in the data container.

.. code-block:: python

    E_mix, traj = dc.get_data('potential', 'trajectory')


Updating data container
-----------------------

Normally :ref:`observers <observers>` are attached to an ensemble at the
beginning of an MC simulation via the :func:`attach_observer
<mchammer.ensembles.CanonicalEnsemble.attach_observer>` function. They can,
however, also be applied after the fact via the :func:`apply_observer
<mchammer.DataContainer.apply_observer>` function, provided the trajectory is
available via a :class:`DataContainer <mchammer.DataContainer>` object.

.. code-block:: python

    obs = ClusterExpansionObserver(ce, tag='new_obs')
    dc = DataContainer.read('my_dc.dc')
    dc.apply_observer(obs)
    new_obs_data = dc.get_data('')

Afterwards the data container, including the new data, can be written back to
file using the :func:`write <mchammer.DataContainer.write>` function.


Data analysis
-------------

Data containers also allow more detailed analysis. The :func:`analyze_data
<mchammer.DataContainer.analyze_data>` function computes average, standard
deviation, correlation length, and 95% error estimate of the average for a
given observable.

.. code-block:: python

    summary = dc.analyze_data('potential')
    print(summary)

Here, the correlation length, :math:`s`, is estimated from the autocorrelation
function (ACF). When the ACF has decayed below :math:`\mathrm{e^{-2}}`
observations are said to be uncorrelated, providing an estimate of the
correlation length.

.. figure::
    _static/autocorrelation.svg

An `error estimate <https://en.wikipedia.org/wiki/Standard_error>`_ of the
average can be calculated via

.. math::
    \mathrm{error} = \frac{t \sigma }{\sqrt{Ns}},

where :math:`\sigma` is the standard deviation, :math:`N` the number of
samples, :math:`s` the correlation length and :math:`t` is the `t-factor
<https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_, which can be
adjusted depending on the desired confidence interval.

Obtaining the autocorrelation function directly or carrying out error estimates
can be done via functionality provided in the :ref:`data_analysis
<data_container_analysis_functions>` module.
