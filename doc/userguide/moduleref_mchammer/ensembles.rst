.. index::
   single: Monte Carlo; Ensembles

.. module:: mchammer.ensembles

.. _ensembles:

Ensembles
=========


.. _canonical_ensemble:

.. index::
   single: Class reference; CanonicalEnsemble
   single: Monte Carlo; Canonical ensemble

Canonical ensemble
------------------

.. autoclass:: CanonicalEnsemble
   :members:
   :inherited-members:


.. _canonical_annealing:

.. index::
   single: Class reference; CanonicalAnnealing
   single: Monte Carlo; Canonical ensemble

Canonical annealing
-------------------

.. autoclass:: CanonicalAnnealing
   :members:
   :inherited-members:



.. _sgc_ensemble:

.. index::
   single: Class reference; SemiGrandCanonicalEnsemble
   single: Monte Carlo; Semi-grand canonical ensemble

Semi-grand canonical ensemble
-----------------------------

.. autoclass:: SemiGrandCanonicalEnsemble
   :members:
   :inherited-members:



.. _vcsgc_ensemble:

.. index::
   single: Class reference; VCSGCEnsemble
   single: Monte Carlo; Variance-constrained semi-grand canonical ensemble

Variance-constrained semi-grand canonical ensemble
--------------------------------------------------

.. autoclass:: VCSGCEnsemble
   :members:
   :inherited-members:


.. _hybrid_ensemble:

.. index::
   single: Class reference; HybridEnsemble
   single: Monte Carlo; Hybrid ensemble

Hybrid ensemble
---------------

.. autoclass:: HybridEnsemble
   :members:
   :inherited-members:


.. _wang_landau_ensemble:

.. index::
   single: Class reference; WangLandauEnsemble
   single: Monte Carlo; Wang-Landau ensemble

Wang-Landau ensemble
--------------------

For analysis functions see :ref:`here <data_container_analysis_functions>`.

.. autoclass:: WangLandauEnsemble
   :members:
   :inherited-members:

.. autofunction:: mchammer.ensembles.wang_landau_ensemble.get_bins_for_parallel_simulations


Thermodynamic-integration ensemble
----------------------------------

For analysis functions see :ref:`here <data_container_analysis_functions>`.

.. autoclass:: ThermodynamicIntegrationEnsemble
   :members:
   :inherited-members:
