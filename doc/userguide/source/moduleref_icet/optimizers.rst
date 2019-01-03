.. _optimizers:

.. index::
   single: Function reference; optimizers
   single: Class reference; optimizers

.. module:: icet.fitting

Optimizers
==========

.. index::
   single: Optimization

Overview
--------

The `scikit-learn library <https://scikit-learn.org/>`_ provides functionality
for training linear models and a large number of related tools. The present
module provides simplified interfaces for various linear model regression
methods. These methods are set up in a way that work out of the box for typical
problems in cluster expansion and force constant potential construction,
including slight adjustments to scitkit-learn default values.

If you need more flexibility, extended functionality or the ability to
fine-tune parameters that are not included in this interface, it is possible to
use scikit-learn directly.

The most commonly used fit methods in the present context are :term:`LASSO`,
:term:`automatic relevance determination regression (ARDR) <ARDR>`,
:term:`recursive feature elimination <RFE>` with :math:`\ell_2`-fitting
(RFE-L2) as well as ordinary least-squares optimization (OLS). Below follows
a short summary of the main algorithms. More information about the available
linear models can be found in the `scikit-learn documentation
<http://scikit-learn.org/stable/modules/linear_model.html>`_.


Least-squares
^^^^^^^^^^^^^

Ordinary least-squares (OLS) optimization is providing a solution to the linear
problem

.. math::

   \boldsymbol{A}\boldsymbol{x} = \boldsymbol{y},

where :math:`\boldsymbol{A}` is the sensing matrix, :math:`\boldsymbol{y}` is
the vector of target values, and :math:`\boldsymbol{x}` is the solution
(parameter vector) that one seeks to obtain. The objective is given by

.. math::

   \left\Vert\boldsymbol{A}\boldsymbol{x} - \boldsymbol{y}\right\Vert^2_2


The OLS method is chosen by setting the ``fit_method`` keyword to
``least-squares``.


LASSO
^^^^^

The `least absolute shrinkage and selection operator (LASSO)
<https://en.wikipedia.org/wiki/Lasso_(statistics)>`_ is a method for
performing variable selection and regularization in problems in
statistics and machine learning. The optimization objective is given by

.. math::

   \frac{1}{2 n_\text{samples}}
   \left\Vert\boldsymbol{A}\boldsymbol{x} - \boldsymbol{y}\right\Vert^2_2
   + \alpha \Vert\boldsymbol{x}\Vert_1.

While the first term ensures that :math:`\boldsymbol{x}` is a solution to the
linear problem at hand, the second term introduces regularization and guides
the algorithm toward finding sparse solutions, in the spirit of
:term:`compressive sensing`. In general, LASSO is suited for solving strongly
underdetermined problems.

The LASSO optimizer is chosen by setting the ``fit_method`` keyword to
``lasso``. The :math:`\alpha` parameter is set via the ``alpha`` keyword. If no
value is specified a line scan will be carried out automatically to determine
the optimal value.

==================== ========= ============================================ =========
Parameter            Type      Description                                  Default
==================== ========= ============================================ =========
``alpha``            ``float`` controls the sparsity of the solution vector ``None``
==================== ========= ============================================ =========


ARDR
^^^^

Automatic relevance determination regression (ARDR) is an optimization
algorithm provided by `scikit-learn
<https://scikit-learn.org/stable/modules/linear_model.html#automatic-relevance-determination-ard>`_
that is similar to Bayesian Ridge Regression, which provides a
probabilistic model of the regression problem at hand. The method is also known
as Sparse Bayesian Learning and Relevance Vector Machine.

The ARDR optimizer is chosen by setting the ``fit_method`` keyword to ``ardr``.
The threshold lambda parameter, which controls the sparsity of the solution
vector, is set via the ``threshold_lambda`` keyword (default: 1e6).

==================== ========= ============================================ =========
Parameter            Type      Description                                  Default
==================== ========= ============================================ =========
``threshold_lambda`` ``float`` controls the sparsity of the solution vector ``1e6``
==================== ========= ============================================ =========


RFE-L2
^^^^^^

Recursive feature elimination (RFE) with :math:`\ell_2`-fitting (RFE-L2) is a
mix between first obtaining the important features using :term:`recursive
feature elimination (RFE) <RFE>` as `implemented in scikit-learn
<https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html>`_
and then carrying out an ordinary least-square fit using the selected features.

The RFE-L2 optimizer is chosen by setting the ``fit_method`` keyword to
``rfe-l2``. The ``n_features`` keyword allows one to specify the number of
features to select. If this parameter is left unspecified RFE with
cross-validation will be used to determine the optimal number of features.

==================== ========= =================================================== =========
Parameter            Type      Description                                         Default
==================== ========= =================================================== =========
``n_features``       ``int``   number of features to select                        ``None``
``step``             ``int``   number of parameters to eliminate in each iteration ``False``
==================== ========= =================================================== =========


Other methods
^^^^^^^^^^^^^

Some other optimization methods are also available, including

* `Elastic net <https://scikit-learn.org/stable/modules/linear_model.html#elastic-net>`_ (``elasticnet``)
* `split-Bregman <https://dx.doi.org/10.1137/080725891>`_ (``split-bregman``)
* `Bayesian ridge regression <https://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression>`_ (``bayesian-ridge``)


.. index::
   single: Class reference; Optimizer

Optimizer
---------

.. autoclass:: Optimizer
   :members:
   :undoc-members:
   :inherited-members:
   :noindex:


.. index::
   single: Class reference; EnsembleOptimizer

EnsembleOptimizer
-----------------

.. autoclass:: EnsembleOptimizer
   :members:
   :undoc-members:
   :inherited-members:
   :noindex:


.. index::
   single: Class reference; CrossValidationEstimator

CrossValidationEstimator
------------------------

.. autoclass:: CrossValidationEstimator
   :members:
   :undoc-members:
   :inherited-members:
   :noindex:
