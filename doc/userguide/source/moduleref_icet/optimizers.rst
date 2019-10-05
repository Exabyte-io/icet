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
including slight adjustments to scikit-learn default values.
If you need more flexibility, extended functionality or the ability to
fine-tune parameters that are not included in this interface, it is possible to
use scikit-learn directly.

The most commonly used fit methods in the present context are :term:`LASSO`,
:term:`automatic relevance determination regression (ARDR) <ARDR>`,
:term:`recursive feature elimination <RFE>` with :math:`\ell_2`-fitting
(RFE-L2) as well as ordinary least-squares optimization (OLS). Below follows a
short summary of the main algorithms. More information about the available
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


Automatic relevance determination regression (ARDR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


split-Bregman
^^^^^^^^^^^^^

The split-Bregman method [GolOsh09]_ is designed to solve a broad class of
:math:`\ell_1`-regularized problems. The solution vector :math:`\boldsymbol{x}`
is given by

.. math::

    \boldsymbol{x}
    = \arg\min_{\boldsymbol{x}, \boldsymbol{d}} \left\Vert\boldsymbol{d}\right\Vert_1
    + \frac{1}{2} \left\Vert\boldsymbol{A}\boldsymbol{x} - \boldsymbol{y}\right\Vert^2
    + \frac{\lambda}{2} \left\Vert\boldsymbol{d} - \mu \boldsymbol{x} \right\Vert^2,

where :math:`\boldsymbol{d}` is an auxiliary quantity, while :math:`\mu` and
:math:`\lambda` are hyperparameters that control the sparseness of the solution
and the efficiency of the algorithm.

The split-Bregman implementation supports the following additional
keywords.

==================== ========= =================================================== =========
Parameter            Type      Description                                         Default
==================== ========= =================================================== =========
``mu``               ``float`` sparseness parameter                                ``1e-3``
``lmbda``            ``float`` weight of additional L2-norm in split-Bregman       ``100``
``n_iters``          ``int``   maximal number of split-Bregman iterations          ``1000``
``tol``              ``float`` convergence criterion iterative minimization        ``1e-6``
``verbose``          ``bool``  print additional information to stdout              ``False``
==================== ========= =================================================== =========


Recursive feature elimination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recursive feature elimination (RFE) is a feature selection algorithm that
obtains the optimal features by carrying out a series of fits, starting with
the full set of parameters and then iteratively eliminating the less important
ones. RFE needs to be combined with a specific fit method. Since RFE may
require many hundreds of single fits its often advisable to use ordinary
least-squares as training method, which is the default behavior. The present
implementation is based on the `implementation of feature selection in scikit-learn
<https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html>`_.

The RFE optimizer is chosen by setting the ``fit_method`` keyword to
``rfe``. The ``n_features`` keyword allows one to specify the number of
features to select. If this parameter is left unspecified RFE with
cross-validation will be used to determine the optimal number of features.

After the optimal number of features has been determined the final model is
trained. The fit method for the final fit can be controlled via
``final_estimator``. Here, ``estimator`` and ``final_estimator`` can be set to
any of the fit methods described in this section. For example,
``estimator='lasso'`` implies that a LASSO-CV scan is carried out for each fit
in the RFE algorithm.

+----------------------------+-----------+------------------------------------------------------------------------+---------------------+
| Parameter                  | Type      | Description                                                            | Default             |
+============================+===========+========================================================================+=====================+
| ``n_features``             | ``int``   | number of features to select                                           | ``None``            |
+----------------------------+-----------+------------------------------------------------------------------------+---------------------+
| ``step``                   | ``int``   | number parameters to eliminate                                         |                     |
+----------------------------+-----------+------------------------------------------------------------------------+---------------------+
|                            | ``float`` | percentage of parameters to eliminate                                  | ``0.04``            |
+----------------------------+-----------+------------------------------------------------------------------------+---------------------+
| ``cv_splits``              | ``int``   | number of CV splits (90/10) used when optimizing ``n_features``        | ``5``               |
+----------------------------+-----------+------------------------------------------------------------------------+---------------------+
| ``estimator``              | ``str``   | fit method to be used in RFE algorithm                                 | ``'least-squares'`` |
+----------------------------+-----------+------------------------------------------------------------------------+---------------------+
| ``final_estimator``        | ``str``   | fit method to be used in the final fit                                 | = ``estimator``     |
+----------------------------+-----------+------------------------------------------------------------------------+---------------------+
| ``estimator_kwargs``       | ``dict``  | keyword arguments for fit method defined by ``estimator``              | ``{}``              |
+----------------------------+-----------+------------------------------------------------------------------------+---------------------+
| ``final_estimator_kwargs`` | ``dict``  | keyword arguments for fit method defined by ``final_estimator``        | ``{}``              |
+----------------------------+-----------+------------------------------------------------------------------------+---------------------+

.. note::

   When running on multi-core systems please be mindful of memory
   consumption. By default all CPUs will be used (`n_jobs=-1`), which
   will duplicate data and can require a lot of memory, potentially
   giving rise to errors. To prevent this behavior you can set the
   [`n_jobs`
   parameter](https://scikit-learn.org/stable/glossary.html#term-n-jobs)
   explicitly, which is handed over directly to scikit-learn.


Other methods
^^^^^^^^^^^^^

The optimizers furthermore support the `ridge
method <https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression>`_
(``ridge``), the `elastic net
method <https://scikit-learn.org/stable/modules/linear_model.html#elastic-net>`_
(``elasticnet``) as well as `Bayesian ridge regression
<https://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression>`_
(``bayesian-ridge``).


.. index::
   single: Class reference; Optimizer

Optimizer
---------

.. autoclass:: Optimizer
   :members:
   :undoc-members:
   :inherited-members:

.. index::
   single: Class reference; EnsembleOptimizer

EnsembleOptimizer
-----------------

.. autoclass:: EnsembleOptimizer
   :members:
   :undoc-members:
   :inherited-members:


.. index::
   single: Class reference; CrossValidationEstimator

CrossValidationEstimator
------------------------

.. autoclass:: CrossValidationEstimator
   :members:
   :undoc-members:
   :inherited-members:
