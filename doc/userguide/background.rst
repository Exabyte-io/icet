.. _background:
.. index:: Background

Background
**********

Introduction
============

In atomic scale materials modeling (alloy) the cluster expansion (CE)
technique is most commonly used for predicting the energy as a
function of the distribution of different elements over lattice
sites. Such :term:`CEs` are usually trained to match as closely as possible a
series of first-principles calculations and subsequently sampled using
Monte Carlo (MC) simulations, which enables one to predict alloy phase
diagrams.

More generally, :term:`CEs` provide a mapping between a configuration and a
property of interest. The configuration is not restricted to be atomic
but could for example also represent the sequence of amino acids in a
proteine [NelHarZho13]_. Similarly, properties of interest can also be
much more general and include for example migration barriers, lattice
constants, or transport coefficients [AngLinErh16]_.


.. _cluster-expansions:
.. index::
   single: Cluster expansion; Formalism

Cluster expansions
==================

In the following, we are concerned with configurations corresponding
to a distribution of :math:`M` different species over :math:`N` sites
that can be written as a vector :math:`\boldsymbol{\sigma}`, whose
elements can assume :math:`M` different values, e.g., from
:math:`S=\{-m, -m+1, \ldots m-1, m\}` where :math:`M=2m` (for even
:math:`M`) or :math:`M=2m+1` (for odd :math:`M`) [SanDucGra84]_. One
now seeks to represent a property :math:`Q` of the system, such as the
total energy, as a function of :math:`\boldsymbol{\sigma}`,
i.e. :math:`Q = f(\boldsymbol{\sigma})`. To this end, one can
construct :ref:`a complete orthonormal basis of cluster functions
<cluster-functions>` :math:`\Gamma_{\alpha}(\boldsymbol{\sigma})`
[SanDucGra84]_, which allows one to express :math:`Q` in the form
[Wal09]_

.. math::

   Q
   = \sum_\alpha
   m_\alpha
   J_\alpha
   \left<\Gamma_{\alpha'}(\boldsymbol{\sigma})\right>_{\alpha}.
   
Here, the sum extends over all symmetry equivalent clusters
:math:`\alpha`, :math:`m_{\alpha}` denotes the multiplicity [#]_
whereas the coefficients :math:`J_{\alpha}` are the so-called
effective cluster interactions (ECIs). The last term in the above
expression represents the average over cluster functions
:math:`\Gamma_{\alpha}(\boldsymbol{\sigma})` belonging to symmetry
equivalent clusters. The cluster functions themselves are obtained as
a product over basis functions :math:`\Theta` as described in the
`section detailing the construction of cluster functions
<cluster-functions>`.

.. [#] Note that some authors include :math:`m_{\alpha}` in the
       symmetrized product over cluster functions
       :math:`\left<\Gamma_{\alpha'}(\boldsymbol{\sigma})\right>_{\alpha}`.

.. index::
   single: Cluster expansion; Construction

CE construction
===============

The task of training a :term:`CE` can be formally written as a linear set of
equations

.. math::
   \mathbf{\bar{\Pi}} \boldsymbol{J} = \boldsymbol{Q}

where :math:`\boldsymbol{Q}` is the vector of target properties,
:math:`\boldsymbol{J}` represents the vector of coefficients, and
:math:`\mathbf{\bar{\Pi}}` is a matrix that is obtained by stacking
the vectors that represent the clusters present in each structure of
the training set. This problem can be approached by choosing the
number of structures :math:`n_{\boldsymbol{Q}}=||\boldsymbol{Q}||_0`
(and thus the dimensionality of :math:`\boldsymbol{Q}`), to be (much)
larger than the number of ECIs
:math:`n_{\boldsymbol{J}=||\boldsymbol{J}||_0}` (and thus the
dimensionality of :math:`\boldsymbol{J}`,
(:math:`n_{\boldsymbol{Q}}>n_{\boldsymbol{J}}`). The set of equations
is thus overdetermined. The optimal set of ECIs for fixed training set
and cluster function basis is then obtained by minimizing the
:math:`l_2`-norm of :math:`\mathbf{\bar{\Pi}} \boldsymbol{J} -
\boldsymbol{Q}`

.. math::
   \boldsymbol{J}_{\text{opt}}
    = \arg\min_{\boldsymbol{J}}
   \left\{ || \mathbf{\bar{\Pi}} \boldsymbol{J}
    - \boldsymbol{Q} ||_2 \right\}.

Common algorithms [Wal09]_ then proceed by generating a series of
:term:`CEs` corresponding to different basis set choices,
i.e. different values of :math:`n_{\boldsymbol{J}}`. By comparing the
performance of each :term:`CE` by means of its :ref:`cross validation
(CV) score <cross-validation>` the best performing :term:`CE` is
selected.


.. _compressive-sensing:
.. index::
   single: Algorithms; Compressive sensing
   single: Compressive sensing

Compressive Sensing
===================

An alternative approach that offers considerable advantages with
regard to accuracy, transferability as well as efficiency is based on
the compressive sensing (CS) technique [CanWak08]_ [#]_. It was
proposed in the context of :term:`CEs` in [NelHarZho13]_. The
advantages of the :term:`CS`-approach to the construction of physical
models have been demonstrated for example in [NelHarZho13]_ and
[AngLinErh16]_.

In this case the basis set (i.e., the number of cluster functions) is
deliberately chosen to be larger than the number of configurations
(:math:`n_{\boldsymbol{J}}>n_{\boldsymbol{Q}}`), implying that the set
of equations is underdetermined. In addition, one, however, minimizes
the :math:`\ell_1`-norm of the ECI vector,

.. math::
   \boldsymbol{J}_{\text{opt}}
    = \arg\min_{\boldsymbol{J}}
   \left\{
   ||\boldsymbol{J}||_1
   \big|
   || \mathbf{\bar{\Pi}} \boldsymbol{J} - \boldsymbol{Q} || \leq \epsilon
   \right\}.

By enforcing the :math:`\ell_1`-norm one specifically seeks *sparse*
solutions, which in the context of :term:`CEs` usually leads to
physcally sensible, short-ranged solutions. The equation above
expresses the so-called :term:`LASSO` problem, where :math:`\epsilon`
determines the desired accuracy. An optimal value for the latter
parameter can in principle be obtained by assessing the performance of
CEs obtained with different :math:`\epsilon` values by using e.g.,
cross validation (:math:`CV`) techniques.

.. _mu-parameter:

Following a practice from signal processin [NelHarZho13]_ it is,
however, often more convenient to solve the following unconstrained
problem,

.. math::
   \boldsymbol{J}_{\text{opt}}
   = \arg\min_{\boldsymbol{J}}
   \left\{
   \mu ||\boldsymbol{J}||_1
   + \frac{1}{2}
   || \mathbf{\bar{\Pi}} \boldsymbol{J} - \boldsymbol{Q} ||^2
   \right\}.
   
Here, the parameter :math:`\mu` balances the accuracy of the
description of the training set vs the sparseness of the
solution. Increasing :math:`\mu` leads to sparser solutions with a
larger fitting error (underfitting) and vice versa. In practice, the
CE is relatively insensitive to the choice of this parameter. This
will be explicitly demonstrated in the `tutorial section <.

.. todo::
   create a link here once the respective tutorial exists


.. [#] The terms *compressive sensing* and *compressive sampling* are
       used interchangeably [CanWak08]_.

.. _split-bregman:
.. index::
   single: Algorithms; split-Bregman
   single: split-Bregman algorithm

Algorithms for compressive sensing
==================================

The minimization task described in the previous equation is known as
the `basis pursuit denoising problem
<https://en.wikipedia.org/wiki/Basis_pursuit_denoising>`. It can be
addressed in a number of ways including fixed-point continuation
[HalYinZha08]_, (regular) Bregman as well as split-Bregman
iteration. The latter is the default algorithm in :program:`iceT`. As
the name suggest in this approach the optimization is split in two
steps, where the first one corresponds to an ordinary :math:`\ell_2`
minimization, which can be handled using standard convex optimization
algorithms.

.. _lambda-parameter-split-bregman:
.. index:: split-Bregman algorithm; lambda parameter

The split-Bregman algorithm introduces a new parameter
:math:`\lambda`, which controls the efficiency of the minimization
process (i.e. the number of steps/matrix-vector multiplications
required for convergence) but does not affect the final result.  A
suitable value for :math:\lambda` can be determined by running a few
test calculations at fixed :math:`\mu`.

Note that thanks to its python interface, :program:`iceT` has access
to optimization algorithms that are available via other python
libraries such as `SciPy <https://www.scipy.org/>`_, `scikit-learn
<http://scikit-learn.org/>`_, or `TensorFlow
<https://www.tensorflow.org/>`_. As demonstrated in :ref:`this example
<xx>`, this enables one to employ the LASSO method directly.

.. todo::
   insert link to scikit-learn example


Significance of the :math:`\mu` parameter
=========================================

.. todo::

   rewrite/update the section on :math:`\mu` and noise

The amount of noise can be of two origins. One is from the ab initio
calculations. This can be both numeric noise and systematic
errors. The second one is the lack of information due to the
truncation of the cluster expansion.  The contributions from higher
orders of clusters and larger cutoffs will for a well converged
expansion look like random noise.

In the case where the noise is quite large the algorithm can be used
as a noise filter by increasing mu. This will make the algorithm
produce sparser solutions by making the fit to the actual data less
important. The idea is that the fit to the noise will correspond to a
dense solution which can be filtered out by demanding a sparser
solution.
