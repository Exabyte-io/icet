.. _background:
.. index:: Background

Background
**********

(expand and add more references; some references to topics such as cluster
functions are intentionally left broken as a reminder to cover these things)

.. _cluster-expansions:
.. index::
   single: Cluster expansion; Formalism

Cluster expansions
==================

In the following, we are concerned with configurations corresponding to a
distribution of :math:`M` different species over :math:`N` sites that can be
written as a vector :math:`\boldsymbol{\sigma}`, whose elements can assume
:math:`M` different values, e.g., from :math:`S=\{-m, -m+1, \ldots m-1, m\}`
where :math:`M=2m` (for even :math:`M`) or :math:`M=2m+1` (for odd :math:`M`)
[SanDucGra84]_. One now seeks to represent a property :math:`Q` of the system,
such as the total energy, as a function of :math:`\boldsymbol{\sigma}`, i.e.
:math:`Q = f(\boldsymbol{\sigma})`. To this end, one can construct a
complete orthonormal basis of cluster functions
:math:`\Gamma_{\alpha}(\boldsymbol{\sigma})` [SanDucGra84]_ [San10]_, which
allows one to express :math:`Q` in the form [Wal09]_

.. math::

   Q
   = \sum_\alpha
   m_\alpha
   J_\alpha
   \left<\Gamma_{\alpha'}(\boldsymbol{\sigma})\right>_{\alpha}.

Here, the sum extends over all symmetry equivalent clusters  (:term:`orbit`)
:math:`\alpha`, :math:`m_{\alpha}` denotes the  multiplicity [#]_ whereas the
coefficients :math:`J_{\alpha}` are the  so-called effective cluster
interactions (ECIs). The last term in the  above expression represents the
average over cluster functions  :math:`\Gamma_{\alpha}(\boldsymbol{\sigma})`
belonging to symmetry  equivalent clusters (:term:`orbits`). The cluster
functions  themselves are obtained as a product over basis functions
:math:`\Theta` as described in the section detailing the  construction of
cluster functions.

.. todo::

   insert section on cluster functions

.. [#] Note that some authors include :math:`m_{\alpha}` in the
       symmetrized product over cluster functions
       :math:`\left<\Gamma_{\alpha'}(\boldsymbol{\sigma})\right>_{\alpha}`.


.. index::
   single: Cluster expansion; Construction

CE construction
===============

The task of training a :term:`CE` can be formally written as a linear problem

.. math::
   \mathbf{\bar{\Pi}} \boldsymbol{J} = \boldsymbol{Q}

where :math:`\boldsymbol{Q}` is the vector of target properties, the
vector :math:`\boldsymbol{J}` represents the ECIs, and
:math:`\mathbf{\bar{\Pi}}` is a matrix that is obtained by stacking
the vectors that represent each structure (also referred to in this
documentation as cluster vectors) of the training set. This problem
can be approached by choosing the number of structures
:math:`n_{\boldsymbol{Q}}` (and thus the dimensionality of
:math:`\boldsymbol{Q}`), to be (much) larger than the number of ECIs
:math:`n_{\boldsymbol{J}}` (and thus the dimensionality of
:math:`\boldsymbol{J}`,
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

Common algorithms [Wal09]_ then proceed by generating a series of :term:`CEs`
corresponding to different basis set choices, i.e. different values of
:math:`n_{\boldsymbol{J}}`. By comparing the performance of each :term:`CE` by
means of a cross validation (CV) score the best performing :term:`CE` is selected.
