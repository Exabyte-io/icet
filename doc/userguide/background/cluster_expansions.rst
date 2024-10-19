.. _cluster_expansions:

Cluster expansions
==================

This page provides a short introduction to cluster expansions. For more
extensive descriptions, please consult for example [SanDucGra84]_, [Wal09]_ or
[AngMunRah19]_.
A comprehensive tutorial can be found in [EkbRosFra24]_.

.. index::
   single: Cluster expansion; Formalism

Formalism
---------

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

Here, the sum extends over all symmetry equivalent clusters
(:term:`orbit`) :math:`\alpha`, :math:`m_{\alpha}` denotes the
multiplicity, whereas the coefficients :math:`J_{\alpha}` are the
so-called effective cluster interactions (:term:`ECIs`). The last term
in the above expression represents the average over cluster functions
:math:`\Gamma_{\alpha}(\boldsymbol{\sigma})` belonging to symmetry
equivalent clusters. The cluster functions themselves are obtained as
a product over point functions :math:`\Theta`. In :program:`icet`, the
point functions from [Wal09]_ are used:

.. math::
    
    \Theta_{n}(\sigma_p) &=
    \begin{cases}
      1 & \qquad \text{if $n=0$} \\
    - \cos\left(\pi(n+1)\sigma_p/M\right) & \qquad \text{if $n$ is odd} \\
    -\sin\left(\pi n \sigma_p/M\right) & \qquad \text{if $n$ is even},
    \end{cases}

These point functions form an orthogonal set over all possible occupation
numbers. For further details, please consult [Wal09]_ or [AngMunRah19]_.

In :program:`icet`, the formalism is handled internally by the
:class:`ClusterSpace <icet.ClusterSpace>` class.


.. index::
   single: Cluster expansion; Construction

Cluster expansion construction
------------------------------

The task of training a :term:`CE` can be formally written as a linear problem

.. math::
   \mathbf{\bar{\Pi}} \boldsymbol{J} = \boldsymbol{Q}

where :math:`\boldsymbol{Q}` is the vector of target properties, the
vector :math:`\boldsymbol{J}` represents the ECIs, and
:math:`\mathbf{\bar{\Pi}}` is a matrix that is obtained by stacking
the vectors that represent each structure (also referred to in this
documentation as cluster vectors) of the training set. Here, the
multiplicities :math:`m_{\alpha}` have been included in 
:math:`\mathbf{\bar{\Pi}}`.

This problem can be approached by choosing the number of structures
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

Traditional algorithms [Wal09]_ then proceed by generating a series of :term:`CEs` corresponding to different basis set choices, i.e. different values of :math:`n_{\boldsymbol{J}}`.
By comparing the performance of each :term:`CE` by means of a cross validation (CV) score the best performing :term:`CE` is selected. 
An alternative approach is to choose the number of ECIs larger than the number of structures (:math:`n_{\boldsymbol{J}}>n_{\boldsymbol{Q}}`).
The resulting problem is then underdetermined, but can be solved by means of compressive sensing [CanWak08]_ [NelHarZho13]_ [NelOzoRee13]_.
In practice, one obtains a sparse solution, i.e., a solution in which many ECIs are zero.
It is also possible to employ Bayesian approaches [MueCed09]_ or genetic algorithms [BluHarWal05]_.

:program:`icet` is agnostic to a particular optimization approach and can in principle be used in conjunction with any of these techniques.
For many applications, the :program:`trainstation` package, the documentation of which can be found `here <https://trainstation.materialsmodeling.org>`_, provides a particular simple interface and many of our examples use this package.
