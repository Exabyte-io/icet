.. _tutorial_construct_cluster_expansion:
.. highlight:: python
.. index::
   single: Tutorial; Constructing a cluster expansion

Construction of a cluster expansion
===================================

In this step we will construct a cluster expansion using the structures
generated :ref:`previously <tutorial_prepare_reference_data>` that will be
analyzed in the following steps.

General preparations
--------------------

A number of `ASE <https://wiki.fysik.dtu.dk/ase>`_ and :program:`icet`
functions are needed in order to set up and train the cluster expansion.
Specifically, :func:`ase.build.bulk` and :func:`ase.db.connect()
<ase.db.core.connect>` are required to build a primitive structure and import
relaxed configurations from the database that was generated :ref:`previously
<tutorial_prepare_reference_data>`. The :program:`icet` classes
:class:`ClusterSpace <icet.ClusterSpace>`, :class:`StructureContainer
<icet.StructureContainer>`, :class:`Optimizer <icet.Optimizer>` and
:class:`ClusterExpansion <icet.ClusterExpansion>` are used, in sequence, during
preparation, compilation and training of the cluster expansion followed by the
extraction of information in the form of predicted energies from the latter. In
the final step, the function :func:`enumerate_structures()
<icet.tools.structure_enumeration.enumerate_structures>` is employed to
generate a large pool of structures for which the mixing energies can be
calculated with help of the finalized cluster expansion. These data are plotted
using the `matplotlib <https://matplotlib.org>`_ library.

.. literalinclude:: ../../../../tutorial/basic/2_construct_cluster_expansion.py
   :end-before: # step 1

Preparation of cluster space
----------------------------

In order to be able to build a cluster expansion, it is first necessary to
create a :class:`ClusterSpace <icet.ClusterSpace>` object based on a prototype
structure, here in the form of a bulk gold unit cell. When initiating the
former, one must also provide cutoffs and a list of elements that should be
considered, in this case gold and silver. Here, the cutoffs are set to 6, 5,
and 4 Å for pairs, triplets and quadruplets.

.. literalinclude:: ../../../../tutorial/basic/2_construct_cluster_expansion.py
   :start-after: # step 1
   :end-before: # step 2

As with many other :program:`icet` objects, it is possible to print core
information in a tabular format by simply calling the :func:`print` function
with the instance of interest as input argument. For the case at hand, the
output should be the following::

  ========================== Cluster Space ===========================
   subelements: Ag Au
   cutoffs: 6.0000 5.0000 4.0000
   total number of orbits: 14
   number of orbits by order: 0= 1  1= 1  2= 4  3= 7  4= 1
  --------------------------------------------------------------------
  index | order |   size   | multiplicity | orbit index |  MC vector
  --------------------------------------------------------------------
     0  |   0   |   0.0000 |        1     |      -1
     1  |   1   |   0.0000 |        1     |       0     |    [0]
     2  |   2   |   1.4425 |        6     |       1     |  [0, 0]
     3  |   2   |   2.0400 |        3     |       2     |  [0, 0]
     4  |   2   |   2.4985 |       12     |       3     |  [0, 0]
     5  |   2   |   2.8850 |        6     |       4     |  [0, 0]
     6  |   3   |   1.6657 |        8     |       5     | [0, 0, 0]
     7  |   3   |   1.8869 |       12     |       6     | [0, 0, 0]
     8  |   3   |   2.0168 |       24     |       7     | [0, 0, 0]
     9  |   3   |   2.3021 |       24     |       8     | [0, 0, 0]
    10  |   3   |   2.4967 |       24     |       9     | [0, 0, 0]
    11  |   3   |   2.7099 |       24     |      10     | [0, 0, 0]
    12  |   3   |   2.8850 |        8     |      11     | [0, 0, 0]
    13  |   4   |   1.7667 |        2     |      12     | [0, 0, 0, 0]
  ====================================================================

Compilation of structure container
----------------------------------

Once a :class:`ClusterSpace <icet.ClusterSpace>` has been prepared, the next
step is to compile a :class:`StructureContainer <icet.StructureContainer>`. To
this end, we first initialize an empty :class:`StructureContainer
<icet.StructureContainer>` and then add the tructures from the database
prepared previously including for each structure the mixing energy in the
property dictionary.

.. literalinclude:: ../../../../tutorial/basic/2_construct_cluster_expansion.py
   :start-after: # step 2
   :end-before: # step 3

By calling the :func:`print` function with the :class:`StructureContainer
<icet.StructureContainer>` as input argument, one obtains the following
result::

  ====================== Structure Container ========================
  otal number of structures: 137
  -------------------------------------------------------------------
  ndex |       user_tag        | natoms | chemical formula |  energy
  -------------------------------------------------------------------
    0  | 0                     |     1  | Ag               |    0.000
    1  | 1                     |     1  | Au               |    0.000
    2  | 2                     |     2  | AgAu             |   -0.010
    3  | 3                     |     2  | AgAu             |   -0.011
    4  | 4                     |     3  | Ag2Au            |   -0.008
    5  | 5                     |     3  | AgAu2            |   -0.008
    6  | 6                     |     3  | Ag2Au            |   -0.009
    7  | 7                     |     3  | AgAu2            |   -0.011
    8  | 8                     |     3  | Ag2Au            |   -0.011
    9  | 9                     |     3  | AgAu2            |   -0.010
  ...
  127  | 127                   |     6  | Ag2Au4           |   -0.010
  128  | 128                   |     6  | AgAu5            |   -0.006
  129  | 129                   |     6  | Ag5Au            |   -0.006
  130  | 130                   |     6  | Ag4Au2           |   -0.009
  131  | 131                   |     6  | Ag4Au2           |   -0.009
  132  | 132                   |     6  | Ag3Au3           |   -0.011
  133  | 133                   |     6  | Ag3Au3           |   -0.012
  134  | 134                   |     6  | Ag2Au4           |   -0.011
  135  | 135                   |     6  | Ag2Au4           |   -0.011
  136  | 136                   |     6  | AgAu5            |   -0.007
  ===================================================================

Training of parameters
----------------------

Since the :class:`StructureContainer <icet.StructureContainer>` object created
in the previous section, contains all the information required for constructing
a cluster expansion, the next step is to train the parameters, i.e. to fit the
*effective cluster interactions* (ECIs) using the target data. More precisely,
the goal is to achieve the best possible agreement with set of training
structures, which represent a subset of all the structures in the
:class:`StructureContainer <icet.StructureContainer>`. In practice, this is a
two step process that involves the initiation of an :class:`Optimizer
<icet.Optimizer>` object with the a list of target properties produced by the
:func:`StructureContainer.get_fit_data()
<icet.StructureContainer.get_fit_data>` method as input argument.

.. literalinclude:: ../../../../tutorial/basic/2_construct_cluster_expansion.py
   :start-after: # step 3
   :end-before: # step 4

The training process is started by calling the :func:`Optimizer.train
<icet.Optimizer.train>` method. Once it is finished, the results can be
displayed by providing the :class:`Optimizer <icet.Optimizer>` object to the
:func:`print` function, which gives the output shown below::

  ===================== Optimizer ======================
  fit_method                : least-squares
  number_of_target_values   : 137
  number_of_parameters      : 14
  rmse_train                : 0.000177534
  rmse_test                 : 0.000216184
  train_size                : 102
  test_size                 : 35
  ======================================================


Finalize cluster expansion.
---------------------------

At this point, the task of constructing the cluster expansion is almost
complete. The only step that remains is to tie the parameter values obtained
from the optimization to the cluster space. This is achieved through the
initiation of a :class:`ClusterExpansion <icet.ClusterExpansion>` object with
the previously created :class:`ClusterSpace <icet.ClusterSpace>` instance
together with the list of parameters, available via the
:class:`Optimizer.parameters <icet.Optimizer.parameters>` attribute, as input
arguments. The final CE is finally written to file in order to be reused in the
next steps of the tutorial.

.. literalinclude:: ../../../../tutorial/basic/2_construct_cluster_expansion.py
   :start-after: # step 4


Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``tutorial/basic/2_construct_cluster_expansion.py``

    .. literalinclude:: ../../../../tutorial/basic/2_construct_cluster_expansion.py
