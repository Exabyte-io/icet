.. _tutorial_construct_cluster_expansion:
.. highlight:: python
.. index::
   single: Tutorial; Constructing a cluster expansion

Constructing a cluster expansion
================================

In this step we will construct a cluster expansion using a dataset of Ag-Pd structures, which have been relaxed using density functional theory calculations.

General preparations
--------------------

A number of `ASE <https://wiki.fysik.dtu.dk/ase>`_ and :program:`icet` functions are needed in order to set up and train the cluster expansion.
Since the reference data is provided in the form of an `ASE <https://wiki.fysik.dtu.dk/ase>`_ database we require the :func:`ase.db.connect() <ase.db.core.connect>` function.
Furthermore, the :program:`icet` classes :class:`ClusterSpace <icet.ClusterSpace>`, :class:`StructureContainer <icet.StructureContainer>`, :class:`CrossValidationEstimator <trainstation.CrossValidationEstimator>` and :class:`ClusterExpansion <icet.ClusterExpansion>` are used, :ref:`in sequence <workflow>`, during preparation, compilation and training of the cluster expansion followed by the extraction of information in the form of predicted energies from the latter.

.. literalinclude:: ../../../examples/tutorial/1_construct_cluster_expansion.py
   :end-before: # step 1

Then we open a connection to the reference database and use the first structure in the database as the primitive structure as we happen to have prepared the database in this way.

.. literalinclude:: ../../../examples/tutorial/1_construct_cluster_expansion.py
   :start-after: # step 1
   :end-before: # step 2

Preparing a cluster space
-------------------------

In order to be able to build a cluster expansion, it is first necessary to create a :class:`ClusterSpace <icet.ClusterSpace>` object based on a prototype structure.
When initiating the former, one must also provide cutoffs and the chemical elements that are allowed to occupy different lattice sites.

The *cutoffs* are specified in the form of a list with the different elements
corresponding to clusters of increasing order (pairs, triplets, quadruplets,
etc). The values then specify the longest distance allowed between any two
atoms in clusters of the respective order. In the example below, the cluster
space will contain all pairs of atoms that are 13.5 Å or closer to each other,
all triplets among which the longest distance is 6.5 Å or less, and all
quadruplets among which the longest distance is 6.0 Å or less. If higher-order
clusters (quintuplets etc.) are to be included, one would simply extend the
list. Note here that one should typically select cutoffs with some care based on
how cluster expansions accuracy changes with cutoffs.

The *allowed chemical elements* are specified as a list. Two formats are
possible. If all sites in the structure are to be occupied identically, it
suffices to provide a simple list of chemical symbols, e.g., ``['Ag', 'Pd']``
in the case below. If there are multiple sites that are to be occupied in
different fashion, one has to provide instead a list of lists, where the outer
list must contain as many elements as there are sites in the primitive
structure and each "inner" list specifies the occupation for the respective
site. Examples for this approach can be found in the :class:`ClusterSpace
<icet.ClusterSpace>` documentation.


.. literalinclude:: ../../../examples/tutorial/1_construct_cluster_expansion.py
   :start-after: # step 2
   :end-before: # step 3

.. note::

  Here, we include *all* structures from the database with six or less atoms in
  the unit cell. This approach has been adopted in this basic tutorial for the
  sake of simplicity. In general this is *not* the preferred approach to
  assembling a data set.

.. tip::

  As with many other :program:`icet` objects, it is possible to print core
  information in a tabular format by simply calling the :func:`print` function
  with the instance of interest as input argument. When using jupyter notebooks
  you can also use the :func:`display` command which renders the output in a
  visually more appealing format.

For the case at hand, the output of the :func:`print` command should look as follows::

   ====================================== Cluster Space ======================================
    space group                            : Fm-3m (225)
    chemical species                       : ['Ag', 'Pd'] (sublattice A)
    cutoffs                                : 13.5000 6.5000 6.0000
    total number of parameters             : 82
    number of parameters by order          : 0= 1  1= 1  2= 25  3= 20  4= 35
    fractional_position_tolerance          : 2e-06
    position_tolerance                     : 1e-05
    symprec                                : 1e-05
   -------------------------------------------------------------------------------------------
   index | order |  radius  | multiplicity | orbit_index | multicomponent_vector | sublattices
   -------------------------------------------------------------------------------------------
      0  |   0   |   0.0000 |        1     |      -1     |           .           |      .     
      1  |   1   |   0.0000 |        1     |       0     |          [0]          |      A     
      2  |   2   |   1.4460 |        6     |       1     |        [0, 0]         |     A-A    
      3  |   2   |   2.0450 |        3     |       2     |        [0, 0]         |     A-A    
      4  |   2   |   2.5046 |       12     |       3     |        [0, 0]         |     A-A    
      5  |   2   |   2.8921 |        6     |       4     |        [0, 0]         |     A-A    
      6  |   2   |   3.2334 |       12     |       5     |        [0, 0]         |     A-A    
      7  |   2   |   3.5420 |        4     |       6     |        [0, 0]         |     A-A    
      8  |   2   |   3.8258 |       24     |       7     |        [0, 0]         |     A-A    
      9  |   2   |   4.0900 |        3     |       8     |        [0, 0]         |     A-A    
    ...
     72  |   4   |   2.7940 |       12     |      71     |     [0, 0, 0, 0]      |   A-A-A-A  
     73  |   4   |   2.7940 |       24     |      72     |     [0, 0, 0, 0]      |   A-A-A-A  
     74  |   4   |   2.8281 |       24     |      73     |     [0, 0, 0, 0]      |   A-A-A-A  
     75  |   4   |   2.8921 |        3     |      74     |     [0, 0, 0, 0]      |   A-A-A-A  
     76  |   4   |   2.8921 |        6     |      75     |     [0, 0, 0, 0]      |   A-A-A-A  
     77  |   4   |   2.8921 |       12     |      76     |     [0, 0, 0, 0]      |   A-A-A-A  
     78  |   4   |   2.9632 |       24     |      77     |     [0, 0, 0, 0]      |   A-A-A-A  
     79  |   4   |   2.9862 |        8     |      78     |     [0, 0, 0, 0]      |   A-A-A-A  
     80  |   4   |   3.0951 |       24     |      79     |     [0, 0, 0, 0]      |   A-A-A-A  
     81  |   4   |   3.5420 |        2     |      80     |     [0, 0, 0, 0]      |   A-A-A-A  
   ===========================================================================================


.. note::

  The ``radius`` is *not* the same as the longest distance between atoms in
  the cluster (which was the measure used for initialization via ``cutoffs``),
  but is the average distance from all atoms to the geometric center of mass (assuming
  all atoms have the same mass).


Compiling a structure container
-------------------------------

Once a :class:`ClusterSpace <icet.ClusterSpace>` has been prepared, the next
step is to compile a :class:`StructureContainer <icet.StructureContainer>`. To
this end, we first initialize an empty :class:`StructureContainer
<icet.StructureContainer>` and then add the structures from the database
prepared previously including for each structure the mixing energy in the
property dictionary.

.. literalinclude:: ../../../examples/tutorial/1_construct_cluster_expansion.py
   :start-after: # step 3
   :end-before: # step 4

.. note::

  While here we add only *one* property, the :class:`StructureContainer
  <icet.StructureContainer>` allows the addition of *several* properties. This
  can be useful e.g., when constructing (and sampling) CEs for a couple of
  different properties.

By calling the :func:`print` function with the :class:`StructureContainer
<icet.StructureContainer>` as input argument, one obtains the following
result::

   ====================== Structure Container ======================
   Total number of structures: 625
   -----------------------------------------------------------------
   index | user_tag  | n_atoms | chemical formula | mixing_energy
   -----------------------------------------------------------------
   0     | Ag        | 1       | Ag               |    0.0000    
   1     | Pd        | 1       | Pd               |    0.0000    
   2     | AgPd_0002 | 2       | AgPd             |   -0.0398    
   3     | AgPd_0003 | 3       | AgPd2            |   -0.0286    
   4     | AgPd_0004 | 3       | Ag2Pd            |   -0.0485    
   5     | AgPd_0005 | 3       | AgPd2            |   -0.0178    
   6     | AgPd_0006 | 3       | Ag2Pd            |   -0.0557    
   7     | AgPd_0007 | 3       | AgPd2            |   -0.0297    
   8     | AgPd_0008 | 3       | Ag2Pd            |   -0.0477    
   9     | AgPd_0009 | 4       | AgPd3            |   -0.0173    
   10    | AgPd_0010 | 4       | Ag3Pd            |   -0.0356    
   11    | AgPd_0011 | 4       | Ag2Pd2           |   -0.0331    
   12    | AgPd_0012 | 4       | AgPd3            |   -0.0139    
   13    | AgPd_0013 | 4       | Ag3Pd            |   -0.0459    
   14    | AgPd_0014 | 4       | Ag2Pd2           |   -0.0478    
   15    | AgPd_0015 | 4       | AgPd3            |   -0.0201    
   16    | AgPd_0016 | 4       | Ag3Pd            |   -0.0465    
   17    | AgPd_0017 | 4       | Ag2Pd2           |   -0.0498    
   18    | AgPd_0018 | 4       | AgPd3            |   -0.0162    
   19    | AgPd_0019 | 4       | Ag3Pd            |   -0.0369    
   20    | AgPd_0020 | 4       | Ag2Pd2           |   -0.0428    
   21    | AgPd_0021 | 4       | AgPd3            |   -0.0151    
   22    | AgPd_0022 | 4       | Ag3Pd            |   -0.0527    
   23    | AgPd_0023 | 4       | Ag2Pd2           |   -0.0520    
   24    | AgPd_0024 | 4       | AgPd3            |   -0.0292    
   25    | AgPd_0025 | 4       | Ag3Pd            |   -0.0470    
   26    | AgPd_0026 | 4       | AgPd3            |   -0.0238    
   27    | AgPd_0027 | 4       | Ag3Pd            |   -0.0526    
   28    | AgPd_0028 | 5       | AgPd4            |   -0.0152    
   29    | AgPd_0029 | 5       | Ag2Pd3           |   -0.0180    
   30    | AgPd_0030 | 5       | Ag2Pd3           |   -0.0417    
   31    | AgPd_0031 | 5       | Ag4Pd            |   -0.0283    
   32    | AgPd_0032 | 5       | Ag3Pd2           |   -0.0236    
   33    | AgPd_0033 | 5       | Ag3Pd2           |   -0.0547    
   34    | AgPd_0034 | 5       | AgPd4            |   -0.0089    
   35    | AgPd_0035 | 5       | Ag2Pd3           |   -0.0324    
   36    | AgPd_0036 | 5       | Ag2Pd3           |   -0.0341    
   37    | AgPd_0037 | 5       | Ag4Pd            |   -0.0362    
   38    | AgPd_0038 | 5       | Ag3Pd2           |   -0.0471    
   39    | AgPd_0039 | 5       | Ag3Pd2           |   -0.0527    
   40    | AgPd_0040 | 5       | AgPd4            |   -0.0128    
   41    | AgPd_0041 | 5       | Ag2Pd3           |   -0.0379    
   42    | AgPd_0042 | 5       | Ag2Pd3           |   -0.0403    
   43    | AgPd_0043 | 5       | Ag4Pd            |   -0.0396    
   44    | AgPd_0044 | 5       | Ag3Pd2           |   -0.0603    
   45    | AgPd_0045 | 5       | Ag3Pd2           |   -0.0546    
   46    | AgPd_0046 | 5       | AgPd4            |   -0.0135    
   47    | AgPd_0047 | 5       | Ag2Pd3           |   -0.0281    
   48    | AgPd_0048 | 5       | Ag2Pd3           |   -0.0345    
    ...
   624   | AgPd_0505 | 8       | Ag2Pd6           |   -0.0165    
   =================================================================


Training CE parameters
----------------------

Since the :class:`StructureContainer <icet.StructureContainer>` object created
in the previous section, contains all the information required for
constructing a cluster expansion, the next step is to train the parameters,
i.e. to fit the *effective cluster interactions* (:term:`ECIs`) using the
target mixing energies. More precisely, the goal is to achieve the best possible
agreement with a set of training structures, which represent a subset of all
the structures in the :class:`StructureContainer
<trainstation.StructureContainer>`. In practice, this is a two step process
that involves the initiation of an optimizer object (here a
:class:`CrossValidationEstimator <trainstation.CrossValidationEstimator>`)
with a list of target properties produced by the :func:`get_fit_data()
<icet.StructureContainer.get_fit_data>` method of the
:class:`StructureContainer <icet.StructureContainer>` as input argument.

.. literalinclude:: ../../../examples/tutorial/1_construct_cluster_expansion.py
   :start-after: # step 4
   :end-before: # step 5

The :class:`CrossValidationEstimator <trainstation.CrossValidationEstimator>` optimizer used here is intended to provide a reliable estimate for the cross validation score.
This is achieved by calling the :func:`validate <trainstation.CrossValidationEstimator.validate>` method.
With the default settings used here the :class:`CrossValidationEstimator <trainstation.CrossValidationEstimator>` then randomly splits the available data into a training and a validation set.
Here, the effective cluster interactions (:term:`ECIs`) are obtained using the
`LASSO <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_ method (``fit_method``).
This procedure is repeated 10 times (the default value for ``number_of_splits``).

.. note::

   The optimized parameters returned by the optimizer are actually
   *not* the :term:`ECIs` but the :term:`ECIs` times the multiplicity
   of the respective orbit. The distinction is handled internally but
   it is something to be aware of when inspecting the parameters
   directly.

.. note::

   You will likely see a few "Convergence errors" when executing this command.
   For the present purpose these can be ignored.
   They arise since the internal scikit-learn optimization loops have exceeded the number of iterations without reaching the default target accuracy.
   It is possible to extend the iterations but this increases the run time without a marked improvement in the quality of the cluster expansion.

The "final" CE is ultimately constructed using *all* available data by calling the :func:`train <trainstation.CrossValidationEstimator.train>` method.
Once it is finished, the results can be displayed by providing the :class:`CrossValidationEstimator <trainstation.CrossValidationEstimator>` object to the :func:`print` function, which gives the output shown below::

   ============== CrossValidationEstimator ==============
   seed                           : 42
   fit_method                     : ardr
   standardize                    : True
   n_target_values                : 625
   n_parameters                   : 82
   n_nonzero_parameters           : 44
   parameters_norm                : 0.07314093
   target_values_std              : 0.01486413
   rmse_train                     : 0.001632753
   R2_train                       : 0.988
   AIC                            : -7937.285
   BIC                            : -7742.024
   validation_method              : k-fold
   n_splits                       : 10
   rmse_validation                : 0.001889973
   R2_validation                  : 0.9829637
   shuffle                        : True
   ======================================================



We have thus constructed a CE with an average root mean square error (RMSE,
``rmse_validation``) for the validation set of only 1.8 meV/atom. The original
cluster space included 82 parameters (``number_of_parameters``), 44 of which
are non-zero (``number_of_nonzero_parameters``) in the final CE. The efficiency
of the ARDR method for finding sparse solutions is evident from the
number of non-zero parameters (44) being much smaller than the total number of
parameters (82). The performance and application area of different optimization
algorithms are analyzed and compared in the advanced tutorial section.

.. note::

  Here we have used the :class:`CrossValidationEstimator <trainstation.CrossValidationEstimator>` from the :program:`trainstation` package.
  More information can be found in the `documentation of this package <https://trainstation.materialsmodeling.org/>`_.


Finalizing the cluster expansion
--------------------------------

At this point, the task of constructing the cluster expansion is almost
complete. The only step that remains is to tie the parameter values obtained
from the optimization to the cluster space. This is achieved through the
initiation of a :class:`ClusterExpansion <icet.ClusterExpansion>` object using
the previously created :class:`ClusterSpace <icet.ClusterSpace>` instance and
the list of parameters, available via the :class:`parameters
<trainstation.Optimizer.parameters>` attribute of the optimizer, as input arguments.

.. literalinclude:: ../../../examples/tutorial/1_construct_cluster_expansion.py
   :start-after: # step 5

Information regarding the parameters and associated cluster space
can be displayed by using the :func:`print` function with the
:class:`ClusterExpansion <icet.ClusterExpansion>` object as input argument::

   ================================================ Cluster Expansion ================================================
    space group                            : Fm-3m (225)
    chemical species                       : ['Ag', 'Pd'] (sublattice A)
    cutoffs                                : 13.5000 6.5000 6.0000
    total number of parameters             : 82
    number of parameters by order          : 0= 1  1= 1  2= 25  3= 20  4= 35
    fractional_position_tolerance          : 2e-06
    position_tolerance                     : 1e-05
    symprec                                : 1e-05
    total number of nonzero parameters     : 44
    number of nonzero parameters by order  : 0= 1  1= 1  2= 14  3= 13  4= 15 
   -------------------------------------------------------------------------------------------------------------------
   index | order |  radius  | multiplicity | orbit_index | multicomponent_vector | sublattices | parameter |    ECI   
   -------------------------------------------------------------------------------------------------------------------
      0  |   0   |   0.0000 |        1     |      -1     |           .           |      .      |    -0.046 |    -0.046
      1  |   1   |   0.0000 |        1     |       0     |          [0]          |      A      |   -0.0366 |   -0.0366
      2  |   2   |   1.4460 |        6     |       1     |        [0, 0]         |     A-A     |    0.0306 |    0.0051
      3  |   2   |   2.0450 |        3     |       2     |        [0, 0]         |     A-A     |    0.0141 |    0.0047
      4  |   2   |   2.5046 |       12     |       3     |        [0, 0]         |     A-A     |    0.0206 |   0.00171
      5  |   2   |   2.8921 |        6     |       4     |        [0, 0]         |     A-A     |   0.00163 |  0.000272
      6  |   2   |   3.2334 |       12     |       5     |        [0, 0]         |     A-A     |         0 |         0
      7  |   2   |   3.5420 |        4     |       6     |        [0, 0]         |     A-A     |   0.00328 |  0.000819
      8  |   2   |   3.8258 |       24     |       7     |        [0, 0]         |     A-A     |         0 |         0
      9  |   2   |   4.0900 |        3     |       8     |        [0, 0]         |     A-A     |  0.000574 |  0.000191
    ...
     72  |   4   |   2.7940 |       12     |      71     |     [0, 0, 0, 0]      |   A-A-A-A   |  0.000911 |  7.59e-05
     73  |   4   |   2.7940 |       24     |      72     |     [0, 0, 0, 0]      |   A-A-A-A   |         0 |         0
     74  |   4   |   2.8281 |       24     |      73     |     [0, 0, 0, 0]      |   A-A-A-A   |  -0.00278 | -0.000116
     75  |   4   |   2.8921 |        3     |      74     |     [0, 0, 0, 0]      |   A-A-A-A   |         0 |         0
     76  |   4   |   2.8921 |        6     |      75     |     [0, 0, 0, 0]      |   A-A-A-A   |  -0.00067 | -0.000112
     77  |   4   |   2.8921 |       12     |      76     |     [0, 0, 0, 0]      |   A-A-A-A   |  -0.00054 |  -4.5e-05
     78  |   4   |   2.9632 |       24     |      77     |     [0, 0, 0, 0]      |   A-A-A-A   |         0 |         0
     79  |   4   |   2.9862 |        8     |      78     |     [0, 0, 0, 0]      |   A-A-A-A   |   0.00121 |  0.000151
     80  |   4   |   3.0951 |       24     |      79     |     [0, 0, 0, 0]      |   A-A-A-A   |         0 |         0
     81  |   4   |   3.5420 |        2     |      80     |     [0, 0, 0, 0]      |   A-A-A-A   | -0.000936 | -0.000468
   ===================================================================================================================


Note that in the table above the parameters obtained from the
optimizer and the :term:`ECIs` are shown separately, with the
multiplication factor being the multiplicity of the respective orbit.

Finally, the CE is written to file in order to be reused in the
following steps of the tutorial.


Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``examples/tutorial/1_construct_cluster_expansion.py``

    .. literalinclude:: ../../../examples/tutorial/1_construct_cluster_expansion.py
