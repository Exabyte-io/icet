.. _generate-alloy-database:
.. index::
   single: Construct a clustervector
   single: Tutorials; construction of clustervector

Construction of a clustervector 
===================================

This example demonstrates the generation of a clustervector.

Step 1: Prepare your clusterspace  
----------------------------------
To be able to construct a clustervector first you need to define the clusterspace, i.e. what clusters and subelements will be considered
when mapping a crystal structure into a clustervector. In icet the clusterspace object keeps track of all these definitions.
of this clusterspace. First we want to define the lattice. Here we use
ASE atoms build module to create a bulk system of silicon. 

.. code-block:: python

    conf = bulk("Si")


Next we want to limit the number of allowed 
components we want to consider. This can be done by defining all the subelements that should be considered,

.. code-block:: python

    subelements = ["Si", "Ge"]

Next we want to constrain the geometrical size of the clusters that will be included. 
This is defined as a list of cutoffs, the length of the cutoffs i.e. how many cutoffs you
enter in the list indicates how many bodies in a cluster to consider, first cutoff is for the pairs then triplet, quatuplet and so on.
We want to create a clusterspace up to quatuplets where each has a cutoff of 5.0 Å:

.. code-block:: python

    cutoffs = [5.0, 5.0, 5.0]

Now we have enough information to create our clusterspace,

.. code-block:: python

    clusterspace = create_clusterspace(conf, cutoffs, subelements)


To summarize here are the final example script needed to setup a basic clusterspace

.. literalinclude:: ../../../examples/get_clustervector.py
   :language: python
   :start-after: # Step 1
   :end-before: # Step 1.1

To get information about the clusters included we could simple print the clusterspace

.. code-block:: python

    print(clusterspace)
    
    Clusterspace
    Subelements: SiGe
    Cutoffs: 5.0 5.0 5.0
    Total number of dimensions 21
    Cluster bodies : cluster radius : multiplicity : (clusterspace index : orbit index)  : mc vector
    -------------------------------------
     1 : 0.0000 : 2 : (0 : 0) : [0]
     2 : 1.1756 : 4 : (1 : 1) : [0, 0]
     2 : 1.9198 : 12 : (2 : 2) : [0, 0]
     2 : 2.2512 : 12 : (3 : 3) : [0, 0]
     3 : 1.6166 : 12 : (4 : 4) : [0, 0, 0]
     3 : 2.0671 : 48 : (5 : 5) : [0, 0, 0]
     3 : 2.2168 : 8 : (6 : 6) : [0, 0, 0]
     3 : 2.2168 : 8 : (7 : 7) : [0, 0, 0]
     3 : 2.4725 : 12 : (8 : 8) : [0, 0, 0]
     3 : 2.4725 : 24 : (9 : 9) : [0, 0, 0]
     4 : 1.8160 : 8 : (10 : 10) : [0, 0, 0, 0]
     4 : 1.9825 : 24 : (11 : 11) : [0, 0, 0, 0]
     4 : 2.1392 : 24 : (12 : 12) : [0, 0, 0, 0]
     4 : 2.1591 : 24 : (13 : 13) : [0, 0, 0, 0]
     4 : 2.2512 : 12 : (14 : 14) : [0, 0, 0, 0]
     4 : 2.3502 : 24 : (15 : 15) : [0, 0, 0, 0]
     4 : 2.3502 : 24 : (16 : 16) : [0, 0, 0, 0]
     4 : 2.3513 : 2 : (17 : 17) : [0, 0, 0, 0]
     4 : 2.3513 : 2 : (18 : 18) : [0, 0, 0, 0]
     4 : 2.4174 : 48 : (19 : 19) : [0, 0, 0, 0]
     4 : 2.5525 : 8 : (20 : 20) : [0, 0, 0, 0]




Step 2: Get a clustervector
----------------------------------
When a clusterspace has been initialized it can find the clustervector from a structure based on the same lattice and subelements.

Let's again construct a bulk crystal structure but this time let it contain a few more atoms,

.. code-block:: python

    supercell = bulk("Si").repeat(2)

This gives us a 2x2x2 pure Si crystal structure.

Now a clustervetor can be retrieved by:

.. code-block:: python

    cv = clusterspace.get_clustervector(supercell)

    print(cv)

    
    [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

The first element in a clustervector is always a 1. To know what the subsequent elements correspond to we can look at the 
clusterspace information we got when printing. The first cluster is made from only one site in the lattice and is this the singlet and
will end up on the second element in the clustervector. This has a value of -1 meaning that Si has a "element spin" of -1. Next are three pairs, here the values 
are 1.0 since these will all be the spin squared. The remaining elements in the clustervectors are -1 since these have their spin cubed.


Let's insert a germanium in the supercell and get the clustervector:

.. literalinclude:: ../../../examples/get_clustervector.py
   :language: python
   :start-after: # Step 3
   

Resulting with

.. code-block:: python

    [1.0, -0.875, 0.75, 0.75, 0.75, -0.625, -0.625, -0.625, -0.625, -0.625, -0.625, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]


Now the singlet became -0.875 since the supercell is a 2x2x2 with 2 atoms in the basis since

.. math::

    \left ( -1\sum Si  + \sum Ge \right ) / 16 = (-15+1)/16 = -0.875.


Similarly we can understand the 0.75 for pairs by imagining that a site can, for each pair, form x bonds. All Si-Si and Ge-Ge bonds will
contribute with 1/multicplicty to the clustervector-element. Thus the site that changed from one Si to a Ge will create x Si-Ge bonds out of a total x * 2x2x2 bonds (no double counting a bond).
Thus:

.. math::

    (((8-1)x - x)/8x = 0.75


Entire example script:
----------------------------------

.. literalinclude:: ../../../examples/get_clustervector.py
   :language: python

