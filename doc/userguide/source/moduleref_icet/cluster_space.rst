.. index::
   single: Function reference; ClusterSpace
   single: Class reference; ClusterSpace

Cluster space
=============

Setting up a cluster space
---------------------------


The cluster space serves as a basis for a cluster expansions.
Setting up a clusterspace object is commonly done via::

    cs = ClusterSpace(atoms, cutoffs=[7.0, 5.0], chemical_symbols=['Si', 'Ge'])

where the first element in the cutoffs list means to include pairs with an interatomic distance smaller than 7Å, and the second element to consider triplets for which all pair-wise interatomic distance are smaller than 5Å.
The chemical_symbols defines which elements are allowed on the lattice.


Sublattices
```````````
A ClusterSpace can also be constructed with multiple sublattices. For example consider a rocksalt structure, NaCl,
where we'd like to mix Na with Li on the Na sites and Cl with F on the Cl sites.  This can be achived by::

    from ase.build import bulk
    atoms = bulk('NaCl', 'rocksalt', a=4.0)
    chemical_symbols = [['Na', 'Li'], ['Cl', 'F']]
    cs = ClusterSpace(atoms, [7.0, 5.0], chemical_symbols)

where the chemical_symbols now specifies which speicies are allowed for each lattice site in atoms.

Inactive sites
``````````````
The sublattice functionality also allows one to have inactive sites. For example if we consider the system above but would like to keep the Cl lattice fixed it can be achived via::

    chemical_symbols = [['Na', 'Li'], ['Cl']]
    cs = ClusterSpace(atoms, [7.0, 5.0], chemical_symbols)



2D - systems
`````````````
The cluster space requires your input structure to have periodic boundary conditions (pbc).
In order to treat 2D systems, or systems in general without pbc, you thus have to embedded the structure in a cell containing vacuum and then applying pbc. This can easily be achived with the ase function::

  atoms.center()




Documentation
-------------

.. module:: icet

.. autoclass:: ClusterSpace
   :members:
   :undoc-members:
   :inherited-members:

Supplementary functions
-----------------------
      
.. module:: icet.core.cluster_space

.. automethod:: icet.core.cluster_space.get_singlet_info

.. automethod:: icet.core.cluster_space.view_singlets
