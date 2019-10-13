.. index::
   single: Function reference; ClusterSpace
   single: Class reference; ClusterSpace

Cluster space
=============

Setting up a cluster space
---------------------------


The cluster space serves as a basis for cluster expansions. Setting up a clusterspace object is commonly done via::

    cs = ClusterSpace(atoms, cutoffs=[7.0, 5.0], chemical_symbols=['Si', 'Ge'])

The cutoffs are set up to include pairs with an interatomic distance
smaller than 7 Å and triplets for which all pair-wise interatomic
distance are smaller than 5 Å. Here, Si and Ge are allowed to occupy
the reference lattice.


Sublattices
```````````

A cluster space can also be constructed with multiple
sublattices. Consider for example a rocksalt lattice, on which we want
to allow mixing of Na and Li on the Na sublattice and Cl and F on the
Cl sublattices. This can be achived by::

    from ase.build import bulk
    atoms = bulk('NaCl', 'rocksalt', a=4.0)
    chemical_symbols = [['Na', 'Li'], ['Cl', 'F']]
    cs = ClusterSpace(atoms, [7.0, 5.0], chemical_symbols)

where ``chemical_symbols`` now specifies which speicies are allowed
for each lattice site in atoms.

Inactive sites
``````````````

The sublattice functionality also allows one to have inactive
sites. For example, if we consider the system above but would like to
keep the Cl lattice fixed it can be achived via::

    chemical_symbols = [['Na', 'Li'], ['Cl']]
    cs = ClusterSpace(atoms, [7.0, 5.0], chemical_symbols)



2D systems
``````````

:program:`icet` requires input structures to have periodic boundary
conditions (PBCs).  In order to treat two-dimensional systems, or more
generally without PBCs in at least one direction, one has to surround
the prototype structure with vacuum and then apply PBCs in all
directions. This can easily be achived with the ASE functions::

    from ase.build import surface
    atoms = surface('Cu', (1,1,1), layers=7)

which creates ase atoms object without PBC in the z-direction. The
structure can be modified to have PBC in the z-direction with vacuum
via::

    atoms.pbc = [True, True, True]
    atoms.center(vacuum=20, axis=2)

which can now be used to create a cluster space.


Interface
---------

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
