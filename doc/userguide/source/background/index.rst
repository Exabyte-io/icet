.. _background:
.. index:: Background

Background
**********

In atomic scale materials modeling the alloy cluster expansion (CE) technique
is widely used for predicting the energy as a function of composition. Such
:term:`CEs` are usually trained to match as closely as possible a series of
first-principles calculations and subsequently used to sample configuration
space with Monte Carlo (MC) simulations. In this way, :term:`CEs` can be used
to predict alloy phase diagrams.

More generally, :term:`CEs` provide a mapping between a configuration and a
property of interest. The configuration is not restricted to be atomic but
could for example also represent the sequence of amino acids in a protein
[NelHarZho13]_. Similarly, properties of interest can also be much more
general and include for example migration barriers, lattice constants, or
transport coefficients [AngLinErh16]_.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   cluster_expansions
   workflow
