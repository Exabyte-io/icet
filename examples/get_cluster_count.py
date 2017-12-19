'''
This example demonstrates how to count the number of clusters for a structure.
'''
# Start import
import numpy as np

from ase.build import bulk

from icetdev.cluster_counts import ClusterCounts
from icetdev.orbit_list import create_orbit_list
from icetdev.permutation_map import __get_primitive_structure
from icetdev.structure import Structure

# End import

# Create a titanium, single-layered, sheet and randomly populate some of the
# sites with W atoms.
# Start setup
atoms = bulk("Ti", "bcc", a=3.43).repeat([2, 2, 1])
atoms.pbc = [True, True, False]

atoms.set_chemical_symbols([['Ti', 'W'][n] for n in
                            np.round(np.random.random((len(atoms),
                                                       ))).astype(int)])
# End setup

# Determine the orbitlist for the corresponding primitive structure for all
# pair clusters withhin the cutoff distance (4 angstrom).
# Start orbitlist
cutoffs = [4]
prim_atoms = __get_primitive_structure(atoms)
prim_structure = Structure.from_atoms(prim_atoms)
prim_orbitlist = create_orbit_list(prim_structure, cutoffs)
# End orbitlist

# Use the primitive orbitlist to count the number of clusters.
# Start counting
structure = Structure.from_atoms(atoms)
cluster_counts = ClusterCounts()
cluster_counts.count_clusters(structure, prim_orbitlist)
# End counting

# Print all of the clusters that were found.
# Start results
print("number of atoms {0}".format(len(atoms)))
print("Found {} clusters".format(len(cluster_counts)))
cluster_counts.print()
# End results
