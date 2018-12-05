"""
This example demonstrates how to count the number of clusters for a structure.
"""

# Start import
from ase.build import bulk
from icet.core.cluster_counts import ClusterCounts
from icet import OrbitList
# End import

# Create a titanium, single-layered, sheet and randomly populate some of the
# sites with W atoms.
# Start setup
prim_atoms = bulk('Ti', 'sc', a=3.0)
atoms = prim_atoms.repeat([2, 1, 1])
atoms.set_chemical_symbols(['Ti', 'W'])
cutoffs = [5.0]
# End setup

# Determine the orbit list for the corresponding primitive structure for all
# pair clusters within the cutoff distance
prim_orbitlist = OrbitList(prim_atoms, cutoffs)
# Use the primitive orbit list to count the number of clusters.
cluster_counts = ClusterCounts(prim_orbitlist, atoms)
# Print all of the clusters that were found.
print('Number of atoms: {0}'.format(len(atoms)))
print('Found {} orbits'.format(len(cluster_counts)))
print(cluster_counts)
