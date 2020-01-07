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
prim_structure = bulk('Ti', 'sc', a=3.0)
structure = prim_structure.repeat([2, 1, 1])
structure.set_chemical_symbols(['Ti', 'W'])
cutoffs = [5.0]
# End setup

# Determine the orbit list for the corresponding primitive structure for all
# pair clusters within the cutoff distance
symprec = 1e-5  # tolerance used by spglib
position_tolerance = 1e-5  # tolerance used when comparing positions
fractional_position_tolerance = position_tolerance / 3  # ... in fractional coordinates
prim_orbitlist = OrbitList(prim_structure, cutoffs, symprec,
                           position_tolerance, fractional_position_tolerance)
# Use the primitive orbit list to count the number of clusters.
cluster_counts = ClusterCounts(prim_orbitlist, structure, fractional_position_tolerance)
# Print all of the clusters that were found.
print('Number of atoms: {0}'.format(len(structure)))
print('Found {} orbits'.format(len(cluster_counts)))
print(cluster_counts)
