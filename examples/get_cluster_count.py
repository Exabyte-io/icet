'''
This example demonstrates how to count the number of clusters for a structure.
'''

# Start import
from ase.build import bulk
from icet import Structure
from icet.tools import get_primitive_structure
from icet.core.cluster_counts import ClusterCounts
from icet.core.orbit_list import create_orbit_list
# End import

# Create a titanium, single-layered, sheet and randomly populate some of the
# sites with W atoms.
# Start setup
atoms = bulk('Ti', 'bcc', a=3.43).repeat([2, 2, 1])
atoms.pbc = [True, True, False]
atoms.set_chemical_symbols(['Ti', 'W', 'W', 'Ti'])
# End setup

# Determine the orbit list for the corresponding primitive structure for all
# pair clusters within the cutoff distance
cutoffs = [4.0]
prim_atoms = get_primitive_structure(atoms)
prim_structure = Structure.from_atoms(prim_atoms)
prim_orbitlist = create_orbit_list(prim_structure, cutoffs)

# Use the primitive orbit list to count the number of clusters.
cluster_counts = ClusterCounts()
structure = Structure.from_atoms(atoms)
cluster_counts.count_clusters(structure, prim_orbitlist)

# Print all of the clusters that were found.
print('Number of atoms {0}'.format(len(atoms)))
print('Found {} clusters'.format(len(cluster_counts)))
cluster_counts.print()
