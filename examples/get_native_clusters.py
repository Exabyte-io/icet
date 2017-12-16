'''
This examples demonstrates how to find the native clusters for a structure
'''

# Start import
from icetdev.cluster_space import ClusterSpace
<<<<<<< HEAD
=======
from icetdev.cluster_counts import ClusterCounts
>>>>>>> 6844742a734f72f18f511b85533342dd47d6efe7
from icetdev.structure import Structure
from ase.build import bulk
import numpy as np
# End import

# Create a prototype structure, decide which additional elements to populate
# it with (Si, Ge) and set the cutoff for pairs (10.0 A)
# Start setup
conf = bulk("Si")
cutoffs = [10.0]
subelements = ["Si", "Ge"]
# End setup

# Generate the cluster space.
# Start clusterspace
clusterspace = ClusterSpace(conf, cutoffs, subelements)
# End clusterspace

<<<<<<< HEAD
# Prepare 2x2x1 supercells, populate these, randomly, with Si and Ge atoms.

# Start supercell
supercell = bulk("Si").repeat([2, 2, 1])
for atom in supercell:
    atom.symbol = np.random.choice(subelements)
structure = Structure.from_atoms(supercell)
# End supercell

# Extract and print the native clusters for the supercell.
# Start native
nativeclusters = clusterspace.get_native_clusters(structure)
print("Native cluster counts for:")
print(structure)
nativeclusters.print()
=======
# Prepare 2x2x1 supercells, populate these, randomly, with Si and Ge atoms,
# then extract and print the native clusters.
# Start native
for i in range(1):
    supercell = bulk("Si").repeat([2, 2, 1])
    for atom in supercell:
        atom.symbol = np.random.choice(subelements)
    structure = Structure.from_atoms(supercell)
    nativeclusters = clusterspace.get_native_clusters(structure)
    print("Native cluster counts for:")
    print(structure)
    nativeclusters.print()
>>>>>>> 6844742a734f72f18f511b85533342dd47d6efe7
# End native
