'''
This examples demonstrates how to construct cluster vectors.
'''

# Start import
from icetdev.cluster_space import ClusterSpace
from ase.build import bulk
from ase import Atoms
# End import

# Create a prototype structure, decide which additional elements to populate
# it with (Si, Ge) and set the cutoffs for pairs (5.0 A), triplets (5.0 A)
# and quadruplets (5.0 A).
# Start setup
conf = bulk("Si")
cutoffs = [5.0, 5.0, 5.0]
subelements = ["Si", "Ge"]
# End setup

# Generate and print the cluster space.
# Start clusterspace
clusterspace = ClusterSpace(conf, cutoffs, subelements)
print(clusterspace)
# End clusterspace


# Generate and print the cluster vector for a pure Si 2x2x2 supercell.
# Start clustervector1
supercell = bulk("Si").repeat(2)
cv = clusterspace.get_clustervector(supercell)
print(cv)
# End clustervector1

# Generate and print the cluster vector for a mixed Si-Ge 2x2x2 supercell
# Start clustervector2
supercell_2 = bulk("Si").repeat(2)
supercell_2[0].symbol = "Ge"
cv_2 = clusterspace.get_clustervector(supercell_2)
print(cv_2)
# End clustervector2
