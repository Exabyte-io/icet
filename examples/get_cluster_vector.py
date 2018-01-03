'''
This example demonstrates how to construct cluster vectors.
'''
# Import modules
from ase.build import bulk

from icetdev.cluster_space import ClusterSpace

# Create a prototype structure, decide which additional elements to populate
# it with (Si, Ge) and set the cutoffs for pairs (5.0 Å), triplets (5.0 Å)
# and quadruplets (5.0 Å).
conf = bulk("Si")
cutoffs = [5.0, 5.0, 5.0]
subelements = ["Si", "Ge"]

# Initiate and print the cluster space.
cluster_space = ClusterSpace(conf, cutoffs, subelements)
print(cluster_space)

# Generate and print the cluster vector for a pure Si 2x2x2 supercell.
supercell_1 = bulk("Si").repeat(2)
cluster_vector_1 = cluster_space.get_cluster_vector(supercell_1)
print(cluster_vector_1)

# Generate and print the cluster vector for a mixed Si-Ge 2x2x2 supercell
supercell_2 = bulk("Si").repeat(2)
supercell_2[0].symbol = "Ge"
cluster_vector_2 = cluster_space.get_cluster_vector(supercell_2)
print(cluster_vector_2)
