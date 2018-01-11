'''
This example demonstrates how to obtain basic information about a cluster
space.
'''

# Import modules
from ase.build import bulk
from icet import ClusterSpace, get_singlet_info, view_singlets

# Create a prototype structure, decide which additional elements to populate
# it with (Re, Ti, W and Mo) and set the cutoffs for pairs (10.0 A),
# triplets (7.0 A) and quadruplets (5.0 A).
prototype = bulk('Re')
subelements = ['Re', 'Ti', 'W', 'Mo']
cutoffs = [10.0, 7.0, 5.0]

# Generate and print the cluster space.
cluster_space = ClusterSpace(prototype, cutoffs, subelements)
print(cluster_space)

# Extract and print additional information regarding the singlets.
print('\nSinglets:')
cluster_data = get_singlet_info(prototype)
for singlet in cluster_data:
    for key in singlet.keys():
        print(' {:22} : {}'.format(key, singlet[key]))
view_singlets(prototype)
