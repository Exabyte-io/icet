'''
This example demonstrates how to obtain basic information about a cluster
space.
'''
# Start import
from ase.build import bulk

from icetdev import ClusterSpace
from icetdev.cluster_space import get_singlet_info, view_singlets

# End import

# Create a prototype structure, decide which additional elements to populate
# it with (Re, Ti, W and Mo) and set the cutoffs for pairs (10.0 A),
# triplets (7.0 A) and quadruplets (5.0 A).
# Start setup
prototype = bulk('Re')
subelements = ['Re', 'Ti', 'W', 'Mo']
cutoffs = [10.0, 7.0, 5.0]
# End setup

# Generate and print the cluster space.
# Start clusterspace
clusterspace = ClusterSpace(prototype, cutoffs, subelements)
print(clusterspace)
# End clusterspace

# Extract and print additional information regarding the singlets.
# Start singlets
print('\nSinglets:')
cluster_data = get_singlet_info(prototype)
for singlet in cluster_data:
    for key in singlet.keys():
        print(' {:22} : {}'.format(key, singlet[key]))
view_singlets(prototype)
# End singlets
