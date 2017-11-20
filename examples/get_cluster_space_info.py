'''
This scripts shows how to otbain basic information about a cluster space.
'''

from icetdev import ClusterSpace
from icetdev.cluster_space import get_singlet_info, view_singlets
from ase.build import bulk

cutoffs = [10.0, 7.0, 5.0]
subelements = ['Re', 'Ti', 'W', 'Mo']
prototype = bulk('Re')

clusterspace = ClusterSpace(prototype, cutoffs, subelements)
print(clusterspace)

print('\nSinglets:')
cluster_data = get_singlet_info(prototype)
for singlet in cluster_data:
    for key in singlet.keys():
        print(' {:22} : {}'.format(key, singlet[key]))

view_singlets(prototype)
