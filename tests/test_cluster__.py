import icet as it
import numpy as np

# set up a test cluster
sites = np.array([0, 0, 0])
distances = np.array([.0, 1.0, 3.0])
cluster = it.Cluster(distances=distances, sites=sites, sortedCluster=True)

# TODO :set up a cluster using structure and lattice neigbhors


assert cluster.get_count(
    [0, 0, 0]) == 0, 'if no count has happened, then its zero'

# count
cluster.count([0, 0, 0])

assert cluster.get_count([0, 0, 0]) == 1, 'first count is one'

# test getters (note that the distances are allready sorted otherwise they will
# reorder to lowest form)
assert (cluster.get_distances() == distances).all(), \
    'The distances should be equal {} {}'.format(cluster.get_distances(),
                                                 distances)

assert (cluster.get_sites() == sites).all(), 'The sites should be equal'


# test equality

sites_big = np.array([10, 10, 10])
distances_big = np.array([10, 10, 10])
cluster_big = it.Cluster(sites_big, distances_big)

assert cluster < cluster_big, 'cluster_big is not bigger'

# try distances equal but sites different

sites_slightly_bigger = np.array([11, 10, 10])
distances_slightly_bigger = np.array([10, 10, 10])
cluster_slightly_bigger = it.Cluster(
    sites_slightly_bigger, distances_slightly_bigger)


assert cluster_big < cluster_slightly_bigger, \
    'cluster slightly bigger is not bigger'

# try distances different but sites same

sites_slightly_bigger = np.array([10, 10, 10])
distances_slightly_bigger = np.array([10.1, 10, 10])
cluster_slightly_bigger = it.Cluster(
    sites_slightly_bigger, distances_slightly_bigger)

assert cluster_big < cluster_slightly_bigger

# make sure that the same cluster is not less then
cluster_equal = it.Cluster(sites_slightly_bigger, distances_slightly_bigger)
assert not (cluster_slightly_bigger < cluster_equal)

assert not (cluster_equal < cluster_slightly_bigger)

# make sure that smaller bodied cluster are consider smaller than higher
# bodied clusters

sites_few = np.array([100, 1000])
distances_few = np.array([1000000.0])
cluster_pair = it.Cluster(sites_few, distances_few)
assert cluster_pair < cluster_big


# try sorting

my_clusters = [cluster_big, cluster_slightly_bigger,
               cluster_pair, cluster_equal]
my_clusters.sort()

# check that cluster_pair is at the front
# note this is using pythons own equal method
assert my_clusters[0] == cluster_pair
assert my_clusters[0] != cluster_big
