from _icet import ClusterCounts
from .local_orbit_list_generator import LocalOrbitListGenerator


def __count_clusters(self, structure, prim_orbit_list,
                     keep_order_intact=False):
    '''
    Count all clusters in a structure by finding their local orbit list.

    Note that this requires creating the local_orbit_list_generator object for
    generating the local orbit lists.

    Parameters
    ----------
    structure : Structure object
        supercell of the structure `prim_orbit_list` is based on
    prim_orbit_list : OrbitList object
        orbit list based on a primitive of the input structure
    keep_order_intact: boolean
        count the clusters in the orbit with the same orientation as the
        prototype cluster
    '''

    local_orbit_list_generator = LocalOrbitListGenerator(prim_orbit_list,
                                                         structure)

    for i in range(local_orbit_list_generator.get_unique_offsets_count()):
        # sending local orbit list directly into function was about 10% faster
        # than:
        # local_orbit_list = \
        #    local_orbit_list_generator.generate_local_orbit_list(i)
        self.count_orbit_list(
            structure,
            local_orbit_list_generator.generate_local_orbit_list(i),
            keep_order_intact)


ClusterCounts.count_clusters = __count_clusters
