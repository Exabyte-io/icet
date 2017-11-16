from _icetdev import ClusterCounts
from icetdev.local_orbitlist_generator import LocalOrbitlistGenerator


def __count_clusters(self, structure, prim_orbitlist, keep_order_intact=False):
    '''
    Count all clusters in a structure by finding their local orbitlist.

    This will need to create the local_orbit_list_generator object for
    generating the local orbitlists.

    Parameters
    ----------
    structure : icet structure object
        supercell of the structure `prim_orbitlist` is based on
    prim_orbitlist : icet orbitlist object
        based on a primitive of the input structure
    keep_order_intact: bool
        count the clusters in the orbit with the same orientation as the
        prototype cluster
    '''

    local_orbitlist_generator = LocalOrbitlistGenerator(prim_orbitlist,
                                                        structure)

    for i in range(local_orbitlist_generator.get_unique_offsets_count()):
        # sending local orbitlist directly into function was about 10% faster
        # than:
        # local_orbitlist = \
        #    local_orbit_list_generator.generate_local_orbitlist(i)
        self.count_orbitlist(
            structure,
            local_orbitlist_generator.generate_local_orbitlist(i),
            keep_order_intact)


ClusterCounts.count_clusters = __count_clusters
