#include "ClusterCounts.hpp"

/// Count clusters given this compact form of lattice neighbors (see ManyBodyNeighborList for more details)
// build(const NeighborList &nl, int index, int order, bool);
void ClusterCounts::countLatticeSites(const Structure &structure,
                                      const std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> &latticeNeighbors)
{
    for (const auto &neighborPair : latticeNeighbors)
    {
        //Now we have std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>
        //pair.first == the base indices and pair.second is all indices that form clusters with the base indices
        if (neighborPair.second.size() > 0)
        {
            for (const auto &combinationIndice : neighborPair.second)
            {
                auto latticePointsForCluster = neighborPair.first;
                latticePointsForCluster.push_back(combinationIndice);
                count(structure, latticePointsForCluster);
            }
        }
        else
        {
            //count singlets here
            count(structure, neighborPair.first);
        }
    }
}
/**
The simplest form of counting clusters using the mbnl format

Get the indice of one set of indices and counts this
*/
void ClusterCounts::count(const Structure &structure,
                          const std::vector<LatticeSite> &latticeNeighbors)
{
    size_t clusterSize = latticeNeighbors.size();
    std::vector<int> elements(clusterSize);
    for (size_t i = 0; i < latticeNeighbors.size(); i++)
    {
        elements[i] = structure._atomicNumbers[latticeNeighbors[i].index()];
    }

    // Don't do intact order since there is no reason for it
    Cluster cluster = Cluster(structure, latticeNeighbors);
    countCluster(cluster, elements, false);
}

/**
@details Will count the vectors in latticeNeighbors and assuming these sets of sites are represented by the cluster 'cluster'.
@param structure the structure that will have its clusters counted
@param latticeSites A group of sites, represented by 'cluster', that will be counted
@param cluster A cluster used as identification on what sites the clusters belong to
@param orderIntact if true the order of the sites will stay the same otherwise the vector of species being counted will be sorted
*/
void ClusterCounts::count(const Structure &structure, const std::vector<std::vector<LatticeSite>> &latticeSites,
                          const Cluster &cluster, bool orderIntact)
{
    std::vector<int> elements(latticeSites[0].size());
    for (const auto &sites : latticeSites)
    {
        for (size_t i = 0; i < sites.size(); i++)
        {
            elements[i] = structure._atomicNumbers[sites[i].index()];
        }
        countCluster(cluster, elements, orderIntact);
    }
}

///Count cluster only through this function
void ClusterCounts::countCluster(const Cluster &cluster, const std::vector<int> &elements, bool orderIntact)
{
    if (orderIntact)
    {
        _clusterCounts[cluster][elements] += 1;
    }
    else
    {
        std::vector<int> sortedElements = elements;
        std::sort(sortedElements.begin(), sortedElements.end());
        _clusterCounts[cluster][sortedElements] += 1;
    }
}

/**
 @brief Counts the clusters in the input structure.
 @param structure input configuration
 @param orbitList orbit list
 @param orderIntact if true do not reorder clusters before comparison (i.e., ABC != ACB)
 @param permuteSites if true the sites will be permuted according to the correspondin permutations in the orbit
*/
void ClusterCounts::countOrbitList(const Structure &structure, const OrbitList &orbitList, bool orderIntact, bool permuteSites)
{
    for (size_t i = 0; i < orbitList.size(); i++)
    {
        Cluster repr_cluster = orbitList._orbits[i].getRepresentativeCluster();
        repr_cluster.setTag(i);
        if (permuteSites && orderIntact && repr_cluster.order() != 1)
        {
            count(structure, orbitList.getOrbit(i).getPermutedEquivalentSites(), repr_cluster, orderIntact);
        }
        else if (!permuteSites && orderIntact && repr_cluster.order() != 1)
        {
            count(structure, orbitList._orbits[i]._equivalentSites, repr_cluster, orderIntact);
        }
        else
        {
            count(structure, orbitList._orbits[i]._equivalentSites, repr_cluster, orderIntact);
        }
    }
}


