#include "ClusterSpace.hpp"

/**
Calculate a cluster vector from the structure and  the internal state of the clusterspace.

The first element in the clustervector will always be a constant (1)

The second element in the clustervector is the average of the first orbit.
The third element might be the first orbits second multicomponent vector
or it is the second orbit if it only has one multicomponent vector.

The length of the clustervector will be 1 + sum_i( orbit_i * orbit_i.multicimponentvectors.size() )

*/
std::vector<double> ClusterSpace::generateClustervector(const Structure &structure2) const
{
    Structure structure = structure2;
    //structure.setNumberOfAllowedComponents(_Mi);
    bool orderIntact = true; // count the clusters in the orbit with the same orientation as the prototype cluster
    LocalOrbitlistGenerator localOrbitListGenerator = LocalOrbitlistGenerator(_primitive_orbitlist, structure);
    size_t uniqueOffsets = localOrbitListGenerator.getUniqueOffsetsCount();
    ClusterCounts clusterCounts = ClusterCounts();
    for (int i = 0; i < uniqueOffsets; i++)
    {
        const auto local_orbitlist = localOrbitListGenerator.generateLocalOrbitlist(i);
        clusterCounts.countOrbitlist(structure, local_orbitlist, orderIntact);
    }

    const auto clusterMap = clusterCounts.getClusterCounts();
    std::vector<double> clusterVector;
    clusterVector.push_back(1);
    //Finally begin occupying the clustervector
    for (size_t i = 0; i < _primitive_orbitlist.size(); i++)
    {
        auto repCluster = _primitive_orbitlist.getOrbit(i).getRepresentativeCluster();
        auto allowedOccupations = getAllowedOccupations(_primitive_structure, _primitive_orbitlist.getOrbit(i).getRepresentativeSites());
        auto mcVectors = _primitive_orbitlist.getOrbit(i).getMCVectors(allowedOccupations);
        repCluster.setClusterTag(i);

        for (const auto &mcVector : mcVectors)
        {
            double clusterVectorElement = 0;
            int multiplicity = 0;

            for (const auto &elementsCountPair : clusterMap.at(repCluster))
            {
                clusterVectorElement += getClusterProduct(mcVector, allowedOccupations, elementsCountPair.first) * elementsCountPair.second;
                multiplicity += elementsCountPair.second;
            }

            clusterVectorElement /= ((double)multiplicity);
            clusterVector.push_back(clusterVectorElement);
        }
    }
    return clusterVector;
}

///Returns the native clusters count in this structure, i.e. only clusters inside the unitcell
ClusterCounts ClusterSpace::getNativeClusters(const Structure &structure) const
{
    bool orderIntact = true; // count the clusters in the orbit with the same orientation as the prototype cluster
    LocalOrbitlistGenerator localOrbitListGenerator = LocalOrbitlistGenerator(_primitive_orbitlist, structure);
    size_t uniqueOffsets = localOrbitListGenerator.getUniqueOffsetsCount();
    ClusterCounts clusterCounts = ClusterCounts();

    int tags=0;
    for (int i = 0; i < uniqueOffsets; i++)
    {
        const auto local_orbitlist = localOrbitListGenerator.generateLocalOrbitlist(i);
        for (int j = 0; j < local_orbitlist.size(); j++)
        {
            Cluster repr_cluster = local_orbitlist.getOrbit(j).getRepresentativeCluster();

            for (const auto sites : local_orbitlist.getOrbit(j).getPermutatedEquivalentSites())
            {
                bool cont = true;
                for (auto site : sites)
                {
                    if (site.unitcellOffset().norm() > 0.1)
                    {
                        cont = false;
                    }
                }
                if (!cont)
                {
                    continue;
                }
                repr_cluster.setClusterTag(j);
                if (repr_cluster.order() != 1)
                {
                    std::vector<int> elements(sites.size());
                    for (size_t i = 0; i < sites.size(); i++)
                    {
                        elements[i] = structure.getAtomicNumber(sites[i].index());
                    }
                    clusterCounts.countCluster(repr_cluster, elements);
                }
                else
                {
                    std::vector<int> elements(sites.size());
                    for (size_t i = 0; i < sites.size(); i++)
                    {
                        elements[i] = structure.getAtomicNumber(sites[i].index());
                    }
                    clusterCounts.countCluster(repr_cluster, elements);
                }
            }
        }
    }
    return clusterCounts;
}

/**
This is the default clusterfunction
*/
double ClusterSpace::defaultClusterFunction(const int Mi, const int clusterFunction, const int element) const
{
    // if (clusterFunction == 0)
    // {
    //     return 1.0;
    // }

    if (((clusterFunction + 2) % 2) == 0)
    {
        return -cos(2.0 * M_PI * (double)((int)(clusterFunction + 2) / 2) * (double)element / ((double)Mi));

        // return -cos(2.0 * M_PI * (double) ((int) (clusterFunction + 2) / 2)
        // * (double) element / ((double) Mi));
    }
    else
    {
        return -sin(2.0 * M_PI * (double)((int)(clusterFunction + 2) / 2) * (double)element / ((double)Mi));

        // return -sin(2.0 * M_PI * (double) ((int) (clusterFunction + 2) / 2)
        // * (double) element / ((double) Mi));
    }
}

///Return the full cluster product of entire cluster (elements vector). Assuming all sites have same Mi
double ClusterSpace::getClusterProduct(const std::vector<int> &mcVector, const std::vector<int> &Mi, const std::vector<int> &elements) const
{
    double clusterProduct = 1;
    for (int i = 0; i < elements.size(); i++)
    {
        // std::cout<<Mi[i]<< " "<< (mcVector[i]  )<< " "<< _elementRepresentation.at(elements[i])<< " "<<defaultClusterFunction(Mi[i], mcVector[i], _elementRepresentation.at(elements[i]) )<< std::endl;
        clusterProduct *= defaultClusterFunction(Mi[i], mcVector[i], _elementRepresentation.at(elements[i]));
    }
    return clusterProduct;
}

///Returns the allowed occupations on the sites
std::vector<int> ClusterSpace::getAllowedOccupations(const Structure &structure, const std::vector<LatticeSite> &latticeNeighbors) const
{
    std::vector<int> Mi;
    Mi.reserve(latticeNeighbors.size());
    for (const auto &latnbr : latticeNeighbors)
    {
        Mi.push_back(structure.getNumberOfAllowedComponents(latnbr.index()));
    }
    return Mi;
}

///Setup the clusterspace info
void ClusterSpace::setupClusterspaceInfo()
{
    _clusterSpaceInfo.clear();

    for (int i = 0; i < _primitive_orbitlist.size(); i++)
    {
        auto allowedOccupations = getAllowedOccupations(_primitive_structure, _primitive_orbitlist.getOrbit(i).getRepresentativeSites());
        auto mcVectors = _primitive_orbitlist.getOrbit(i).getMCVectors(allowedOccupations);
        for (const auto &mcVector : mcVectors)
        {
            _clusterSpaceInfo.push_back(std::make_pair(i, mcVector));
        }
    }
    _isClusterspaceInitialized = true;
}

///Get a clusterspace info
std::pair<int, std::vector<int>> ClusterSpace::getClusterSpaceInfo(const unsigned int index)
{
    if (!_isClusterspaceInitialized)
    {
        setupClusterspaceInfo();
    }

    if (index >= _clusterSpaceInfo.size())
    {
        std::string errMSG = "Error: out of range in ClusterSpace::getClusterSpaceInfo " + std::to_string(index) + " >= " + std::to_string(_clusterSpaceInfo.size());
        throw std::out_of_range(errMSG);
    }

    return _clusterSpaceInfo[index];
}

///Returns the clusterspace size i.e. the length of a clustervector
size_t ClusterSpace::getClusterSpaceSize()
{
    if (!_isClusterspaceInitialized)
    {
        setupClusterspaceInfo();
    }

    return _clusterSpaceInfo.size();
}
