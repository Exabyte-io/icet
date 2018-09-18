#include "ClusterExpansionCalculator.hpp"

ClusterExpansionCalculator::ClusterExpansionCalculator(const ClusterSpace &clusterSpace, const Structure &structure)
{
    _clusterSpace = clusterSpace;
    _superCell = structure;
    _basisAtomOrbitList.clear();
    // Set up stuff.
    _theLog = LocalOrbitListGenerator(clusterSpace.getOrbitList(), _superCell);
    size_t uniqueOffsets = _theLog.getNumberOfUniqueOffsets();
    int numberOfOrbits = _clusterSpace._orbitList.size();
    Vector3d zeroOffset = {0.0, 0.0, 0.0};
    int primitiveSize = _clusterSpace.getPrimitiveStructure().size();
    std::vector<Orbit> orbitVector;

    for (const auto orbit : clusterSpace._orbitList._orbitList)
    {
        orbitVector.push_back(Orbit(orbit.getRepresentativeCluster()));
    }

    std::cout << "Primitive size: " << primitiveSize << std::endl;
    // Permutations for the sites in the orbits
    std::vector<std::vector<std::vector<int>>> permutations(numberOfOrbits);
    OrbitList basisOrbitList = OrbitList();
    /* Strategy to construct the "full" primitive orbitlists
    
    We first fill up a std::vector<std::vector<Orbit>> orbitVector
    where the inner vector<orbit> is essentially an orbitlist.
    the outer vector is over basis atoms. When the entire vector<vector<orbit>> 
    is constructed we create a vector<orbitlists>

    The existing methods to construct the full orbitlist is to loop over all the
    unique offsets.

    We loop over each local orbitlist (by looping over offsetIndex)
    The local orbitlist is retrieved here:
        `_theLog.getLocalOrbitList(offsetIndex).getOrbitList()`

    Then for each group of latticesites in orbit.equivalentSites() we add them
    to orbitVector[i][orbitIndex] if the latticesites has the basis atom `i` in them.

    */
    for (int offsetIndex = 0; offsetIndex < uniqueOffsets; offsetIndex++)
    {
        int orbitIndex = -1;
        // This orbit is a local orbit related to the supercell
        for (const auto orbit : _theLog.getLocalOrbitList(offsetIndex).getOrbitList())
        {
            orbitIndex++;

            auto orbitPermutations = orbit.getEquivalentSitesPermutations();

            int eqSiteIndex = -1;

            for (const auto latticeSites : orbit.getEquivalentSites())
            {
                eqSiteIndex++;

                for (int i = 0; i < primitiveSize; i++)
                {
                    Vector3d primPos = _clusterSpace.getPrimitiveStructure().getPositions().row(i);
                    LatticeSite primitiveSite_i = _clusterSpace.getPrimitiveStructure().findLatticeSiteByPosition(primPos);
                    LatticeSite superCellEquivalent = _superCell.findLatticeSiteByPosition(primPos);

                    std::vector<LatticeSite> primitiveEquivalentSites;
                    for (const auto site : latticeSites)
                    {
                        Vector3d sitePosition = _superCell.getPosition(site);
                        auto primitiveSite = _clusterSpace.getPrimitiveStructure().findLatticeSiteByPosition(sitePosition);
                        primitiveEquivalentSites.push_back(primitiveSite);
                    }

                    std::vector<std::vector<LatticeSite>> latticeSitesTranslated = _clusterSpace._orbitList.getSitesTranslatedToUnitcell(primitiveEquivalentSites);
                    for (auto latticesitesPrimTrans : latticeSitesTranslated)
                    {

                        auto eqSites = orbitVector[orbitIndex].getEquivalentSites();
                        if (std::any_of(latticesitesPrimTrans.begin(), latticesitesPrimTrans.end(), [=](LatticeSite ls) { return (ls.unitcellOffset() - zeroOffset).norm() < 1e-4; }))
                        {
                            if (!orbitVector[orbitIndex].contains(latticesitesPrimTrans, true))
                            {
                                orbitVector[orbitIndex].addEquivalentSites(latticesitesPrimTrans);
                                permutations[orbitIndex].push_back(orbitPermutations[eqSiteIndex]);
                            }
                        }
                    }
                }
            }
        }
    }

    OrbitList orbitList = OrbitList();
    orbitList.setPrimitiveStructure(_clusterSpace.getPrimitiveStructure());
    int orbitIndex = -1;
    for (auto &orbit : orbitVector)
    {
        orbitIndex++;
        orbit.setEquivalentSitesPermutations(permutations[orbitIndex]);
        orbitList.addOrbit(orbit);
    }
    _basisAtomOrbitList.push_back(orbitList);
    validateBasisAtomOrbitLists();
    for(int i =0; i < structure.size(); i++)
    {
        Vector3d localPosition = structure.getPositions().row(i);
        LatticeSite localSite = _clusterSpace.getPrimitiveStructure().findLatticeSiteByPosition(localPosition);
        Vector3d offsetVector = localSite.unitcellOffset();

        if(_localOrbitlists.find(offsetVector) == _localOrbitlists.end())
        {
            _localOrbitlists[offsetVector] = _basisAtomOrbitList[0].getLocalOrbitList(structure, offsetVector, _primToSupercellMap);        
        }
    }

}

std::vector<double> ClusterExpansionCalculator::getLocalClusterVector(const Structure &structure, int index, std::vector<int> ignoredIndices)
{

    if (structure.size() != _superCell.size())
    {
        throw std::runtime_error("Input structure and internal supercell structure mismatch in size");
    }

    if (index >= structure.size())
    {
        throw std::runtime_error("index larger than Input structure size in method ClusterExpansionCalculator::getLocalClusterVector");
    }

    for (auto ignoreIndex : ignoredIndices)
    {
        if (ignoreIndex >= structure.size())
        {
            throw std::runtime_error("index larger than Input structure size in method ClusterExpansionCalculator::getLocalClusterVector");
        }
    }

    int dprint = 0;
    bool orderIntact = true; // count the clusters in the orbit with the same orientation as the prototype cluster

    ClusterCounts clusterCounts = ClusterCounts();

    Vector3d localPosition = structure.getPositions().row(index);

    LatticeSite localSite = _clusterSpace.getPrimitiveStructure().findLatticeSiteByPosition(localPosition);

    // int basisIndex = localSite.index();
    int basisIndex = 0;
    if (basisIndex >= _basisAtomOrbitList.size())
    {
        throw std::runtime_error("basisIndex and _basisAtomOrbitList are not the same size");
    }

    for (const auto orbit : _basisAtomOrbitList[basisIndex]._orbitList)
    {
        for (const auto eqSites : orbit.getEquivalentSites())
        {
            for (const auto site : eqSites)
            {
                if (site.index() > _basisAtomOrbitList[basisIndex].getPrimitiveStructure().size())
                {
                    throw std::runtime_error("lattice site index in orbit is larger than prim.size()");
                }
                if (site.index() > _clusterSpace.getPrimitiveStructure().size())
                {
                    throw std::runtime_error("lattice site index in orbit is larger than prim.size()");
                }
            }
        }
    }

    // Calc offset for this site

    Vector3d positionVector = structure.getPositions().row(localSite.index()) - structure.getPositions().row(0);
    Vector3d localIndexZeroPos = localPosition - positionVector;
    auto indexZeroLatticeSite = _clusterSpace.getPrimitiveStructure().findLatticeSiteByPosition(localIndexZeroPos);

    // Vector3d offsetVector = indexZeroLatticeSite.unitcellOffset();
    Vector3d offsetVector = localSite.unitcellOffset();
    
    
    // if(_localOrbitlists.find(offsetVector) == _localOrbitlists.end())
    // {
    //     _localOrbitlists[offsetVector] = _basisAtomOrbitList[0].getLocalOrbitList(structure, offsetVector, _primToSupercellMap);        
    // }

    auto translatedOrbitList = _localOrbitlists[offsetVector];
    

    // std::cout << "Ignored indices:  size = " << ignoredIndices.size() << " indices: ";
    for (auto ignoredIndex : ignoredIndices)
    {
        translatedOrbitList.removeSitesContainingIndex(ignoredIndex);
    }
    for (const auto orbit : translatedOrbitList._orbitList)
    {
        for (const auto sites : orbit.getEquivalentSites())
        {
            for (const auto site : sites)
            {
                for (const auto ignoredIndex : ignoredIndices)
                {
                    if (site.index() == ignoredIndex)
                    {
                        throw std::runtime_error("lattice site was not removed");
                    }
                }
            }
        }
    }
 
    clusterCounts.countOrbitList(structure, translatedOrbitList, orderIntact);

    const auto clusterMap = clusterCounts.getClusterCounts();
    std::vector<double> clusterVector;
    clusterVector.push_back(1.0 / structure.size());
    // Finally begin occupying the cluster vector
    for (size_t i = 0; i < _basisAtomOrbitList[basisIndex].size(); i++)
    {
        Cluster repCluster = _basisAtomOrbitList[basisIndex]._orbitList[i].getRepresentativeCluster();
        std::vector<int> allowedOccupations;

        if (i >= _clusterSpace._orbitList.size())
        {
            std::cout << _basisAtomOrbitList[basisIndex].size() << " >= " << _clusterSpace._orbitList.size() << std::endl;
            throw std::runtime_error("index i larger than cs.orbit_list.size() in ClusterExpansionCalculator::getLocalClusterVector");
        }
        try
        {
            allowedOccupations = _clusterSpace.getNumberOfAllowedSpeciesBySite(_clusterSpace.getPrimitiveStructure(), _clusterSpace._orbitList._orbitList[i].getRepresentativeSites());
        }
        catch (const std::exception &e)
        {
            std::cout << e.what() << std::endl;
            throw std::runtime_error("Failed getting allowed occupations in genereteClusterVector");
        }

        // Skip rest if any sites aren't active sites (i.e. allowed occupation < 2)
        if (std::any_of(allowedOccupations.begin(), allowedOccupations.end(), [](int allowedOccupation) { return allowedOccupation < 2; }))
        {
            continue;
        }

        auto mcVectors = _clusterSpace._orbitList._orbitList[i].getMultiComponentVectors(allowedOccupations);
        auto allowedPermutationsSet = _clusterSpace._orbitList._orbitList[i].getAllowedSitesPermutations();
        auto elementPermutations = _clusterSpace.getMultiComponentVectorPermutations(mcVectors, i);
        repCluster.setTag(i);
        int currentMCVectorIndex = 0;
        for (const auto &mcVector : mcVectors)
        {
            double clusterVectorElement = 0;
            int multiplicity = 0;

            // std::cout<<"do clusterMap.at(repCluster) "<<std::endl;
            // repCluster.print();

            auto clusterFind = clusterMap.find(repCluster);

            if (clusterFind == clusterMap.end())
            {
                clusterVector.push_back(0);
                continue;
            }
            for (const auto &elementsCountPair : clusterMap.at(repCluster))
            {

                // TODO check if allowedOccupations should be permuted as well.
                for (const auto &perm : elementPermutations[currentMCVectorIndex])
                {
                    auto permutedMCVector = icet::getPermutedVector(mcVector, perm);
                    auto permutedAllowedOccupations = icet::getPermutedVector(allowedOccupations, perm);
                    clusterVectorElement += _clusterSpace.evaluateClusterProduct(permutedMCVector, permutedAllowedOccupations, elementsCountPair.first) * elementsCountPair.second;
                    multiplicity += elementsCountPair.second;
                }
            }
            int realMultiplicity = _clusterSpace._orbitList._orbitList[i].getEquivalentSites().size();
            clusterVectorElement /= ((double)structure.size() * realMultiplicity);
            //  clusterVectorElement /= ((double)structure.size());
            clusterVector.push_back(clusterVectorElement * repCluster.order());
            // clusterVector.push_back(clusterVectorElement);

            currentMCVectorIndex++;
        }
    }
    return clusterVector;
}

// std::vector<double> getLocalClusterVector(const Structure &, const std::vector<int>)

void ClusterExpansionCalculator::validateBasisAtomOrbitLists()
{

    for (const auto orbitList : _basisAtomOrbitList)
    {
        for (const auto orbit : orbitList._orbitList)
        {
            for (int i = 0; i < orbit.getEquivalentSites().size(); i++)
            {
                for (int j = i + 1; j < orbit.getEquivalentSites().size(); j++)
                {
                    auto sites_i = orbit.getEquivalentSites()[i];
                    std::sort(sites_i.begin(), sites_i.end());
                    auto sites_j = orbit.getEquivalentSites()[j];
                    std::sort(sites_j.begin(), sites_j.end());
                    if (std::equal(sites_i.begin(), sites_i.end(), sites_j.begin()))
                    {
                        throw std::runtime_error("Two eq. sites in an orbit were equal.");
                    }
                }
            }
        }
    }
}