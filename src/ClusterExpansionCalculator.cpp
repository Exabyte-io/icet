#include "ClusterExpansionCalculator.hpp"

ClusterExpansionCalculator::ClusterExpansionCalculator(const ClusterSpace &clusterSpace, const Structure &structure)
{
    _clusterSpace = clusterSpace;
    _superCell = structure;

    LocalOrbitListGenerator _theLog = LocalOrbitListGenerator(clusterSpace.getOrbitList(), _superCell);
    size_t uniqueOffsets = _theLog.getNumberOfUniqueOffsets();
    int numberOfOrbits = _clusterSpace._orbitList.size();
    std::vector<Orbit> orbitVector;
    for (const auto orbit : clusterSpace._orbitList._orbits)
    {
        orbitVector.push_back(Orbit(orbit.getRepresentativeCluster()));
    }

    // Permutations for the sites in the orbits
    std::vector<std::vector<std::vector<int>>> permutations(numberOfOrbits);

    /* Strategy for constructing the "full" primitive orbitlists.

    First we fill up a std::vector<Orbit> orbitVector,
    where vector<orbit> is essentially an orbit list.

    The existing method for constructing the _full_ orbit list proceeds
    by looping over all local orbit lists with LocalOrbitListGenerator and
    adding the sites to the local orbit list.

    Now we do something similar by looping over each local orbit list
    (by looping over offsetIndex)
    The local orbitlist is retrieved here:
        `_theLog.getLocalOrbitList(offsetIndex).getOrbits()`

    Then for each orbit `orbitIndex` in `_theLog.getLocalOrbitList(offsetIndex).getOrbits()`
    each group of lattice sites in orbit.equivalentSites() is added to
    orbitVector[orbitIndex] if the lattice sites have a site with offset [0, 0, 0].

    When the full primitive orbitlist is used to create a local orbit list for
    site `index` in the supercell it should thus contain all lattice sites that
    contain `index`.
    */

    for (size_t offsetIndex = 0; offsetIndex < uniqueOffsets; offsetIndex++)
    {
        int orbitIndex = -1;
        // This orbit is a local orbit related to the supercell
        for (const auto orbit : _theLog.getLocalOrbitList(offsetIndex).getOrbits())
        {
            orbitIndex++;

            auto orbitPermutations = orbit.getPermutationsOfEquivalentSites();

            int eqSiteIndex = -1;

            for (const auto latticeSites : orbit.getEquivalentSites())
            {
                eqSiteIndex++;

                std::vector<LatticeSite> primitiveEquivalentSites;
                for (const auto site : latticeSites)
                {
                    Vector3d sitePosition = _superCell.getPosition(site);
                    auto primitiveSite = _clusterSpace.getPrimitiveStructure().findLatticeSiteByPosition(sitePosition);
                    primitiveEquivalentSites.push_back(primitiveSite);
                }
                std::vector<std::vector<LatticeSite>> latticeSitesTranslated = _clusterSpace._orbitList.getSitesTranslatedToUnitcell(primitiveEquivalentSites, false);

                for (auto latticesitesPrimTrans : latticeSitesTranslated)
                {
                    if (std::any_of(latticesitesPrimTrans.begin(), latticesitesPrimTrans.end(), [=](LatticeSite ls) { return (ls.unitcellOffset()).norm() < 1e-4; }))
                    {
                        // false or true here seems to not matter
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

    // Now create the full primitive orbit list using the vector<orbit>
    _fullPrimitiveOrbitList.setPrimitiveStructure(_clusterSpace.getPrimitiveStructure());
    int orbitIndex = -1;
    for (auto orbit : orbitVector)
    {
        orbitIndex++;
        _fullPrimitiveOrbitList.addOrbit(orbit);
    }

    // Calculate the permutation for each orbit in this orbit list.
    // This is normally done in the constructor but since we made one manually
    // we have to do this ourself.
    _fullPrimitiveOrbitList.addPermutationInformationToOrbits(_clusterSpace.getOrbitList().getFirstColumnOfPermutationMatrix(),
                                                              _clusterSpace.getOrbitList().getPermutationMatrix());

    _primToSupercellMap.clear();
    _indexToOffset.clear();

    // Precompute all possible local orbitlists for this supercell and map it to the offset
    for (size_t i = 0; i < structure.size(); i++)
    {
        Vector3d localPosition = structure.getPositions().row(i);
        LatticeSite localSite = _clusterSpace.getPrimitiveStructure().findLatticeSiteByPosition(localPosition);
        Vector3d offsetVector = localSite.unitcellOffset();
        _indexToOffset[i] = offsetVector;

        if (_localOrbitlists.find(offsetVector) == _localOrbitlists.end())
        {
            _localOrbitlists[offsetVector] = _fullPrimitiveOrbitList.getLocalOrbitList(structure, offsetVector, _primToSupercellMap);

            // Set eq sites equal to the permuted sites so no permutation is required in the orbit list counting.
            /// @todo If one replaces the reference to the internal _orbits member of OrbitList with getOrbits(),
            /// multiple tests fails (for ternaries). This needs to be fixed.
            for (auto &orbit : _localOrbitlists[offsetVector]._orbits)
            {
                auto permutedSites = orbit.getPermutedEquivalentSites();
                orbit._equivalentSites = permutedSites;
            }
        }
    }
}

/**
@details This constructs a cluster vector that only considers clusters that contain the input index.
@param occupations the occupation vector for the supercell
@param index the local index of the supercell
@param ignoredIndices a vector of indices which have already had their local energy calculated. This is required to input so that no double counting occurs.
*/
std::vector<double> ClusterExpansionCalculator::getLocalClusterVector(const std::vector<int> &occupations,
								      int index,
                                                                      std::vector<size_t> ignoredIndices)
{
    _superCell.setAtomicNumbers(occupations);

    if (occupations.size() != _superCell.size())
    {
        throw std::runtime_error("Input occupations and internal supercell structure mismatch in size");
    }

    for (auto ignoreIndex : ignoredIndices)
    {
        if (ignoreIndex >= _superCell.size())
        {
            throw std::runtime_error("Index larger than input structure size in method ClusterExpansionCalculator::getLocalClusterVector");
        }
    }

    // do not sort the clusters
    bool orderIntact = true;

    // count the clusters in the order they lie in equivalent sites
    // since these sites are already in the permuted order
    bool permuteSites = false;

    // Remove all sites in the orbits that do not contain index regardless of offset?
    bool onlyConsiderZeroOffsetNotContain = true;

    // Remove all sites in the orbits ignored indices regardless of offset?
    bool onlyConsiderZeroOffsetContain = false;

    ClusterCounts clusterCounts = ClusterCounts();

    // Get one of the translated orbitlists
    OrbitList translatedOrbitList = _localOrbitlists[_indexToOffset[index]];

    // Remove sites not containing the local index
    if (_clusterSpace.getPrimitiveStructure().size() > 1)
    {
        translatedOrbitList.removeSitesNotContainingIndex(index, onlyConsiderZeroOffsetNotContain);
    }

    // Purge the orbitlist of all sites containing the ignored indices
    for (auto ignoredIndex : ignoredIndices)
    {
        translatedOrbitList.removeSitesContainingIndex(ignoredIndex, onlyConsiderZeroOffsetContain);
    }

    // Count clusters and get cluster count map
    clusterCounts.countOrbitList(_superCell, translatedOrbitList, orderIntact, permuteSites);

    const auto clusterMap = clusterCounts._clusterCounts;

    // Finally begin occupying the cluster vector
    std::vector<double> clusterVector;
    clusterVector.push_back(1.0 / _superCell.size());
    for (size_t i = 0; i < _fullPrimitiveOrbitList.size(); i++)
    {
        Cluster repCluster = _fullPrimitiveOrbitList._orbits[i]._representativeCluster;
        std::vector<int> allowedOccupations;

        if (i >= _clusterSpace._orbitList.size())
        {
            std::cout << _fullPrimitiveOrbitList.size() << " >= " << _clusterSpace._orbitList.size() << std::endl;
            throw std::runtime_error("Index i larger than cs.orbit_list.size() in ClusterExpansionCalculator::getLocalClusterVector");
        }
        try
        {
            allowedOccupations = _clusterSpace.getNumberOfAllowedSpeciesBySite(_clusterSpace.getPrimitiveStructure(), _clusterSpace._orbitList._orbits[i].getRepresentativeSites());
        }
        catch (const std::exception &e)
        {
            std::cout << e.what() << std::endl;
            throw std::runtime_error("Failed getting allowed occupations in generateClusterVector");
        }

        // Skip the rest if any of the sites are inactive (i.e. allowed occupation < 2)
        if (std::any_of(allowedOccupations.begin(), allowedOccupations.end(), [](int allowedOccupation) { return allowedOccupation < 2; }))
        {
            continue;
        }
        auto representativeSites = _clusterSpace._orbitList._orbits[i].getRepresentativeSites();
        std::vector<int> representativeSitesIndices;
        for (const auto site : representativeSites)
        {
            representativeSitesIndices.push_back(site.index());
        }

        const auto &mcVectors = _clusterSpace._multiComponentVectors[i];
        repCluster.setTag(i);

        /// Loop over all multi component vectors for this orbit
        for (size_t currentMCVectorIndex = 0; currentMCVectorIndex < _clusterSpace._multiComponentVectors[i].size(); currentMCVectorIndex++)
        {
            double clusterVectorElement = 0;

            auto clusterFind = clusterMap.find(repCluster);

            /// Push back zero if nothing was counted for this orbit
            if (clusterFind == clusterMap.end())
            {
                clusterVector.push_back(0);
                continue;
            }

            /// Loop over all the counts for this orbit
            for (const auto &elementsCountPair : clusterMap.at(repCluster))
            {
                /// Loop over all equivalent permutations for this orbit and mc vector
                for (const auto &perm : _clusterSpace._sitePermutations[i][currentMCVectorIndex])
                {
                    /// Permute the mc vector and the allowed occupations
                    const auto &permutedMCVector = icet::getPermutedVector(mcVectors[currentMCVectorIndex], perm);
                    const auto &permutedAllowedOccupations = icet::getPermutedVector(allowedOccupations, perm);
                    const auto &permutedRepresentativeIndices = icet::getPermutedVector(representativeSitesIndices, perm);

                    clusterVectorElement += _clusterSpace.evaluateClusterProduct(permutedMCVector, permutedAllowedOccupations, elementsCountPair.first, permutedRepresentativeIndices) * elementsCountPair.second;
                }
            }

            /// This is the multiplicity one would have gotten during a full cluster vector calculation and is needed as normalizing factor
            double realMultiplicity = (double)_clusterSpace._sitePermutations[i][currentMCVectorIndex].size() * (double)_clusterSpace._orbitList._orbits[i]._equivalentSites.size() / (double)_clusterSpace.getPrimitiveStructure().size();
            clusterVectorElement /= ((double)realMultiplicity * (double)_superCell.size());
            clusterVector.push_back(clusterVectorElement);
        }
    }
    return clusterVector;
}
