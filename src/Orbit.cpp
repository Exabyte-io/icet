#include "Orbit.hpp"

/**
@details This constructor creates an orbit from a set of equivalent clusters and a structure.
         Note that the sites of each cluster need to be ordered in a consistent manner
         (This ordering is enforced when Orbit objects are created via an OrbitList object)
@param clusters
    The clusters that together make up the orbit. Normally, these should be symmetry equivalent,
    but note that this needs to be ensured before constructing the orbit; it is not enforced by
    the constructor.
@param allowedClusterPermutations
    Allowed permutations for this orbit; e.g., if 0, 2, 1 is in this set
    then 0, 1, 0 is the same multi-component vector as 0, 0, 1
**/
Orbit::Orbit(const std::vector<Cluster> clusters,
             const std::set<std::vector<int>> allowedClusterPermutations)
{
    _clusters = clusters;
    _allowedClusterPermutations = allowedClusterPermutations;

    // Compute multi-component vectors for this orbit.
    // First get number of allowed species per site in the representative cluster.
    std::vector<int> numberOfAllowedSpecies = representativeCluster().getNumberOfAllowedSpeciesPerSite();

    // The orbit is only active if all of its sites have two or more allowed occupations.
    if (std::none_of(numberOfAllowedSpecies.begin(), numberOfAllowedSpecies.end(), [](const int i)
                     { return i < 2; }))
    {
        _active = true;
        auto distinctMCVectors = _getDistinctMultiComponentVectors(numberOfAllowedSpecies);
        auto sitePermutations = _getMultiComponentVectorPermutations(distinctMCVectors);

        // Save information in a ClusterVectorElement and append it to
        // the list of such ClusterVectorElements.
        // Note that in binary systems there will only be one
        // ClusterVectorElement in each orbit.
        // In ternary and higher-order systems each orbit generally
        // corresponds to multiple cluster vector elements, since there
        // are multiple distinct multi-component vectors for such systems.
        for (size_t j = 0; j < distinctMCVectors.size(); j++)
        {
            ClusterVectorElement cvElement = {distinctMCVectors[j],
                                              sitePermutations[j],
                                              sitePermutations[j].size() * size()};
            _clusterVectorElements.push_back(cvElement);
        }
    }
    else
    {
        _active = false;
    }
}

/**
@brief Add a cluster to the orbit.
@details Note that this function only appends the new cluster to the end without resorting.
@param cluster Cluster to be added represented by a group of lattice sites
*/
void Orbit::addCluster(const Cluster &cluster)
{
    _clusters.push_back(cluster);
}

/**
  @details This method returns the multi-component vector permutations for
  each multi-component vector.

  Example 1: Given multi-component vectors [0, 0], [0, 1] and [1, 1]
  the returned permutations should be [[1, 0]], [[0, 1],[1, 0]], [1, 1].
  i.e. the [0, 1] multi-component vector should count elements with
  permutations [1, 0] and [1, 0].

  Example 2: Given multi-component vectors [0, 0], [0, 1], [1, 0] and [1, 1]
  the returned permutations will only be the self permutations since the
  multi-component vectors [0, 1] and [1, 0] will handle the AB vs BA choice.

  @param multiComponentVectors multi-component vectors for this orbit

  @returns a vector of a vector of a vector of ints; here the innermost index
*/
std::vector<std::vector<std::vector<int>>> Orbit::_getMultiComponentVectorPermutations(const std::vector<std::vector<int>> &multiComponentVectors) const
{
    std::vector<std::vector<std::vector<int>>> elementPermutations;
    std::vector<int> selfPermutation;
    for (size_t i = 0; i < multiComponentVectors[0].size(); i++)
    {
        selfPermutation.push_back(i);
    }

    for (const auto &mc : multiComponentVectors)
    {
        std::vector<std::vector<int>> mcPermutations;
        mcPermutations.push_back(selfPermutation);
        std::vector<std::vector<int>> takenPermutations;
        takenPermutations.push_back(selfPermutation);
        for (const std::vector<int> &perm : _allowedClusterPermutations)
        {
            auto permutedMultiComponentVector = icet::getPermutedVector(mc, perm);
            auto findPerm = find(multiComponentVectors.begin(), multiComponentVectors.end(), permutedMultiComponentVector);
            auto findIfTaken = find(takenPermutations.begin(), takenPermutations.end(), permutedMultiComponentVector);
            if (findPerm == multiComponentVectors.end() && findIfTaken == takenPermutations.end() && mc != permutedMultiComponentVector)
            {
                mcPermutations.push_back(perm);
                takenPermutations.push_back(permutedMultiComponentVector);
            }
        }
        sort(mcPermutations.begin(), mcPermutations.end());
        elementPermutations.push_back(mcPermutations);
    }
    return elementPermutations;
}

/**
@brief Compute all distinct multi-component vectors for this orbit.
@param numberOfAllowedSpeciesPerSite
    The number of species that are allowed on each site of the cluster.
@return std::vector<std::vector<int>>
 */
std::vector<std::vector<int>> Orbit::_getDistinctMultiComponentVectors(const std::vector<int> &numberOfAllowedSpeciesPerSite) const
{
    std::vector<std::vector<int>> cartesianFactors(numberOfAllowedSpeciesPerSite.size());

    // Each element in cartesianFactors corresponds to a site in the cluster,
    // and each such element is a list of the permissible point function
    // indices for that site.
    for (size_t i = 0; i < numberOfAllowedSpeciesPerSite.size(); i++)
    {
        // Use -1 so a binary only gets one cluster function
        for (int j = 0; j < numberOfAllowedSpeciesPerSite[i] - 1; j++)
        {
            cartesianFactors[i].push_back(j);
        }
    }

    // Get all candidate multi-component vectors as all Cartesian products
    // over the permissible point function indices.
    std::vector<std::vector<int>> allPossibleMCPermutations;
    std::vector<int> firstVector(numberOfAllowedSpeciesPerSite.size(), 0);
    do
    {
        allPossibleMCPermutations.push_back(firstVector);
    } while (icet::nextCartesianProduct(cartesianFactors, firstVector));
    std::sort(allPossibleMCPermutations.begin(), allPossibleMCPermutations.end());

    // Filter out all multi-component vectors that are just permutations
    // of other multi-component vectors (given the permutations allowed
    // for this orbit).
    std::vector<std::vector<int>> distinctMCVectors;
    for (const auto &mcVector : allPossibleMCPermutations)
    {
        // Construct all permutations of the current multi-component vector
        // given the permutations allowed for this orbit.
        std::vector<std::vector<int>> permutedMCVectors;
        for (const auto &allowedPermutation : _allowedClusterPermutations)
        {
            permutedMCVectors.push_back(icet::getPermutedVector<int>(mcVector, allowedPermutation));
        }
        std::sort(permutedMCVectors.begin(), permutedMCVectors.end());

        // If none of the vectors in permutedMCVectors exist in distinctMCVectors,
        // we can add a new multi-component vector
        if (std::none_of(permutedMCVectors.begin(), permutedMCVectors.end(), [&](const std::vector<int> &permMcVector)
                         { return !(std::find(distinctMCVectors.begin(), distinctMCVectors.end(), permMcVector) == distinctMCVectors.end()); }))
        {
            distinctMCVectors.push_back(mcVector);
        }
    }
    return distinctMCVectors;
}

/**
@details
    Check if this orbit contains a cluster in its list of clusters.
    Note that order matters; if the input cluster is a shuffled
    version of a cluster in this orbit, this function will
    return false.
@param cluster cluster (represented by a list of sites) to look for
@returns true if the cluster is present in the orbit
**/
bool Orbit::contains(const std::vector<LatticeSite> cluster) const
{
    auto clusterCopy = cluster;
    for (size_t i = 0; i < _clusters.size(); i++)
    {
        auto sites = _clusters[i].latticeSites();
        if (sites == clusterCopy)
        {
            return true;
        }
    }
    return false;
}

/**
@brief Count the occupations of the clusters in this orbit.
@details
    Note that the orderings of the sites in the clusters matter, meaning,
    for example, that (47, 79) will be counted separately from (79, 47)
    (here 47 and 79 are atomic numbers).
@param structure the structure that will have its clusters counted
@param siteIndexForDoubleCountingCorrection
   In small supercells, clusters may include both a site and its periodic image.
   In such cases this argument can be used to avoid double counting.
   Clusters in which a site with this index occurs more than once will only be counted with
   a factor 1/n, where n is the number of occurrences of this index. By default
   (i.e. siteIndexForDoubleCountingCorrection = -1) no such correction is applied.
*/
std::map<std::vector<int>, double> Orbit::getClusterCounts(const std::shared_ptr<Structure> structure,
                                                           int siteIndexForDoubleCountingCorrection) const
{
    std::map<std::vector<int>, double> tmpCounts;
    std::vector<int> elements(order());
    for (const auto &cluster : _clusters)
    {

        // If we apply the double counting correction for some site we need
        // to ensure that we only count clusters that include this site.
        if (siteIndexForDoubleCountingCorrection < 0 || cluster.isSiteIndexIncludedWithZeroOffset(siteIndexForDoubleCountingCorrection))
        {
            const std::vector<LatticeSite> &sites = cluster.latticeSites();
            for (size_t i = 0; i < sites.size(); i++)
            {
                elements[i] = structure->getAtomicNumbers().at(sites[i].index());
            }
            // If the current atom (siteIndexForDoubleCountingCorrection) occurs more than once,
            // we risk double counting it if we calculate a local cluster vector. To avoid this,
            // we count the clusters in units of 1 / n, where n is the number of occurences of
            // the present atom in the cluster.
            double unit = 1;
            if (siteIndexForDoubleCountingCorrection > -1)
            {
                unit /= cluster.getCountOfOccurencesOfSiteIndex(siteIndexForDoubleCountingCorrection);
            }
            tmpCounts[elements] += unit;
        }
    }
    return tmpCounts;
}

/**
@brief
    Count the change in occupations of the clusters in this orbit caused
    by changing the chemical identity of one site.
@details
    `structure` should contain the original occupations, and the change
    is defined by `flipIndex` (index of the site whose occupation
    changes) and `newOccupation` (the new atomic number on that site).
    Note that the orderings of the sites in the clusters matter, meaning,
    for example, that (47, 79) will be counted separately from (79, 47)
    (here 47 and 79 are atomic numbers).
@param structure the structure for which to count clusters, with occupations before change
@param flipIndex index of site that has been flipped
@param newOccupation new atomic number of site that has been flipped
*/
std::map<std::vector<int>, double> Orbit::getClusterCountChanges(const std::shared_ptr<Structure> structure,
                                                                 const int flipIndex,
                                                                 const int newOccupation) const
{
    std::map<std::vector<int>, double> tmpCounts;
    std::vector<int> elementsOld(order());
    std::vector<int> elementsNew(order());
    int siteIndex;
    int occupation;

    for (const auto &cluster : _clusters)
    {
        // Only count clusters where site flipIndex is included (with zero offset)
        if (cluster.isSiteIndexIncludedWithZeroOffset(flipIndex))
        {
            const std::vector<LatticeSite> &sites = cluster.latticeSites();
            for (size_t i = 0; i < sites.size(); i++)
            {
                siteIndex = sites[i].index();
                occupation = structure->getAtomicNumbers().at(siteIndex);
                elementsOld[i] = occupation;

                // If the present site is the one that was changed,
                // we need to use a different atomic number
                if (siteIndex == flipIndex)
                {
                    elementsNew[i] = newOccupation;
                }
                else
                {
                    elementsNew[i] = occupation;
                }
            }
            // If the current site (flipIndex) occurs more than once,
            // we risk double counting it. To avoid this, we count the clusters in units of
            // 1 / n, where n is the number of occurrences of the present atom in the cluster.
            double unit = 1.0 / (double)cluster.getCountOfOccurencesOfSiteIndex(flipIndex);

            // The old cluster has disappeared and we got elementNew instead
            tmpCounts[elementsOld] -= unit;
            tmpCounts[elementsNew] += unit;
        }
    }
    return tmpCounts;
}

/**
@brief Translates the clusters of this orbit by an offset.
@param cellOffset Offset to translate with.
**/
void Orbit::translate(const Vector3i &cellOffset)
{
    for (auto &cluster : _clusters)
    {
        cluster.translate(cellOffset);
    }
}

/**
@brief Transforms the clusters to a new cell.
@details
    Each cluster in the orbit consists of a list of lattice sites, and
    these sites are defined in relation to a specific atomic structure
    (typically a primitive structure), and a pointer to this structure
    is stored in each cluster. This function redefines the sites
    such that they refer to a new cell (typically a supercell).
@param supercell The new atomic structure
@param primitiveToSupercellMap
    Map from lattice site referring to old structure to lattice site
    referring to the new structure. This map will successivelly be
    populated when executing the function, and is only used for
    reasons of performance.
@param fractionalPositionTolerance
    Tolerance applied when compariong fractional coordinates.
**/
void Orbit::transformToSupercell(std::shared_ptr<Structure> supercell,
                                 std::unordered_map<LatticeSite, LatticeSite> &primitiveToSupercellMap,
                                 const double fractionalPositionTolerance)
{
    for (auto &cluster : _clusters)
    {
        cluster.transformToSupercell(supercell,
                                     primitiveToSupercellMap,
                                     fractionalPositionTolerance);
    }
}

/**
@brief The += operator merges the r.h.s. orbit into this orbit, which means that the clusters of the former will be added to the this orbit.
*/
Orbit &Orbit::operator+=(const Orbit &orbit_rhs)
{
    // Get representative sites
    auto rep_sites_rhs = orbit_rhs.representativeCluster().latticeSites();
    auto rep_sites_this = representativeCluster().latticeSites();

    // Check that order is the same
    if (rep_sites_this.size() != rep_sites_rhs.size())
    {
        throw std::runtime_error("The orbit operands must have the same order (Orbit &operator+=)");
    }

    // Insert rhs clusters
    const auto rhsClusters = orbit_rhs.clusters();
    _clusters.insert(_clusters.end(), rhsClusters.begin(), rhsClusters.end());

    // Update multiplicities
    for (size_t i = 0; i < _clusterVectorElements.size(); i++)
    {
        _clusterVectorElements[i].multiplicity = _clusterVectorElements[i].sitePermutations.size() * size();
    }

    return *this;
}

namespace std
{
    /// Stream operator.
    ostream &operator<<(ostream &os, const Orbit &orbit)
    {
        for (const auto &cluster : orbit.clusters())
        {
            os << "  ";
            for (const auto &site : cluster.latticeSites())
            {
                os << " " << site;
            }
        }
        return os;
    }

}
