#include "OrbitList.hpp"

/**
@details This constructor generates an orbit list for the given (supercell) structure from a set of neighbor lists and a matrix of (symmetry) equivalent sites.
@param structure Structure that the orbits will be based on
@param matrixOfEquivalentSites matrix of symmetry equivalent sites
@param neighborLists neighbor lists for each (cluster) order (0=pairs, 1=triplets etc)
@param positionTolerance tolerance applied when comparing positions in Cartesian coordinates
**/
OrbitList::OrbitList(const Structure &structure,
                     const std::vector<std::vector<LatticeSite>> &matrixOfEquivalentSites,
                     const std::vector<std::vector<std::vector<LatticeSite>>> &neighborLists,
                     const double positionTolerance)
{
    if (!structure.hasAllowedAtomicNumbers())
    {
        std::ostringstream msg;
        msg << "OrbitList must be initialized with a structure with the species allowed on each site specified.";
        throw std::runtime_error(msg.str());
    }
    _structure = structure;
    _matrixOfEquivalentSites = matrixOfEquivalentSites;
    _referenceLatticeSites = getReferenceLatticeSites();

    /**
    The following list is used to compile "raw data" for the orbit list.
    The first index (vector) runs over the orbits, the second index (vector) over the
    equivalent cluster in a given orbit, and the final vector runs over the lattice sites
    that represent a particular cluster. **/
    std::vector<std::vector<std::vector<LatticeSite>>> listOfEquivalentClusters;

    // rows that have already been accounted for
    std::unordered_set<std::vector<int>, VectorHash> rowsTaken;

    ManyBodyNeighborList mbnl = ManyBodyNeighborList();

    // check that there are no duplicates in the first column of the matrix of equivalent sites
    std::set<LatticeSite> uniqueReferenceLatticeSites(_referenceLatticeSites.begin(), _referenceLatticeSites.end());
    if (_referenceLatticeSites.size() != uniqueReferenceLatticeSites.size())
    {
        std::ostringstream msg;
        msg << "Found duplicates in the list of reference lattice sites (= first column of matrix of equivalent sites): ";
        msg << std::to_string(_referenceLatticeSites.size()) << " != " << std::to_string(uniqueReferenceLatticeSites.size());
        msg << " (OrbitList::OrbitList)";
        throw std::runtime_error(msg.str());
    }

    for (size_t index = 0; index < neighborLists[0].size(); index++)
    {
        std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> mbnlLatticeSites = mbnl.build(neighborLists, index, false);
        for (const auto &mbnlPair : mbnlLatticeSites)
        {
            for (const auto &latticeSite : mbnlPair.second)
            {
                // complete cluster by combining the first and the second part of the MBNL pair
                std::vector<LatticeSite> cluster = mbnlPair.first;
                cluster.push_back(latticeSite);

                // check that original sites are sorted
                auto copyOfCluster = cluster;
                std::sort(copyOfCluster.begin(), copyOfCluster.end());
                if (copyOfCluster != cluster)
                {
                    throw std::runtime_error("Original sites are not sorted (OrbitList::OrbitList)");
                }

                // Get all translational variants of cluster, in order to be able to extract
                // the "lowest" one according to how lattice sites are sorted
                std::vector<std::vector<LatticeSite>> clusterWithTranslations = getSitesTranslatedToUnitcell(cluster);

                // get all sites from the matrix of equivalent sites
                auto pairOfSitesAndRowIndices = getMatchesInMatrixOfEquivalentSites(clusterWithTranslations)[0];

                if (rowsTaken.find(pairOfSitesAndRowIndices.second) == rowsTaken.end())
                {
                    // Found new stuff
                    addColumnsFromMatrixOfEquivalentSites(listOfEquivalentClusters, rowsTaken, pairOfSitesAndRowIndices.second);
                }
            }

            // special singlet case
            if (mbnlPair.second.size() == 0)
            {
                std::vector<LatticeSite> cluster = mbnlPair.first;
                auto indices = getReferenceLatticeSiteIndices(cluster);
                auto find = rowsTaken.find(indices);
                if (find == rowsTaken.end())
                {
                    // Found new stuff
                    addColumnsFromMatrixOfEquivalentSites(listOfEquivalentClusters, rowsTaken, indices);
                }
            }
        }
    }

    // Sort list of equivalent clusters
    for (size_t i = 0; i < listOfEquivalentClusters.size(); i++)
    {
        std::sort(listOfEquivalentClusters[i].begin(), listOfEquivalentClusters[i].end());
    }

    // Add orbits from list of equivalent clusters to this orbit list
    for (const auto &equivalentClusters : listOfEquivalentClusters)
    {
        addOrbit(createOrbit(equivalentClusters));
    }

    // Sort the orbit list by order and radius.
    sort(positionTolerance);
}

/**
@brief Sort the orbit list
@details
    This function sorts the orbit list by (1) order, (2) radius,
    (3) number of clusters in the orbit, and (4) coordinates of
    the sites in the clusters. This produces a reproducable
    (stable) order of the orbit list (and thereby the cluster vector).
@param positionTolerance tolerance applied when comparing positions in Cartesian coordinates
*/
void OrbitList::sort(const double positionTolerance)
{
    std::sort(_orbits.begin(), _orbits.end(),
              [positionTolerance](const Orbit &lhs, const Orbit &rhs)
              {
                  // (1) Test against number of bodies in cluster.
                  if (lhs.representativeCluster().order() != rhs.representativeCluster().order())
                  {
                      return lhs.representativeCluster().order() < rhs.representativeCluster().order();
                  }
                  // (2) Compare by radius.
                  if (fabs(lhs.radius() - rhs.radius()) > positionTolerance)
                  {
                      return lhs.radius() < rhs.radius();
                  }

                  // (3) Compare by the number of clusters in the orbit.
                  if (lhs.size() < rhs.size())
                  {
                      return true;
                  }
                  if (lhs.size() > rhs.size())
                  {
                      return false;
                  }

                  // (4) Check the individual sites.
                  return lhs.clusters() < rhs.clusters();
              });
}

/**
@details Adds an orbit the this orbit list.
@param orbit Orbit to add.
**/
void OrbitList::addOrbit(const Orbit &orbit)
{
    _orbits.push_back(orbit);
}

/**
@details Returns reference to the orbit at the given index.
@param index index of orbit
@returns reference to orbit
**/
const Orbit &OrbitList::getOrbit(unsigned int index) const
{
    if (index >= size())
    {
        throw std::out_of_range("Tried accessing orbit at out of bound index (Orbit OrbitList::getOrbit)");
    }
    return _orbits[index];
}

/**
@details
Permutes the sites in a set of equivalent clusters (such that the ordering of the sites
is consistent with the first cluster), then creates an orbit based on the permuted clusters.

Algorithm
---------

For each orbit:

1. Take representative cluster
2. Find the rows, which match the sites that belong to this cluster, and ...
3. Get all columns for these rows, i.e the sites that are directly equivalent, call these equivalentClustersWithTranslations.
4. Construct all possible permutations for the representative cluster, call these representativeClusterWithTranslationsAndPermutations.
5. Construct the intersection of p_equal and p_all, call this consistentEquivalentClustersWithTranslations.
6. Get the index version of consistentEquivalentClustersWithTranslations and these are then the allowed permutations for this orbit.
7. Take the clusters in the orbit:
    if site exists in p_all:
        those cluster are then related to representative_cluster via the permutation
    else:
        loop over permutations of the clusters:
            if the permutation exists in p_all:
                that permutation is then related to repr_cluster through that permutation
            else:
                continue

**/
Orbit OrbitList::createOrbit(const std::vector<std::vector<LatticeSite>> &equivalentClusters)
{

    bool sortRows = false;

    // step one: Get representative cluster
    std::vector<LatticeSite> sitesOfRepresentativeCluster = equivalentClusters[0];
    auto representativeClusterWithTranslations = getSitesTranslatedToUnitcell(sitesOfRepresentativeCluster, sortRows);

    // step two: Find the rows these sites belong to and
    // step three: Get all columns for these rows
    std::vector<std::vector<LatticeSite>> equivalentClustersWithTranslations;
    for (auto reprCluster : representativeClusterWithTranslations)
    {
        auto equivClusters = getSymmetryRelatedSiteGroups(reprCluster);
        equivalentClustersWithTranslations.insert(equivalentClustersWithTranslations.end(), equivClusters.begin(), equivClusters.end());
    }
    std::sort(equivalentClustersWithTranslations.begin(), equivalentClustersWithTranslations.end());

    // Step four: Construct all possible permutations of the representative cluster
    std::vector<std::vector<LatticeSite>> representativeClusterWithTranslationsAndPermutations;
    for (auto reprCluster : representativeClusterWithTranslations)
    {
        std::vector<std::vector<LatticeSite>> permClusters = icet::getAllPermutations<LatticeSite>(reprCluster);
        representativeClusterWithTranslationsAndPermutations.insert(representativeClusterWithTranslationsAndPermutations.end(), permClusters.begin(), permClusters.end());
    }
    std::sort(representativeClusterWithTranslationsAndPermutations.begin(), representativeClusterWithTranslationsAndPermutations.end());

    // Step five: Construct intersection of equivalentClustersWithTranslations and
    // representativeClusterWithTranslationsAndPermutations. This will
    // generate the list of equivalent clusters that is consistent with the
    // permutations of the representative cluster. This is relevant for
    // systems with more than two components, for which one must deal with
    // multi-component vectors.
    std::vector<std::vector<LatticeSite>> consistentEquivalentClustersWithTranslations;
    std::set_intersection(equivalentClustersWithTranslations.begin(), equivalentClustersWithTranslations.end(),
                          representativeClusterWithTranslationsAndPermutations.begin(), representativeClusterWithTranslationsAndPermutations.end(),
                          std::back_inserter(consistentEquivalentClustersWithTranslations));

    // Step six: Get the index version of consistentEquivalentClustersWithTranslations
    std::set<std::vector<int>> allowedPermutations;
    for (const auto &equivCluster : consistentEquivalentClustersWithTranslations)
    {
        size_t failedLoops = 0;
        for (auto reprCluster : representativeClusterWithTranslations)
        {
            try
            {
                std::vector<int> allowedPermutation = icet::getPermutation<LatticeSite>(reprCluster, equivCluster);
                allowedPermutations.insert(allowedPermutation);
            }
            catch (const std::runtime_error &e)
            {
                {
                    failedLoops++;
                    if (failedLoops == representativeClusterWithTranslations.size())
                    {
                        throw std::runtime_error("Did not find integer permutation from allowed permutation to any translated representative site (OrbitList::createOrbit)");
                    }
                    continue;
                }
            }
        }
    }

    // Step seven: Relate equivalent clusters to the representative cluster, i.e. what is the consistent ordering of the cluster
    std::unordered_set<std::vector<LatticeSite>> p_equal_set;
    p_equal_set.insert(equivalentClustersWithTranslations.begin(), equivalentClustersWithTranslations.end());

    std::vector<std::vector<LatticeSite>> permutedEquivalentClusters;

    for (const auto &equivalentOrbitSites : equivalentClusters)
    {
        if (p_equal_set.find(equivalentOrbitSites) == p_equal_set.end())
        {
            // Did not find the cluster in p_equal_set meaning that this cluster is not permuted as it should
            auto equivalentlyTranslatedEquivalentOrbitSites = getSitesTranslatedToUnitcell(equivalentOrbitSites, sortRows);
            std::vector<std::vector<LatticeSite>> translatedPermutationsOfSites;
            for (const auto &eq_trans_equivalentOrbitsites : equivalentlyTranslatedEquivalentOrbitSites)
            {
                const auto allPermutationsOfSites_i = icet::getAllPermutations<LatticeSite>(eq_trans_equivalentOrbitsites);
                for (const auto &perm : allPermutationsOfSites_i)
                {
                    translatedPermutationsOfSites.push_back(perm);
                }
            }
            for (const auto &perm : translatedPermutationsOfSites)
            {
                const auto findOnePerm = p_equal_set.find(perm);
                if (findOnePerm != p_equal_set.end()) // one perm is one of the equivalent sites. This means that equivalentOrbitSites is associated to p_equal
                {
                    permutedEquivalentClusters.push_back(perm);
                    break;
                }
                if (perm == translatedPermutationsOfSites.back())
                {
                    throw std::runtime_error("Did not find a permutation of the orbit sites to the permutations of the representative sites (OrbitList::createOrbit)");
                }
            }
        }
        else
        {
            permutedEquivalentClusters.push_back(equivalentOrbitSites);
        }
    }

    if (permutedEquivalentClusters.size() != equivalentClusters.size())
    {
        std::ostringstream msg;
        msg << "Not all clusters were permuted (OrbitList::createOrbit) " << std::endl;
        msg << permutedEquivalentClusters.size() << " != " << equivalentClusters.size();
        throw std::runtime_error(msg.str());
    }

    // Turn the permuted equivalent clusters into actual Cluster objects
    std::vector<Cluster> clusters;
    std::shared_ptr<Structure> structurePtr = std::make_shared<Structure>(_structure);
    for (auto cluster : permutedEquivalentClusters)
    {
        clusters.push_back(Cluster(cluster, structurePtr));
    }
    Orbit newOrbit = Orbit(clusters, allowedPermutations);
    return newOrbit;
}

/**
@details
    Given a group of sites ("cluster"), this function returns the clusters
    that are equivalent to this cluster by symmetries of the underlying
    crystal. These clusters are obtained by

    (1) identifying the rows in matrixOfEquivalentSites that correspond to
        the input cluster (i.e., where in the first column are the sites of
        the input cluster?),
    (2) looping over the remaining columns of matrixOfEquivalentSites and
        forming new clusters by combining the sites corresponding to the
        rows identified in (1),
    (3) forming translational equivalents of the thus obtained clusters
        by translating them such that one of its sites at a time lies in
        the unit cell.

@param sites The sites of a cluster
@returns Groups of sites ("clusters") that are equivalent to the input cluster
**/
std::vector<std::vector<LatticeSite>> OrbitList::getSymmetryRelatedSiteGroups(const std::vector<LatticeSite> &sites) const
{
    std::vector<int> rowIndicesFromReferenceLatticeSites = getReferenceLatticeSiteIndices(sites, false);

    std::vector<std::vector<LatticeSite>> allColumns;
    for (size_t columnIndex = 0; columnIndex < _matrixOfEquivalentSites[0].size(); columnIndex++)
    {
        std::vector<LatticeSite> nondistinctLatticeSites;

        for (const int &rowIndex : rowIndicesFromReferenceLatticeSites)
        {
            nondistinctLatticeSites.push_back(_matrixOfEquivalentSites[rowIndex][columnIndex]);
        }

        // Include translated sites as well
        auto translatedEquivalentSites = getSitesTranslatedToUnitcell(nondistinctLatticeSites, false);
        allColumns.insert(allColumns.end(), translatedEquivalentSites.begin(), translatedEquivalentSites.end());
    }
    return allColumns;
}

/**
@details
This function creates all possible translations of the input list of lattice sites, for which at
least one of the lattice sites is inside the (original) unit cell.
For example, given a pair with unit cell offsets
  [0, 0, 1], [-3, 0, 3]
one gets
  [0, 0, 0], [-3, 0, 2]
  [3, 0, -2], [0, 0, 0]

This translation gives rise to equivalent clusters that sometimes
are not found by using the set of crystal symmetries given by spglib.

@param latticeSites list of lattice sites
@param sort if true sort the translated sites
*/
std::vector<std::vector<LatticeSite>> OrbitList::getSitesTranslatedToUnitcell(const std::vector<LatticeSite> &latticeSites,
                                                                              bool sort) const
{

    std::vector<std::vector<LatticeSite>> listOfTranslatedLatticeSites;
    listOfTranslatedLatticeSites.push_back(latticeSites);
    for (size_t i = 0; i < latticeSites.size(); i++)
    {
        if (latticeSites[i].unitcellOffset().norm() > 0.5) // only translate sites outside the primitive unitcell
        {
            auto translatedSites = getTranslatedSites(latticeSites, i);
            if (sort)
            {
                std::sort(translatedSites.begin(), translatedSites.end());
            }
            listOfTranslatedLatticeSites.push_back(translatedSites);
        }
    }

    // Sort this so that the lowest vec<latticeSite> will be chosen and therefore the sorting of orbits should be consistent.
    std::sort(listOfTranslatedLatticeSites.begin(), listOfTranslatedLatticeSites.end());

    return listOfTranslatedLatticeSites;
}

/**
@details Takes all lattice sites in vector latticeSites and subtracts the unitcell offset of site latticeSites[index].
@param latticeSites List of lattice sites, typically a cluster
@param index Index of site relative to which to shift
*/
std::vector<LatticeSite> OrbitList::getTranslatedSites(const std::vector<LatticeSite> &latticeSites,
                                                       const unsigned int index) const
{
    Vector3i offset = latticeSites[index].unitcellOffset();
    auto translatedSites = latticeSites;
    for (auto &latticeSite : translatedSites)
    {
        latticeSite.addUnitcellOffset(-offset);
    }
    return translatedSites;
}

/**
@details Adds columns of the matrix of equivalent sites to the orbit list.
@param listOfEquivalentClusters list of lattice sites to which to add
The first index (vector) runs over the orbits,
the second index (vector) over the equivalent cluster in a given orbit, and
the final vector runs over the lattice sites that represent a particular cluster.

@param rowsTaken
@param rowIndices indices of rows in matrix of symmetry equivalent sites
@todo fix the description of this function, including its name
**/
void OrbitList::addColumnsFromMatrixOfEquivalentSites(std::vector<std::vector<std::vector<LatticeSite>>> &listOfEquivalentClusters,
                                                      std::unordered_set<std::vector<int>, VectorHash> &rowsTaken,
                                                      const std::vector<int> &rowIndices) const
{

    std::vector<std::vector<LatticeSite>> columnLatticeSites;
    columnLatticeSites.reserve(_matrixOfEquivalentSites[0].size());
    for (size_t column = 0; column < _matrixOfEquivalentSites[0].size(); column++)
    {
        std::vector<LatticeSite> nondistinctLatticeSites;

        for (const int &row : rowIndices)
        {
            nondistinctLatticeSites.push_back(_matrixOfEquivalentSites[row][column]);
        }
        auto translatedEquivalentSites = getSitesTranslatedToUnitcell(nondistinctLatticeSites);
        auto pairsOfSitesAndRowIndices = getMatchesInMatrixOfEquivalentSites(translatedEquivalentSites);

        auto find = rowsTaken.find(pairsOfSitesAndRowIndices[0].second);
        bool findOnlyOne = true;
        if (find == rowsTaken.end())
        {
            for (size_t i = 0; i < pairsOfSitesAndRowIndices.size(); i++)
            {
                find = rowsTaken.find(pairsOfSitesAndRowIndices[i].second);
                if (find == rowsTaken.end())
                {
                    if (findOnlyOne && validCluster(pairsOfSitesAndRowIndices[i].first))
                    {
                        columnLatticeSites.push_back(pairsOfSitesAndRowIndices[0].first);
                        findOnlyOne = false;
                    }
                    rowsTaken.insert(pairsOfSitesAndRowIndices[i].second);
                }
            }
        }
    }
    if (columnLatticeSites.size() > 0)
    {
        listOfEquivalentClusters.push_back(columnLatticeSites);
    }
}

/**
@details Returns the first set of translated sites that exist in referenceLatticeSites.
*/
std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> OrbitList::getMatchesInMatrixOfEquivalentSites(
    const std::vector<std::vector<LatticeSite>> &translatedSites) const
{
    std::vector<int> perm_matrix_rows;
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> matchedSites;
    for (const auto &sites : translatedSites)
    {
        try
        {
            perm_matrix_rows = getReferenceLatticeSiteIndices(sites);
        }
        catch (std::runtime_error const &)
        {
            continue;
        }
        // No error here indicating that we found matching rows in reference lattice sites
        matchedSites.push_back(std::make_pair(sites, perm_matrix_rows));
    }
    if (matchedSites.size() > 0)
    {
        return matchedSites;
    }
    else
    {
        // No matching rows in matrix of equivalent sites, this should not happen so we throw an error.
        throw std::runtime_error("Did not find any of the translated sites in reference lattice sites in the matrix of equivalent sites (OrbitList::addColumnsFromMatrixOfEquivalentSites)");
    }
}

/**
@details This function returns true if the cluster includes at least on site from the unit cell at the origin, i.e. its unitcell offset is zero.
@param latticeSites list of sites to check
*/
bool OrbitList::validCluster(const std::vector<LatticeSite> &latticeSites) const
{
    for (const auto &latticeSite : latticeSites)
    {
        if (latticeSite.unitcellOffset().norm() < 0.5)
        {
            return true;
        }
    }
    return false;
}

/**
@details
    For each lattice site in the input vector, this function returns the
    index of the entry in _referenceLatticeSites that holds an equivalent
    lattice site.
@param latticeSites List of sites to search for
@param sort If true, the returned list of indices will be sorted
@return Indices of entries in _referenceLatticeSites that are equivalent to the sites latticeSites
**/
std::vector<int> OrbitList::getReferenceLatticeSiteIndices(const std::vector<LatticeSite> &latticeSites,
                                                           bool sort) const
{
    std::vector<int> rowIndices;
    for (const auto &latticeSite : latticeSites)
    {
        const auto find = std::find(_referenceLatticeSites.begin(), _referenceLatticeSites.end(), latticeSite);
        if (find == _referenceLatticeSites.end())
        {
            throw std::runtime_error("Did not find lattice site in the reference lattice sites in the matrix of equivalent sites (OrbitList::getReferenceLatticeSiteIndices)");
        }
        else
        {
            int rowIndexInReferenceLatticeSites = std::distance(_referenceLatticeSites.begin(), find);
            rowIndices.push_back(rowIndexInReferenceLatticeSites);
        }
    }
    if (sort)
    {
        std::sort(rowIndices.begin(), rowIndices.end());
    }
    return rowIndices;
}

/**
@details Returns reference lattice sites, which is equivalent to returning the first column of the matrix of equivalent sites.
@todo Expand description.
**/
std::vector<LatticeSite> OrbitList::getReferenceLatticeSites() const
{
    std::vector<LatticeSite> referenceLatticeSites;
    referenceLatticeSites.reserve(_matrixOfEquivalentSites[0].size());
    for (const auto &row : _matrixOfEquivalentSites)
    {
        referenceLatticeSites.push_back(row[0]);
    }
    return referenceLatticeSites;
}

/**
@details Removes an orbit identified by index from the orbit list.
@param index index of the orbit in question
**/
void OrbitList::removeOrbit(const size_t index)
{
    if (index >= size())
    {
        std::ostringstream msg;
        msg << "Index " << index << " was out of bounds (OrbitList::removeOrbit)." << std::endl;
        msg << "OrbitList size: " << size();
        throw std::out_of_range(msg.str());
    }
    _orbits.erase(_orbits.begin() + index);
}

/**
@details
    Removes all orbits that have inactive sites,
    i.e., sites with only one allowed species.
**/
void OrbitList::removeOrbitsWithInactiveSites()
{
    // Loop in reverse direction since we may remove orbits
    for (int i = _orbits.size() - 1; i >= 0; i--)
    {
        auto nSpecies = _orbits[i].representativeCluster().getNumberOfAllowedSpeciesPerSite();
        if (std::any_of(nSpecies.begin(), nSpecies.end(), [](int n)
                        { return n < 2; }))
        {
            removeOrbit(i);
        }
    }
}

/**
@brief Getter for stucture object
*/
const Structure &OrbitList::structure() const
{
    if (_structure.size() == 0)
    {
        std::ostringstream msg;
        msg << "No structure has been initialized in this OrbitList (OrbitList::structure)";
        throw std::runtime_error(msg.str());
    }
    return _structure;
}

/**
@brief Adds an orbit to another orbit.
@details
    This function adds the clusters of the orbit with orbit index index2
    to the clusters of orbit with index1. The orbit with index2 is not
    affected.
@param index1 Orbit index of the orbit that will get new clusters
@param index2 Orbit index of the orbit whose clusters will be added to orbit with index index1
**/
void OrbitList::mergeOrbits(int index1, int index2)
{
    _orbits[index1] += _orbits[index2];
}

/**
@details Provides the "+=" operator for adding orbit lists.
First assert that they have the same number of orbits or that this is empty and
then add equivalent sites of orbit i of rhs to orbit i to ->this
**/
OrbitList &OrbitList::operator+=(const OrbitList &rhs_ol)
{
    if (size() == 0)
    {
        _orbits = rhs_ol.orbits();
        return *this;
    }

    if (size() != rhs_ol.size())
    {
        std::ostringstream msg;
        msg << "Left (" << size() << ") and right hand side (" << rhs_ol.size();
        msg << ") differ in size (OrbitList& operator+=).";
        throw std::runtime_error(msg.str());
    }

    for (size_t i = 0; i < rhs_ol.size(); i++)
    {
        _orbits[i] += rhs_ol.getOrbit(i);
    }
    return *this;
}
