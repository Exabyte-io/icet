#include "OrbitList.hpp"

/**
@details This constructor generates an orbit list for the given (supercell) structure from a set of neighbor lists and a matrix of (symmetry) equivalent sites.
@param structure (supercell) structure for which to generate orbit list
@param matrixOfEquivalentSites matrix of symmetry equivalent sites
@param neighborLists neighbor lists for each (cluster) order (0=pairs, 1=triplets etc)
@param positionTolerance tolerance applied when comparing positions in Cartesian coordinates
**/
OrbitList::OrbitList(const Structure &structure,
                     const std::vector<std::vector<LatticeSite>> &matrixOfEquivalentSites,
                     const std::vector<NeighborList> &neighborLists,
                     const double positionTolerance)
{
    bool bothways = false;
    _primitiveStructure = structure;
    std::vector<std::vector<std::vector<LatticeSite>>> latticeSites;
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> many_bodyNeighborIndices;
    ManyBodyNeighborList mbnl = ManyBodyNeighborList();

    // rows that have already been accounted for
    std::unordered_set<std::vector<int>, VectorHash> taken_rows;
    std::vector<LatticeSite> referenceLatticeSites = getReferenceLatticeSites(matrixOfEquivalentSites, false);

    // check that there are no duplicates in the first column of the matrix of equivalent sites
    std::set<LatticeSite> uniqueReferenceLatticeSites(referenceLatticeSites.begin(), referenceLatticeSites.end());
    if (referenceLatticeSites.size() != uniqueReferenceLatticeSites.size())
    {
        std::ostringstream msg;
        msg << "Found duplicates in the list of reference lattice sites (= first column of matrix of equivalent sites): ";
        msg << std::to_string(referenceLatticeSites.size()) << " != " << std::to_string(uniqueReferenceLatticeSites.size());
        msg << " (OrbitList::OrbitList)";
        throw std::runtime_error(msg.str());
    }


    for (size_t index = 0; index < neighborLists[0].size(); index++)
    {

        std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> mbnl_latticeSites = mbnl.build(neighborLists, index, bothways);
        for (const auto &mbnl_pair : mbnl_latticeSites)
        {

            for (const auto &latticeSite : mbnl_pair.second)
            {
                std::vector<LatticeSite> lattice_sites = mbnl_pair.first;
                lattice_sites.push_back(latticeSite);
                auto lattice_sites_copy = lattice_sites;
                std::sort(lattice_sites_copy.begin(), lattice_sites_copy.end());
                if (lattice_sites_copy != lattice_sites && !bothways)
                {
                    throw std::runtime_error("Original sites are not sorted (OrbitList::OrbitList)");
                }
                std::vector<std::vector<LatticeSite>> translatedSites = getSitesTranslatedToUnitcell(lattice_sites);

                auto sites_index_pair = getMatchesInPM(translatedSites, referenceLatticeSites);
                if (!isRowsTaken(taken_rows, sites_index_pair[0].second))
                {
                    // new stuff found
                    addColumnsFromMatrixOfEquivalentSites(latticeSites, taken_rows, sites_index_pair[0].first, sites_index_pair[0].second, matrixOfEquivalentSites, referenceLatticeSites, true);
                }
            }

            // special singlet case
            // copy-paste from above section but with line with lattice_sites.push_back(latticeSite); is removed
            if (mbnl_pair.second.size() == 0)
            {
                std::vector<LatticeSite> lattice_sites = mbnl_pair.first;
                auto pm_rows = getIndicesOfEquivalentLatticeSites(referenceLatticeSites, lattice_sites);
                auto find = taken_rows.find(pm_rows);
                if (find == taken_rows.end())
                {
                    // Found new stuff
                    addColumnsFromMatrixOfEquivalentSites(latticeSites, taken_rows, lattice_sites, pm_rows, matrixOfEquivalentSites, referenceLatticeSites, true);
                }
            }
        }
    }

    for (size_t i = 0; i < latticeSites.size(); i++)
    {
        std::sort(latticeSites[i].begin(), latticeSites[i].end());
    }

    addOrbitsFromPM(structure, latticeSites);

    addPermutationInformationToOrbits(referenceLatticeSites, matrixOfEquivalentSites);

    // Sort the orbit list.
    sort(positionTolerance);

}

/**
@details This function sorts the orbit list by order and radius. This is done to obtain a reproducable (stable) order of the orbit list.
@param positionTolerance tolerance applied when comparing positions in Cartesian coordinates
*/
void OrbitList::sort(const double positionTolerance)
{
    std::sort(_orbits.begin(), _orbits.end(),
              [positionTolerance](const Orbit& lhs, const Orbit& rhs)
              {
                  // Test against number of bodies in cluster.
                  if (lhs.getRepresentativeCluster().order() != rhs.getRepresentativeCluster().order())
                  {
                      return lhs.getRepresentativeCluster().order() < rhs.getRepresentativeCluster().order();
                  }
                  // Compare by radius.
                  if (fabs(lhs.radius() - rhs.radius()) > positionTolerance)
                  {
                      return lhs.radius() < rhs.radius();
                  }

                  // Check size of vector of equivalent sites.
                  if (lhs.size() < rhs.size())
                  {
                      return true;
                  }
                  if (lhs.size() > rhs.size())
                  {
                      return false;
                  }

                  // Check the individual equivalent sites.
                  return lhs.getEquivalentSites() < rhs.getEquivalentSites();
              });
}

/**
@param orbit orbit to add to orbit list
**/
void OrbitList::addOrbit(const Orbit &orbit) {
    _orbits.push_back(orbit);
}

/**
@param nbody number of bodies for which to return the number of clusters
**/
unsigned int OrbitList::getNumberOfNBodyClusters(unsigned int nbody) const
{
    unsigned int count = 0;
    for (const auto &orbit : _orbits)
    {
        if (orbit.getRepresentativeCluster().order() == nbody)
        {
            count++;
        }
    }
    return count;
}

/**
@details Returns the index of the orbit for which the given cluster is representative.
@param cluster cluster to search for
@param clusterIndexMap map of cluster indices for fast lookup
@returns orbit index; -1 if nothing is found
**/
int OrbitList::findOrbitIndex(const Cluster &cluster,
                              const std::unordered_map<Cluster, int> &clusterIndexMap) const
{
    auto search = clusterIndexMap.find(cluster);
    if (search != clusterIndexMap.end())
    {
        return search->second;
    }
    else
    {
        return -1;
    }
}

/**
@details Returns a copy of the orbit at the given index.
@param index index of orbit
@returns copy of orbit
**/
Orbit OrbitList::getOrbit(unsigned int index) const
{
    if (index >= size())
    {
        throw std::out_of_range("Tried accessing orbit at out of bound index (Orbit OrbitList::getOrbit)");
    }
    return _orbits[index];
}

/**
@details
This function adds permutation related information to the orbits.

Algorithm
---------

For each orbit:

@todo review this algorithm; address question marks

1. Take representative sites
2. Find the rows these sites belong to (also find the unit cell offsets equivalent sites??)
3. Get all columns for these rows, i.e the sites that are directly equivalent, call these p_equal.
4. Construct all possible permutations for the representative sites, call these p_all
5. Construct the intersect of p_equal and p_all, call this p_allowed_permutations.
6. Get the indice version of p_allowed_permutations and these are then the allowed permutations for this orbit.
7. Take the sites in the orbit:
    site exist in p_all?:
        those sites are then related to representative_sites through the permutation
    else:
        loop over permutations of the sites:
            does the permutation exist in p_all?:
                that permutation is then related to rep_sites through that permutation
            else:
                continue

**/
void OrbitList::addPermutationInformationToOrbits(const std::vector<LatticeSite> &referenceLatticeSites,
                                                  const std::vector<std::vector<LatticeSite>> &matrixOfEquivalentSites)
{
    _referenceLatticeSites = referenceLatticeSites;
    _matrixOfEquivalentSites = matrixOfEquivalentSites;

    for (size_t i = 0; i < size(); i++)
    {

        bool sortRows = false;

        // step one: Take representative sites
        std::vector<LatticeSite> representativeSites_i = _orbits[i].getRepresentativeSites();
        auto translatedRepresentativeSites = getSitesTranslatedToUnitcell(representativeSites_i, sortRows);

        // step two: Find the rows these sites belong to and,

        // step three: Get all columns for these rows
        std::vector<std::vector<LatticeSite>> all_translated_p_equal;

        for (auto translated_rep_sites : translatedRepresentativeSites)
        {
            auto p_equal_i = getAllColumnsFromSites(translated_rep_sites, referenceLatticeSites, matrixOfEquivalentSites);
            all_translated_p_equal.insert(all_translated_p_equal.end(), p_equal_i.begin(), p_equal_i.end());
        }

        std::sort(all_translated_p_equal.begin(), all_translated_p_equal.end());

        // Step four: Construct all possible permutations for the representative sites
        std::vector<std::vector<LatticeSite>> p_all_with_translated_equivalent;
        for (auto translated_rep_sites : translatedRepresentativeSites)
        {
            std::vector<std::vector<LatticeSite>> p_all_i = icet::getAllPermutations<LatticeSite>(translated_rep_sites);
            p_all_with_translated_equivalent.insert(p_all_with_translated_equivalent.end(), p_all_i.begin(), p_all_i.end());
        }
        std::sort(p_all_with_translated_equivalent.begin(), p_all_with_translated_equivalent.end());

        // Step five:  Construct the intersect of p_equal and p_all
        std::vector<std::vector<LatticeSite>> p_allowed_permutations;
        std::set_intersection(all_translated_p_equal.begin(), all_translated_p_equal.end(),
                              p_all_with_translated_equivalent.begin(), p_all_with_translated_equivalent.end(),
                              std::back_inserter(p_allowed_permutations));

        // Step six: Get the indice version of p_allowed_permutations
        std::set<std::vector<int>> allowedPermutations;
        for (const auto &p_lattNbr : p_allowed_permutations)
        {
            size_t failedLoops = 0;
            for (auto translated_rep_sites : translatedRepresentativeSites)
            {
                try
                {
                    std::vector<int> allowedPermutation = icet::getPermutation<LatticeSite>(translated_rep_sites, p_lattNbr);
                    allowedPermutations.insert(allowedPermutation);
                }
                catch (const std::runtime_error &e)
                {
                    {
                        failedLoops++;
                        if (failedLoops == translatedRepresentativeSites.size())
                        {
                            throw std::runtime_error("Did not find integer permutation from allowed permutation to any translated representative site (OrbitList::addPermutationInformationToOrbits)");
                        }
                        continue;
                    }
                }
            }
        }

        // Step 7
        const auto orbitSites = _orbits[i].getEquivalentSites();
        std::unordered_set<std::vector<LatticeSite>> p_equal_set;
        p_equal_set.insert(all_translated_p_equal.begin(), all_translated_p_equal.end());

        std::vector<std::vector<int>> sitePermutations;
        sitePermutations.reserve(orbitSites.size());

        for (const auto &eqOrbitSites : orbitSites)
        {
            if (p_equal_set.find(eqOrbitSites) == p_equal_set.end())
            {
                // Did not find the orbit.eq_sites in p_equal meaning that this eq site does not have an allowed permutation.
                auto equivalently_translated_eqOrbitsites = getSitesTranslatedToUnitcell(eqOrbitSites, sortRows);
                std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> translatedPermutationsOfSites;
                for (const auto eq_trans_eqOrbitsites : equivalently_translated_eqOrbitsites)
                {
                    const auto allPermutationsOfSites_i = icet::getAllPermutations<LatticeSite>(eq_trans_eqOrbitsites);
                    for (const auto perm : allPermutationsOfSites_i)
                    {
                        translatedPermutationsOfSites.push_back(std::make_pair(perm, eq_trans_eqOrbitsites));
                    }
                }
                for (const auto &onePermPair : translatedPermutationsOfSites)
                {
                    const auto findOnePerm = p_equal_set.find(onePermPair.first);
                    if (findOnePerm != p_equal_set.end()) // one perm is one of the equivalent sites. This means that eqOrbitSites is associated to p_equal
                    {
                        std::vector<int> permutationToEquivalentSites = icet::getPermutation<LatticeSite>(onePermPair.first, onePermPair.second);
                        sitePermutations.push_back(permutationToEquivalentSites);
                        break;
                    }
                    if (onePermPair == translatedPermutationsOfSites.back())
                    {
                        throw std::runtime_error("Did not find a permutation of the orbit sites to the permutations of the representative sites (OrbitList::addPermutationInformationToOrbits)");
                    }
                }
            }
            else
            {
                std::vector<int> permutationToEquivalentSites = icet::getPermutation<LatticeSite>(eqOrbitSites, eqOrbitSites); //the identical permutation
                sitePermutations.push_back(permutationToEquivalentSites);
            }
        }

        if (sitePermutations.size() != _orbits[i].getEquivalentSites().size() || sitePermutations.size() == 0)
        {
            std::ostringstream msg;
            msg << "Not each set of site got a permutation (OrbitList::addPermutationInformationToOrbits) " << std::endl;
            msg << sitePermutations.size() << " != " << _orbits[i].getEquivalentSites().size();
            throw std::runtime_error(msg.str());
        }

        _orbits[i].setEquivalentSitesPermutations(sitePermutations);
        _orbits[i].setAllowedSitesPermutations(allowedPermutations);
    }
}

/**
@details Finds the sites in referenceLatticeSites, extract all columns along with their unit cell translated indistinguishable sites.
@param sites sites that correspond to the columns that will be returned
@param referenceLatticeSites sites in the first column of matrix of equivalent sites
@param matrixOfEquivalentSites matrix of symmetry equivalent sites
@returns columns along with their unit cell translated indistinguishable sites
**/
std::vector<std::vector<LatticeSite>> OrbitList::getAllColumnsFromSites(const std::vector<LatticeSite> &sites,
                                                                        const std::vector<LatticeSite> &referenceLatticeSites,
                                                                        const std::vector<std::vector<LatticeSite>> &matrixOfEquivalentSites) const
{
    bool sortRows = false;
    std::vector<int> rowsFromReferenceLatticeSites = getIndicesOfEquivalentLatticeSites(referenceLatticeSites, sites, sortRows);
    std::vector<std::vector<LatticeSite>> p_equal = getAllColumnsFromRow(rowsFromReferenceLatticeSites, matrixOfEquivalentSites, true, sortRows);
    return p_equal;
}

/// Returns true if rows_sort exists in taken_rows.
bool OrbitList::isRowsTaken(const std::unordered_set<std::vector<int>, VectorHash> &taken_rows,
                            std::vector<int> rows) const
{
    const auto find = taken_rows.find(rows);
    if (find == taken_rows.end())
    {
        return false;
    }
    else
    {
        return true;
    }
}

/**
@brief Returns all columns from the given rows in matrix of symmetry equivalent sites
@param rows indices of rows to return
@param matrixOfEquivalentSites matrix of symmetry equivalent sites
@param includeTranslatedSites If true it will also include the equivalent sites found from the rows by moving each site into the unitcell.
@param sortIt if true (default) the first column will be sorted
**/
std::vector<std::vector<LatticeSite>> OrbitList::getAllColumnsFromRow(const std::vector<int> &rows,
                                                                      const std::vector<std::vector<LatticeSite>> &matrixOfEquivalentSites,
                                                                      bool includeTranslatedSites,
                                                                      bool sortIt) const
{
    std::vector<std::vector<LatticeSite>> allColumns;

    for (size_t column = 0; column < matrixOfEquivalentSites[0].size(); column++)
    {
        std::vector<LatticeSite> indistinctlatticeSites;

        for (const int &row : rows)
        {
            indistinctlatticeSites.push_back(matrixOfEquivalentSites[row][column]);
        }

        if (includeTranslatedSites)
        {
            auto translatedEquivalentSites = getSitesTranslatedToUnitcell(indistinctlatticeSites, sortIt);
            allColumns.insert(allColumns.end(), translatedEquivalentSites.begin(), translatedEquivalentSites.end());
        }
        else
        {
            allColumns.push_back(indistinctlatticeSites);
        }
    }
    return allColumns;
}

/**
@details
This function will take a list of lattice sites and for each site outside the unitcell
will translate it inside the unitcell and translate the other sites with the same translation.

This translation will give rise to equivalent sites that sometimes
are not found by using the set of crystal symmetries given by spglib.

An added requirement to this is that this function should not
give rise to any sites in non-periodic directions.

@param latticeSites list of lattice sites
@param sortIt if true sort the translated sites
@todo Complete description.
*/
std::vector<std::vector<LatticeSite>> OrbitList::getSitesTranslatedToUnitcell(
    const std::vector<LatticeSite> &latticeSites,
    bool sortIt) const
{

    /// Sanity check that the periodic boundary conditions are currently respected.
    if (!isSitesPBCCorrect(latticeSites))
    {
        throw std::runtime_error("Received a latticeSite that had a repeated site in the unitcell direction where pbc was false (OrbitList::getSitesTranslatedToUnitcell)");
    }

    std::vector<std::vector<LatticeSite>> translatedLatticeSites;
    translatedLatticeSites.push_back(latticeSites);
    Vector3d zeroVector = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < latticeSites.size(); i++)
    {
        if ((latticeSites[i].unitcellOffset() - zeroVector).norm() > 0.1) // only translate those outside unitcell
        {
            auto translatedSites = translateSites(latticeSites, i);
            if (sortIt)
            {
                std::sort(translatedSites.begin(), translatedSites.end());
            }

            if (!isSitesPBCCorrect(translatedSites))
            {
                throw std::runtime_error("Translated a latticeSite and got a repeated site in the unitcell direction where pbc was false (OrbitList::getSitesTranslatedToUnitcell)");
            }

            translatedLatticeSites.push_back(translatedSites);
        }
    }

    // sort this so that the lowest vec<latticeSite> will be chosen and therefore the sorting of orbits should be consistent.
    std::sort(translatedLatticeSites.begin(), translatedLatticeSites.end());

    return translatedLatticeSites;
}

/// Checks that the lattice neighbors do not have any unitcell offsets in a pbc=false direction.
/// @todo Complete description.
bool OrbitList::isSitesPBCCorrect(const std::vector<LatticeSite> &sites) const
{
    for (const auto &latticeSite : sites)
    {
        for (size_t i = 0; i < 3; i++)
        {
            if (!_primitiveStructure.hasPBC(i) && latticeSite.unitcellOffset()[i] != 0)
            {
                return false;
            }
        }
    }
    return true;
}

/// Takes all lattice neighbors in vector latticeSites and subtract the unitcelloffset of site latticeSites[index].
/// @todo Complete description.
std::vector<LatticeSite> OrbitList::translateSites(const std::vector<LatticeSite> &latticeSites,
                                                   const unsigned int index) const
{
    Vector3d offset = latticeSites[index].unitcellOffset();
    auto translatedSites = latticeSites;
    for (auto &latticeSite : translatedSites)
    {
        latticeSite.addUnitcellOffset(-offset);
    }
    return translatedSites;
}

/// Debug function for checking that all equivalent sites in every orbit yield the same radius.
/// @param positionTolerance tolerance applied when evaluating positions in Cartesian coordinates
/// @todo Consider this function for removal. Python side check should be sufficient.
void OrbitList::checkEquivalentClusters(const double positionTolerance) const
{
    for (const auto &orbit : _orbits)
    {
        Cluster representative_cluster = orbit.getRepresentativeCluster();
        for (const auto &sites : orbit.getEquivalentSites())
        {
            Cluster equivalentCluster = Cluster(_primitiveStructure, sites, true);
            if (fabs(equivalentCluster.radius() - representative_cluster.radius()) > positionTolerance)
            {
                std::ostringstream msg;
                msg << "Found an 'equivalent' cluster that does not match the representative cluster (OrbitList::checkEquivalentClusters)." << std::endl;
                msg << "representative_cluster: " << representative_cluster << std::endl;
                msg << "equivalentCluster:      " << equivalentCluster << std::endl;
                msg << "geometric size: " << icet::getGeometricalRadius(sites, _primitiveStructure);
                throw std::runtime_error(msg.str());
            }
        }
    }
}

/**
@details This function adds the latticeSites container found in the constructor to the orbits.
Each outer vector is an orbit and the inner vectors are identical sites
@todo Complete description.
*/
void OrbitList::addOrbitsFromPM(const Structure &structure,
                                const std::vector<std::vector<std::vector<LatticeSite>>> &latticeSites)
{
    for (const auto &equivalent_sites : latticeSites)
    {
        addOrbitFromPM(structure, equivalent_sites);
    }
}

/// Adds these equivalent sites as an orbit to orbit list.
void OrbitList::addOrbitFromPM(const Structure &structure,
                               const std::vector<std::vector<LatticeSite>> &equivalent_sites)
{
    Cluster representativeCluster = Cluster(structure, equivalent_sites[0]);
    Orbit newOrbit = Orbit(representativeCluster);
    _orbits.push_back(newOrbit);

    for (const auto &sites : equivalent_sites)
    {
        _orbits.back().addEquivalentSites(sites);
    }
    _orbits.back().sort();
}

/**
@details Adds columns of the matrix of equivalent sites to the orbit list.
@param latticeSites list of lattice sites to which to add
@param taken_rows
@param lattice_sites
@param pm_rows indices of rows in matrix of symmetry equivalent sites
@param matrixOfEquivalentSites
@param referenceLatticeSites
@param add
@todo fix the description of this function, including its name
**/
void OrbitList::addColumnsFromMatrixOfEquivalentSites(std::vector<std::vector<std::vector<LatticeSite>>> &latticeSites,
                                                          std::unordered_set<std::vector<int>, VectorHash> &taken_rows,
                                                          const std::vector<LatticeSite> &lattice_sites,
                                                          const std::vector<int> &pm_rows,
                                                          const std::vector<std::vector<LatticeSite>> &matrixOfEquivalentSites,
                                                          const std::vector<LatticeSite> &referenceLatticeSites,
                                                          bool add) const
{

    std::vector<std::vector<LatticeSite>> columnLatticeSites;
    columnLatticeSites.reserve(matrixOfEquivalentSites[0].size());
    for (size_t column = 0; column < matrixOfEquivalentSites[0].size(); column++)
    {
        std::vector<LatticeSite> indistinctlatticeSites;

        for (const int &row : pm_rows)
        {
            indistinctlatticeSites.push_back(matrixOfEquivalentSites[row][column]);
        }
        auto translatedEquivalentSites = getSitesTranslatedToUnitcell(indistinctlatticeSites);

        auto sites_index_pair = getMatchesInPM(translatedEquivalentSites, referenceLatticeSites);

        auto find = taken_rows.find(sites_index_pair[0].second);
        bool findOnlyOne = true;
        if (find == taken_rows.end())
        {
            for (size_t i = 0; i < sites_index_pair.size(); i++)
            {
                find = taken_rows.find(sites_index_pair[i].second);
                if (find == taken_rows.end())
                {
                    if (add && findOnlyOne && validCluster(sites_index_pair[i].first))
                    {
                        columnLatticeSites.push_back(sites_index_pair[0].first);
                        findOnlyOne = false;
                    }
                    taken_rows.insert(sites_index_pair[i].second);
                }
            }
        }
    }
    if (columnLatticeSites.size() > 0)
    {
        latticeSites.push_back(columnLatticeSites);
    }
}

/// Returns the first set of translated sites that exists in referenceLatticeSites of permutationmatrix.
/// @todo Complete description.
std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> OrbitList::getMatchesInPM(
    const std::vector<std::vector<LatticeSite>> &translatedSites,
    const std::vector<LatticeSite> &referenceLatticeSites) const
{
    std::vector<int> perm_matrix_rows;
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> matchedSites;
    for (const auto &sites : translatedSites)
    {
        try
        {
            perm_matrix_rows = getIndicesOfEquivalentLatticeSites(referenceLatticeSites, sites);
        }
        catch (const std::runtime_error)
        {
            continue;
        }
        // No error here indicating that we found matching rows in referenceLatticeSites
        matchedSites.push_back(std::make_pair(sites, perm_matrix_rows));
    }
    if (matchedSites.size() > 0)
    {
        return matchedSites;
    }
    else
    {
        // No matching rows in matrix of equivalent sites, this should not happen so we throw an error.
        throw std::runtime_error("Did not find any of the translated sites in referenceLatticeSites in the matrix of equivalent sites (OrbitList::addColumnsFromMatrixOfEquivalentSites)");
    }
}
/**
@details This function returns true if the cluster includes at least on site from the unit cell at the origin, i.e. its unitcell offset is zero.
@param latticeSites list of sites to check
*/
bool OrbitList::validCluster(const std::vector<LatticeSite> &latticeSites) const
{
    Vector3d zeroVector = {0., 0., 0.};
    for (const auto &latticeSite : latticeSites)
    {
        if (latticeSite.unitcellOffset() == zeroVector)
        {
            return true;
        }
    }
    return false;
}

/**
@details Returns a list of indices of entries in latticeSites that are equivalent to the sites in referenceLatticeSites.
@param sortIt if true the first column will be sorted
@param referenceLatticeSites list of sites to search for; this commonly corresponds to the sites in the first column of the matrix of equivalent sites
@param latticeSites list of sites to search in
@return indices of entries in latticeSites that are equivalent to sites in referenceLatticeSites
**/
std::vector<int> OrbitList::getIndicesOfEquivalentLatticeSites(const std::vector<LatticeSite> &referenceLatticeSites,
							                                   const std::vector<LatticeSite> &latticeSites,
							                                   bool sortIt) const
{
    std::vector<int> rows;
    for (const auto &latticeSite : latticeSites)
    {
        const auto find = std::find(referenceLatticeSites.begin(), referenceLatticeSites.end(), latticeSite);
        if (find == referenceLatticeSites.end())
        {
            throw std::runtime_error("Did not find lattice site in referenceLatticeSites in matrix of equivalent sites (OrbitList::getIndicesOfEquivalentLatticeSites)");
        }
        else
        {
            int row_in_referenceLatticeSites = std::distance(referenceLatticeSites.begin(), find);
            rows.push_back(row_in_referenceLatticeSites);
        }
    }
    if (sortIt)
    {
        std::sort(rows.begin(), rows.end());
    }
    return rows;
}

/**
@details Returns reference lattice sites, which is equivalent to returning the first column of the matrix of equivalent sites.
@todo Expand description.
@param matrixOfEquivalentSites matrix of symmetry equivalent sites
@param sortIt if true (default) the first column will be sorted
**/
std::vector<LatticeSite> OrbitList::getReferenceLatticeSites(const std::vector<std::vector<LatticeSite>> &matrixOfEquivalentSites,
                                                             bool sortIt) const
{
    std::vector<LatticeSite> referenceLatticeSites;
    referenceLatticeSites.reserve(matrixOfEquivalentSites[0].size());
    for (const auto &row : matrixOfEquivalentSites)
    {
        referenceLatticeSites.push_back(row[0]);
    }
    if (sortIt)
    {
        std::sort(referenceLatticeSites.begin(), referenceLatticeSites.end());
    }
    return referenceLatticeSites;
}

/**
@details This function returns the orbit for a supercell that is associated with a given orbit in the primitive structure.
@param supercell input structure
@param cellOffset offset by which to translate the orbit
@param orbitIndex index of orbit in list of orbits
@param primToSuperMap map from sites in the primitive cell to sites in the supercell
@param fractionalPositionTolerance tolerance applied when comparing positions in fractional coordinates
**/
Orbit OrbitList::getSuperCellOrbit(const Structure &supercell,
                                   const Vector3d &cellOffset,
                                   const unsigned int orbitIndex,
                                   std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap,
                                   const double fractionalPositionTolerance) const
{
    if (orbitIndex >= _orbits.size())
    {
        std::ostringstream msg;
        msg << "Orbit index out of range (OrbitList::getSuperCellOrbit).";
        msg << orbitIndex << " >= " << _orbits.size();
        throw std::out_of_range(msg.str());
    }

    Orbit supercellOrbit = _orbits[orbitIndex] + cellOffset;

    auto equivalentSites = supercellOrbit.getEquivalentSites();

    for (auto &sites : equivalentSites)
    {
        for (auto &site : sites)
        {
            transformSiteToSupercell(site, supercell, primToSuperMap, fractionalPositionTolerance);
        }
    }

    supercellOrbit.setEquivalentSites(equivalentSites);
    return supercellOrbit;
}

/**
@details Transforms a site from the primitive structure to a given supercell.
This involves finding a map from the site in the primitive cell to the supercell.
If no map is found mapping is attempted based on the position of the site in the supercell.
@param site lattice site to transform
@param supercell supercell structure
@param primToSuperMap map from primitive to supercell
@param fractionalPositionTolerance tolerance applied when comparing positions in fractional coordinates
**/
void OrbitList::transformSiteToSupercell(LatticeSite &site,
                                         const Structure &supercell,
                                         std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap,
                                         const double fractionalPositionTolerance) const
{
    auto find = primToSuperMap.find(site);
    LatticeSite supercellSite;
    if (find == primToSuperMap.end())
    {
        Vector3d sitePosition = _primitiveStructure.getPosition(site);
        supercellSite = supercell.findLatticeSiteByPosition(sitePosition, fractionalPositionTolerance);
        primToSuperMap[site] = supercellSite;
    }
    else
    {
        supercellSite = primToSuperMap[site];
    }

    // overwrite site to match supercell index offset
    site.setIndex(supercellSite.index());
    site.setUnitcellOffset(supercellSite.unitcellOffset());
}

/**
@details Returns a "local" orbitList by offsetting each site in the primitive cell by an offset.
@param supercell supercell structure
@param cellOffset offset to be applied to sites
@param primToSuperMap map from primitive to supercell
@param fractionalPositionTolerance tolerance applied when comparing positions in fractional coordinates
**/
OrbitList OrbitList::getLocalOrbitList(const Structure &supercell,
                                       const Vector3d &cellOffset,
                                       std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap,
                                       const double fractionalPositionTolerance) const
{
    OrbitList localOrbitList = OrbitList();
    localOrbitList.setPrimitiveStructure(_primitiveStructure);

    for (size_t orbitIndex = 0; orbitIndex < _orbits.size(); orbitIndex++)
    {
        localOrbitList.addOrbit(getSuperCellOrbit(supercell, cellOffset, orbitIndex, primToSuperMap, fractionalPositionTolerance));
    }
    return localOrbitList;
}

/**
@details This function will loop over all orbits in the list and remove from each orbit the sites that match the given index.
@param index the index for which to check
@param onlyConsiderZeroOffset if true only sites with zero offset will be removed
**/
void OrbitList::removeSitesContainingIndex(const int index,
                                           bool onlyConsiderZeroOffset)
{
    for (auto &orbit : _orbits)
    {
        orbit.removeSitesWithIndex(index, onlyConsiderZeroOffset);
    }
}

/**
@details This function will loop over all orbits in the list and remove from each orbit the sites that _do _not_ match the given index.
@param index the index for which to check
@param onlyConsiderZeroOffset if true only sites with zero offset will be removed
**/
void OrbitList::removeSitesNotContainingIndex(const int index,
                                              bool onlyConsiderZeroOffset)
{
    for (auto &orbit : _orbits)
    {
        orbit.removeSitesNotWithIndex(index, onlyConsiderZeroOffset);
    }
}

/**
@details Removes from the current orbit list all sites that are in the input orbit list.
@param orbitList orbit list with sites that are to be removed.
 **/
void OrbitList::subtractSitesFromOrbitList(const OrbitList &orbitList)
{
    if (orbitList.size() != size())
    {
        throw std::runtime_error("Orbit lists differ in size (OrbitList::subtractSitesFromOrbitList)");
    }
    for (size_t i = 0; i < size(); i++)
    {
        for (const auto sites : orbitList.getOrbit(i)._equivalentSites)
        {
            if (_orbits[i].contains(sites, true))
            {
                _orbits[i].removeSites(sites);
            }
        }
    }
}

/**
@details This function removes an orbit identified by index from the orbit list.
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
@details Removes all orbits that have inactive sites.
@param structure the structure containining the number of allowed species on each lattice site
**/
void OrbitList::removeInactiveOrbits(const Structure &structure)
{
    for (int i = _orbits.size() - 1; i >= 0; i--)
    {
        auto numberOfAllowedSpecies = structure.getNumberOfAllowedSpeciesBySites(_orbits[i].getRepresentativeSites());
        if (std::any_of(numberOfAllowedSpecies.begin(), numberOfAllowedSpecies.end(), [](int n) { return n < 2; }))
        {
            removeOrbit(i);
        }
    }
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
        _orbits = rhs_ol.getOrbits();
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
