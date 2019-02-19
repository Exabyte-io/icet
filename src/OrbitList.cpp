#include "OrbitList.hpp"

/**
@details This constructor generates an orbit list for the given (supercell) structure from a set of neighbor lists and a permutation map.
@param neighbor_lists list of neighbor lists
@param permutationMatrix permutation matrix
@param structure (supercell) structure for which to generate orbit list
**/
OrbitList::OrbitList(const Structure &structure,
		     const std::vector<std::vector<LatticeSite>> &permutationMatrix,
		     const std::vector<NeighborList> &neighbor_lists)
{
    bool bothways = false;
    _primitiveStructure = structure;
    std::vector<std::vector<std::vector<LatticeSite>>> latticeSites;
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> many_bodyNeighborIndices;
    ManyBodyNeighborList mbnl = ManyBodyNeighborList();

    // if [0,1,2] exists in taken_rows then these three rows (with columns) have been accounted for and should not be looked at
    std::unordered_set<std::vector<int>, VectorHash> taken_rows;
    std::vector<LatticeSite> col1 = getColumn1FromPM(permutationMatrix, false);

    std::set<LatticeSite> col1_uniques(col1.begin(), col1.end());
    if (col1.size() != col1_uniques.size())
    {
        std::string msg = "Found duplicates in column1 of permutation matrix " + std::to_string(col1.size()) + " != " + std::to_string(col1_uniques.size());
        throw std::runtime_error(msg);
    }
    for (size_t index = 0; index < neighbor_lists[0].size(); index++)
    {

        std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> mbnl_latticeSites = mbnl.build(neighbor_lists, index, bothways);
        for (const auto &mbnl_pair : mbnl_latticeSites)
        {

            for (const auto &latticeSite : mbnl_pair.second)
            {
                std::vector<LatticeSite> lat_nbrs = mbnl_pair.first;
                lat_nbrs.push_back(latticeSite);
                auto lat_nbrs_copy = lat_nbrs;
                std::sort(lat_nbrs_copy.begin(), lat_nbrs_copy.end());
                if (lat_nbrs_copy != lat_nbrs and !bothways)
                {
                    throw std::runtime_error("Original sites is not sorted");
                }
                std::vector<std::vector<LatticeSite>> translatedSites = getSitesTranslatedToUnitcell(lat_nbrs);

                auto sites_index_pair = getMatchesInPM(translatedSites, col1);
                if (!isRowsTaken(taken_rows, sites_index_pair[0].second))
                {
                    //new stuff found
                    addPermutationMatrixColumns(latticeSites, taken_rows, sites_index_pair[0].first, sites_index_pair[0].second, permutationMatrix, col1, true);
                }
            }

            // special singlet case
            // copy-paste from above section but with line with lat_nbrs.push_back(latticeSite); is removed
            if (mbnl_pair.second.size() == 0)
            {
                std::vector<LatticeSite> lat_nbrs = mbnl_pair.first;
                auto pm_rows = findRowsFromCol1(col1, lat_nbrs);
                auto find = taken_rows.find(pm_rows);
                if (find == taken_rows.end())
                {
                    // Found new stuff
                    addPermutationMatrixColumns(latticeSites, taken_rows, lat_nbrs, pm_rows, permutationMatrix, col1, true);
                }
            }
        }
    }

    for (size_t i = 0; i < latticeSites.size(); i++)
    {
        std::sort(latticeSites[i].begin(), latticeSites[i].end());
    }

    addOrbitsFromPM(structure, latticeSites);

    // @todo Rename this.
    addPermutationInformationToOrbits(col1, permutationMatrix);

    bool debug = true;
    if (debug)
    {
        checkEquivalentClusters();
    }

    sort();
}

/**
@param orbit orbit to add to orbit list
**/
void OrbitList::addOrbit(const Orbit &orbit) {
    _orbits.push_back(orbit);
}

/**
@param nbody number of bodies for which to filter
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

/// Adds cluster to orbit list, if cluster exists add sites if not create a new orbit
void OrbitList::addClusterToOrbitList(const Cluster &cluster,
                                      const std::vector<LatticeSite> &sites,
                                      std::unordered_map<Cluster, int> &clusterIndexMap)
{
    int orbitNumber = findOrbitIndex(cluster, clusterIndexMap);
    if (orbitNumber == -1)
    {
        Orbit newOrbit = Orbit(cluster);
        addOrbit(newOrbit);
        // add to back (assuming addOrbit does not sort orbit list)
        _orbits.back().addEquivalentSites(sites);
        clusterIndexMap[cluster] = _orbits.size() - 1;
        _orbits.back().sortOrbit();
    }
    else
    {
        _orbits[orbitNumber].addEquivalentSites(sites, true);
    }
}

/**
@details Returns the index of the orbit for which the given cluster is representative.
@param cluster cluster to search for
@returns orbit index; -1 if nothing is found
**/
// int OrbitList::findOrbitIndex(const Cluster &cluster) const
// {
//     for (size_t i = 0; i < _orbits.size(); i++)
//     {
//         if (_orbits[i].getRepresentativeCluster() == cluster)
//         {
//             return i;
//         }
//     }
//     return -1;
// }

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
@param index
@returns copy of orbit
**/
Orbit OrbitList::getOrbit(unsigned int index) const
{
    if (index >= size())
    {
        throw std::out_of_range("Error: Tried accessing orbit at out of bound index. Orbit OrbitList::getOrbit");
    }
    return _orbits[index];
}

/**
@details This function prints information about the orbit list.
@param verbosity control verbosity of information
**/
void OrbitList::print(int verbosity = 0) const
{
    int orbitCount = 0;
    for (const auto &orbit : _orbits)
    {
        std::cout << "Orbit number: " << orbitCount++ << std::endl;
        std::cout << "Representative cluster " << std::endl;
        orbit.getRepresentativeCluster().print();

        std::cout << "Multiplicities: " << orbit.size() << std::endl;
        if (verbosity > 1)
        {
            std::cout << "Duplicates: " << orbit.getNumberOfDuplicates() << std::endl;
        }
        if (verbosity > -1)
        {
            for (auto sites : orbit.getEquivalentSites())
            {
                for (auto site : sites)
                {
                    std::cout << "(" << site.index() << " : [" << site.unitcellOffset()[0] << " " << site.unitcellOffset()[1] << " " << site.unitcellOffset()[2] << "]) . ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

/**
@details
This function adds permutation related information to the orbits.

Algorithm
---------

For each orbit:

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
void OrbitList::addPermutationInformationToOrbits(const std::vector<LatticeSite> &col1,
                                                  const std::vector<std::vector<LatticeSite>> &permutationMatrix)
{
    _col1 = col1;
    _permutationMatrix = permutationMatrix;

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
            auto p_equal_i = getAllColumnsFromSites(translated_rep_sites, col1, permutationMatrix);
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
        std::unordered_set<std::vector<int>, VectorHash> allowedPermutations;
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
                            throw std::runtime_error("Error: did not find any integer permutation from allowed permutation to any translated representative site ");
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
                        throw std::runtime_error("Did not find a permutation of the orbit sites to the permutations of the representative sites");
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
            std::string msg = "";
            msg += "Each set of site did not get a permutations " + std::to_string(sitePermutations.size());
            msg += " != " + std::to_string(_orbits[i].getEquivalentSites().size());
            throw std::runtime_error(msg);
        }

        _orbits[i].setEquivalentSitesPermutations(sitePermutations);
        _orbits[i].setAllowedSitesPermutations(allowedPermutations);
    }
}

/// Will find the sites in col1, extract all columns along with their unit cell translated indistinguishable sites.
std::vector<std::vector<LatticeSite>> OrbitList::getAllColumnsFromSites(const std::vector<LatticeSite> &sites,
                                                                        const std::vector<LatticeSite> &col1,
                                                                        const std::vector<std::vector<LatticeSite>> &permutationMatrix) const
{
    bool sortRows = false;
    std::vector<int> rowsFromCol1 = findRowsFromCol1(col1, sites, sortRows);
    std::vector<std::vector<LatticeSite>> p_equal = getAllColumnsFromRow(rowsFromCol1, permutationMatrix, true, sortRows);
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
@brief Returns all columns from the given rows in permutation matrix
@param includeTranslatedSites If true it will also include the equivalent sites found from the rows by moving each site into the unitcell.
@todo complete description
**/
std::vector<std::vector<LatticeSite>> OrbitList::getAllColumnsFromRow(
    const std::vector<int> &rows,
    const std::vector<std::vector<LatticeSite>> &permutationMatrix,
    bool includeTranslatedSites,
    bool sortIt) const
{
    std::vector<std::vector<LatticeSite>> allColumns;

    for (size_t column = 0; column < permutationMatrix[0].size(); column++)
    {
        std::vector<LatticeSite> indistinctlatticeSites;

        for (const int &row : rows)
        {
            indistinctlatticeSites.push_back(permutationMatrix[row][column]);
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
        throw std::runtime_error("Function getSitesTranslatedToUnitcell received a latticeSite that had a repeated site in the unitcell direction where pbc was false");
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
                throw std::runtime_error("Function getSitesTranslatedToUnitcell translated a latticeSite and got a repeated site in the unitcell direction where pbc was false");
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

/// Debug function to check that all equivalent sites in every orbit give the same sorted cluster.
void OrbitList::checkEquivalentClusters() const
{
    for (const auto &orbit : _orbits)
    {
        Cluster representative_cluster = orbit.getRepresentativeCluster();
        for (const auto &sites : orbit.getEquivalentSites())
        {
            Cluster equivalentCluster = Cluster(_primitiveStructure, sites);
            if (representative_cluster != equivalentCluster)
            {
                std::cout << " found an 'equivalent' cluster that does not match the representative cluster" << std::endl;
                std::cout << "representative_cluster:" << std::endl;
                representative_cluster.print();

                std::cout << "equivalentCluster:" << std::endl;
                equivalentCluster.print();

                throw std::runtime_error("found an 'equivalent' cluster that does not match the representative cluster");
            }
            if (fabs(equivalentCluster.radius() - representative_cluster.radius()) > 1e-3)
            {
                std::cout << " found an 'equivalent' cluster that does not match the representative cluster" << std::endl;
                std::cout << "representative_cluster:" << std::endl;
                representative_cluster.print();

                std::cout << "equivalentCluster:" << std::endl;
                equivalentCluster.print();
                std::cout << " test geometric size: " << icet::getGeometricalRadius(sites, _primitiveStructure) << " " << std::endl;
                throw std::runtime_error("Found an 'equivalent' cluster that does not match the representative cluster");
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
    _orbits.back().sortOrbit();
}

/**
@details From all columns in permutation matrix add all the vector<LatticeSites> from pm_rows
When taking new columns update taken_rows accordingly
**/
void OrbitList::addPermutationMatrixColumns(std::vector<std::vector<std::vector<LatticeSite>>> &latticeSites,
                                            std::unordered_set<std::vector<int>,
                                            VectorHash> &taken_rows,
                                            const std::vector<LatticeSite> &lat_nbrs,
                                            const std::vector<int> &pm_rows,
                                            const std::vector<std::vector<LatticeSite>> &permutationMatrix,
                                            const std::vector<LatticeSite> &col1,
                                            bool add) const
{

    std::vector<std::vector<LatticeSite>> columnLatticeSites;
    columnLatticeSites.reserve(permutationMatrix[0].size());
    for (size_t column = 0; column < permutationMatrix[0].size(); column++)
    {
        std::vector<LatticeSite> indistinctlatticeSites;

        for (const int &row : pm_rows)
        {
            indistinctlatticeSites.push_back(permutationMatrix[row][column]);
        }
        auto translatedEquivalentSites = getSitesTranslatedToUnitcell(indistinctlatticeSites);

        auto sites_index_pair = getMatchesInPM(translatedEquivalentSites, col1);

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

/// Returns the first set of translated sites that exists in col1 of permutationmatrix.
/// @todo Complete description.
std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> OrbitList::getMatchesInPM(
    const std::vector<std::vector<LatticeSite>> &translatedSites,
    const std::vector<LatticeSite> &col1) const
{
    std::vector<int> perm_matrix_rows;
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> matchedSites;
    for (const auto &sites : translatedSites)
    {
        try
        {
            perm_matrix_rows = findRowsFromCol1(col1, sites);
        }
        catch (const std::runtime_error)
        {
            continue;
        }
        // No error here indicating that we found matching rows in col1
        matchedSites.push_back(std::make_pair(sites, perm_matrix_rows));
    }
    if (matchedSites.size() > 0)
    {
        return matchedSites;
    }
    else
    {
        // No matching rows in permutation matrix, this should not happen so we throw an error.
        throw std::runtime_error("Did not find any of the translated sites in col1 of permutation matrix in function getFirstMatchInPM in orbit list");
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
@details This function searches for lattice sites in the first column of the permutation matrix and returns the corresponding rows.
@param sortIt if true the first column will be sorted
@todo Complete description.
**/
std::vector<int> OrbitList::findRowsFromCol1(const std::vector<LatticeSite> &col1,
                                             const std::vector<LatticeSite> &latticeSites,
                                             bool sortIt) const
{
    std::vector<int> rows;
    for (const auto &latticeSite : latticeSites)
    {
        const auto find = std::find(col1.begin(), col1.end(), latticeSite);
        if (find == col1.end())
        {
            throw std::runtime_error("Did not find lattice site in col1 of permutation matrix in function findRowsFromCol1 in many-body neighbor list");
        }
        else
        {
            int row_in_col1 = std::distance(col1.begin(), find);
            rows.push_back(row_in_col1);
        }
    }
    if (sortIt)
    {
        std::sort(rows.begin(), rows.end());
    }
    return rows;
}

/**
@todo Expand description.
@param permutationMatrix permutation matrix
@param sortIt if true (default) the first column will be sorted
**/
std::vector<LatticeSite> OrbitList::getColumn1FromPM(const std::vector<std::vector<LatticeSite>> &permutationMatrix,
                                                     bool sortIt) const
{
    std::vector<LatticeSite> col1;
    col1.reserve(permutationMatrix[0].size());
    for (const auto &row : permutationMatrix)
    {
        col1.push_back(row[0]);
    }
    if (sortIt)
    {
        std::sort(col1.begin(), col1.end());
    }
    return col1;
}

/**
@details This function returns the orbit for a supercell that is associated with a given orbit in the primitive structure.
@param superCell input structure
@param cellOffset offset by which to translate the orbit
@param orbitIndex index of orbit in list of orbits
@param primToSuperMap map from sites in the primitive cell to sites in the supercell
**/
Orbit OrbitList::getSuperCellOrbit(const Structure &superCell,
                                   const Vector3d &cellOffset,
                                   const unsigned int orbitIndex,
                                   std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap) const
{
    if (orbitIndex >= _orbits.size())
    {
        std::string msg = "";
        msg += "Orbit index out of range in OrbitList::getSuperCellOrbit ";
        msg += std::to_string(orbitIndex) + " >= " + std::to_string(_orbits.size());
        throw std::out_of_range(msg);
    }

    Orbit superCellOrbit = _orbits[orbitIndex] + cellOffset;

    auto equivalentSites = superCellOrbit.getEquivalentSites();

    for (auto &sites : equivalentSites)
    {
        for (auto &site : sites)
        {
            transformSiteToSupercell(site, superCell, primToSuperMap);
        }
    }

    superCellOrbit.setEquivalentSites(equivalentSites);
    return superCellOrbit;
}

/**
@details Transforms a site from the primitive structure to a given supercell.
This involves finding a map from the site in the primitive cell to the supercell.
If no map is found mapping is attempted based on the position of the site in the supercell.
@param structure supercell structure
@param primToSuperMap map from primitive to supercell
**/
void OrbitList::transformSiteToSupercell(LatticeSite &site,
                                         const Structure &superCell,
                                         std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap) const
{
    auto find = primToSuperMap.find(site);
    LatticeSite supercellSite;
    if (find == primToSuperMap.end())
    {
        Vector3d sitePosition = _primitiveStructure.getPosition(site);
        supercellSite = superCell.findLatticeSiteByPosition(sitePosition);
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
@param superCell supercell structure
@param cellOffset offset to be applied to sites
@param primToSuperMap map from primitive to supercell
**/
OrbitList OrbitList::getLocalOrbitList(const Structure &superCell,
                                       const Vector3d &cellOffset,
                                       std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap) const
{
    OrbitList localOrbitList = OrbitList();
    localOrbitList.setPrimitiveStructure(_primitiveStructure);

    for (size_t orbitIndex = 0; orbitIndex < _orbits.size(); orbitIndex++)
    {
        localOrbitList.addOrbit(getSuperCellOrbit(superCell, cellOffset, orbitIndex, primToSuperMap));
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
@details Removes from each orbit a specific set of sites in this orbit and the corresponding site permutation.
@param sites the list of sites that will be removed; the order of sites is irrelevant.
 **/
void OrbitList::subtractSitesFromOrbitList(const OrbitList &orbitList)
{
    if (orbitList.size() != size())
    {
        throw std::runtime_error("Orbit lists differ in size in function OrbitList::subtractSitesFromOrbitList");
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
        std::string msg = "";
        msg += "Index " + std::to_string(index) + " was out of bounds in OrbitList::removeOrbit ";
        msg += "OrbitList size is " + std::to_string(size());
        throw std::out_of_range(msg);
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
	    std::string msg = "";
	    msg += "Left and right hand side differ in size in OrbitList& operator+= ";
	    msg += std::to_string(size()) + " != " + std::to_string(rhs_ol.size());
        throw std::runtime_error(msg);
    }

    for (size_t i = 0; i < rhs_ol.size(); i++)
    {
        _orbits[i] += rhs_ol.getOrbit(i);
    }
    return *this;
}
