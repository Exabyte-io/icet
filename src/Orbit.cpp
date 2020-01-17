#include "Orbit.hpp"

/**
@brief Adds one group of sites that are equivalent to the ones in this orbit.
@param latticeSiteGroup a list of lattice sites to be added
@param sort_orbit if True the orbit will be sorted
*/
void Orbit::addEquivalentSites(const std::vector<LatticeSite> &latticeSiteGroup, bool sort_orbit)
{
    _equivalentSites.push_back(latticeSiteGroup);
    if (sort_orbit)
    {
        sort();
    }
}

/**
@brief Adds a several groups of sites that are equivalent to the ones in this orbit.
@param latticeSiteGroups a list of list of lattice sites to be added
@param sort_orbit if True the orbit will be sorted
*/
void Orbit::addEquivalentSites(const std::vector<std::vector<LatticeSite>> &latticeSiteGroups, bool sort_orbit)
{
    _equivalentSites.insert(_equivalentSites.end(), latticeSiteGroups.begin(), latticeSiteGroups.end());
    if (sort_orbit)
    {
        sort();
    }
}

/**
@brief Returns the equivalent sites at position `index`.
@param index index of site to return
*/
std::vector<LatticeSite> Orbit::getSitesByIndex(unsigned int index) const
{
    if (index >= _equivalentSites.size())
    {
        throw std::out_of_range("Index out of range in function Orbit::getSitesByIndex");
    }
    return _equivalentSites[index];
}

/**
@brief Returns the equivalent sites at position `index` using the permutation of the representative cluster.
@param index index of site to return
*/
std::vector<LatticeSite> Orbit::getPermutedSitesByIndex(unsigned int index) const
{
    if (index >= _equivalentSites.size())
    {
        std::ostringstream msg;
        msg << "Index out of range in function Orbit::getPermutedSitesByIndex (1).\n";
        msg << " index: " << index << "\n";
        msg << " size(_equivalentSites): " << _equivalentSites.size() << "\n";
        throw std::out_of_range(msg.str());
    }
    if (index >= _equivalentSitesPermutations.size())
    {
        std::ostringstream msg;
        msg << "Index out of range in function Orbit::getPermutedSitesByIndex (2).\n";
        msg << " index: " << index << "\n";
        msg << " size(_equivalentSitesPermutations): " << _equivalentSitesPermutations.size();
        throw std::out_of_range(msg.str());
    }
    return icet::getPermutedVector<LatticeSite>(_equivalentSites[index], _equivalentSitesPermutations[index]);
}

/**
@brief Returns all permuted equivalent sites.
*/
std::vector<std::vector<LatticeSite>> Orbit::getPermutedEquivalentSites() const
{
    std::vector<std::vector<LatticeSite>> permutedSites(_equivalentSites.size());
    for (size_t i = 0; i < _equivalentSites.size(); i++)
    {
        permutedSites[i] = getPermutedSitesByIndex(i);
    }
    return permutedSites;
}

/**
Returns the number of exactly equal sites in equivalent sites vector.

This is used among other things to debug orbits when duplicates are not expected.
*/
int Orbit::getNumberOfDuplicates(int verbosity) const
{
    int numberOfEquals = 0;
    for (size_t i = 0; i < _equivalentSites.size(); i++)
    {
        for (size_t j = i + 1; j < _equivalentSites.size(); j++)
        {
            auto i_sites = _equivalentSites[i];
            auto j_sites = _equivalentSites[j];
            // compare the sorted sites
            std::sort(i_sites.begin(), i_sites.end());
            std::sort(j_sites.begin(), j_sites.end());
            if (i_sites == j_sites)
            {
                if (verbosity > 1)
                {
                    std::cout << "Duplicate in orbit: " << i << " " << j << std::endl;
                    if (verbosity > 2)
                    {
                        std::cout << "sites on " << i << std::endl;
                        for (auto i_latnbr : i_sites)
                        {
                            std::cout << i_latnbr << std::endl;
                        }
                        std::cout << "sites on " << j << std::endl;
                        for (auto j_latnbr : j_sites)
                        {
                            std::cout << j_latnbr << std::endl;
                        }
                    }
                }
                numberOfEquals++;
            }
        }
    }
    return numberOfEquals;
}

/**
  @brief Returns the inequivalent MC vectors for this orbit
  @param Mi_local list of the number of allowed sites
 */
std::vector<std::vector<int>> Orbit::getMultiComponentVectors(const std::vector<int> &Mi_local) const
{
    if (std::any_of(Mi_local.begin(), Mi_local.end(), [](const int i) { return i < 2; }))
    {
        std::vector<std::vector<int>> emptyVector;
        return emptyVector;
    }
    auto allMCVectors = getAllPossibleMultiComponentVectorPermutations(Mi_local);
    std::sort(allMCVectors.begin(), allMCVectors.end());
    std::vector<std::vector<int>> distinctMCVectors;
    for (const auto &mcVector : allMCVectors)
    {
        std::vector<std::vector<int>> permutedMCVectors;
        for (const auto &allowedPermutation : _allowedSitesPermutations)
        {
            permutedMCVectors.push_back(icet::getPermutedVector<int>(mcVector, allowedPermutation));
        }
        std::sort(permutedMCVectors.begin(),permutedMCVectors.end());


        // if not any of the vectors in permutedMCVectors exist in distinctMCVectors
        if (!std::any_of(permutedMCVectors.begin(), permutedMCVectors.end(), [&](const std::vector<int> &permMcVector) { return !(std::find(distinctMCVectors.begin(), distinctMCVectors.end(), permMcVector) == distinctMCVectors.end()); }))
        {
            distinctMCVectors.push_back(mcVector);
        }
    }
    return distinctMCVectors;
}

/// Similar to get all permutations but needs to be filtered through the number of allowed elements
std::vector<std::vector<int>> Orbit::getAllPossibleMultiComponentVectorPermutations(const std::vector<int> &Mi_local) const
{

    std::vector<std::vector<int>> cartesianFactors(Mi_local.size());
    for (size_t i = 0; i < Mi_local.size(); i++)
    {
        for (int j = 0; j < Mi_local[i] - 1; j++) // N.B minus one so a binary only get one cluster function
        {
            cartesianFactors[i].push_back(j);
        }
    }

    std::vector<std::vector<int>> allPossibleMCPermutations;
    std::vector<int> firstVector(Mi_local.size(), 0);

    do
    {
        allPossibleMCPermutations.push_back(firstVector);
    } while (icet::nextCartesianProduct(cartesianFactors, firstVector));

    return allPossibleMCPermutations;
}

/**
@details Check if this orbit contains a set of sites in its equivalent sites vector.
@param sites the sites that will be looked for.
@param sorted if true the order of sites is irrelevant

@returns true if the sites exist in equivalent sites.
**/
bool Orbit::contains(const std::vector<LatticeSite> sites, bool sorted) const
{
    auto sitesCopy = sites;
    if (sorted)
    {
        std::sort(sitesCopy.begin(), sitesCopy.end());
    }

    for (size_t i = 0; i < _equivalentSites.size(); i++)
    {
        auto i_sites = _equivalentSites[i];

        // compare the sorted sites
        if (sorted)
        {
            std::sort(i_sites.begin(), i_sites.end());
        }

        if (i_sites == sitesCopy)
        {
            return true;
        }
    }
    return false;
}


/**
@details Removes sets of sites in equivalent sites for which any site in the set has an index equal to indexToRemove.
@param indexToRemove the index to look for
@param onlyConsiderZeroOffset if true remove only sites with zero offset
**/
void Orbit::removeSitesWithIndex(const size_t indexToRemove, bool onlyConsiderZeroOffset)
{
    for (int i = _equivalentSites.size() - 1; i >= 0; i--)
    {
        if (onlyConsiderZeroOffset)
        {
            if (std::any_of(_equivalentSites[i].begin(), _equivalentSites[i].end(), [=](LatticeSite &ls) { return ls.index() == indexToRemove && ls.unitcellOffset().norm() < 1e-4; }))
            {
                _equivalentSites.erase(_equivalentSites.begin() + i);
                _equivalentSitesPermutations.erase(_equivalentSitesPermutations.begin() + i);
            }
        }
        else
        {
            if (std::any_of(_equivalentSites[i].begin(), _equivalentSites[i].end(), [=](LatticeSite &ls) { return ls.index() == indexToRemove; }))
            {
                _equivalentSites.erase(_equivalentSites.begin() + i);
                _equivalentSitesPermutations.erase(_equivalentSitesPermutations.begin() + i);
            }
        }
    }
}


/**
@details Removes sets of sites in equivalent sites for which no site in the set has an index equal to index.
@param index the index to look for
@param onlyConsiderZeroOffset if true only look for sites with zero offset
*/
void Orbit::removeSitesNotWithIndex(const size_t index, bool onlyConsiderZeroOffset)
{
    for (int i = _equivalentSites.size() - 1; i >= 0; i--)
    {
        if (onlyConsiderZeroOffset)
        {
            if (std::none_of(_equivalentSites[i].begin(), _equivalentSites[i].end(), [=](LatticeSite &ls) { return ls.index() == index && ls.unitcellOffset().norm() < 1e-4; }))
            {
                _equivalentSites.erase(_equivalentSites.begin() + i);
                _equivalentSitesPermutations.erase(_equivalentSitesPermutations.begin() + i);
            }
        }
        else
        {
            if (std::none_of(_equivalentSites[i].begin(), _equivalentSites[i].end(), [=](LatticeSite &ls) { return ls.index() == index; }))
            {
                _equivalentSites.erase(_equivalentSites.begin() + i);
                _equivalentSitesPermutations.erase(_equivalentSitesPermutations.begin() + i);
            }
        }
    }
}

/**
@details Removes a specific set of sites in this orbit and the corresponding site permutation.
@param sites vector of sites that will be removed, order of sites is irrelevant
*/
void Orbit::removeSites(std::vector<LatticeSite> sites)
{

    std::sort(sites.begin(), sites.end());
    for (size_t i = 0; i < _equivalentSites.size(); i++)
    {
        auto i_sites = _equivalentSites[i];

        //compare the sorted sites

        std::sort(i_sites.begin(), i_sites.end());

        if (i_sites == sites)
        {
            _equivalentSites.erase(_equivalentSites.begin() + i);
            _equivalentSitesPermutations.erase(_equivalentSitesPermutations.begin() + i);
            return;
        }
    }
    throw std::runtime_error("did not find any mathcing sites in Orbit::removeSites");
}

namespace std {

    /// Stream operator.
    ostream& operator<<(ostream& os, const Orbit& orbit)
    {
        for (const auto sites : orbit._equivalentSites)
        {
            os << "  ";
            for (const auto site : sites)
            {
                os << " " << site;
            }
        }
        return os;
    }

}
