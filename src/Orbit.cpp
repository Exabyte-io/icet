#include "Orbit.hpp"

Orbit::Orbit(const Cluster &cluster)
{
    _representativeCluster = cluster;
}

/**
Returns the number of exactly equal sites in equivalent sites vector

This is used among other things to debug orbits when duplicates is not expected
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
            //compare the sorted sites
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
                            i_latnbr.print();
                        }
                        std::cout << "sites on " << j << std::endl;
                        for (auto j_latnbr : j_sites)
                        {
                            j_latnbr.print();
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
  @Brief Returns the inequivalent MC vectors for this orbit
    @Param
        Mi_local : Vector of the number of allowed sites

 */
std::vector<std::vector<int>> Orbit::getMCVectors(const std::vector<int> &Mi_local) const
{
    if( std::any_of(Mi_local.begin(), Mi_local.end(), [](const int i){ return  i<2; }))
    {
        std::vector<std::vector<int>> emptyVector;
        return emptyVector;
    }
    auto allMCVectors = getAllPossibleMCVectorPermutations(Mi_local);
    std::sort(allMCVectors.begin(), allMCVectors.end());
    std::vector<std::vector<int>> distinctMCVectors;
    for(const auto &mcVector : allMCVectors)
    {
        std::vector<std::vector<int>> permutatedMCVectors;
        for(const auto &allowedPermutation : _allowedSitesPermutations )
        {
            permutatedMCVectors.push_back(icet::getPermutatedVector<int>(mcVector, allowedPermutation));
        }
        if(! std::any_of(permutatedMCVectors.begin(), permutatedMCVectors.end(),[&](const std::vector<int> &permMcVector){
                 return !(std::find(distinctMCVectors.begin(),distinctMCVectors.end(),permMcVector) == distinctMCVectors.end()); }  ))
        {
            distinctMCVectors.push_back(mcVector);
        }
    }
    return distinctMCVectors;
}

///Similar to get all permutations but needs to be filtered through the number of allowed elements
std::vector<std::vector<int>> Orbit::getAllPossibleMCVectorPermutations(const std::vector<int> &Mi_local) const
{

    std::vector<std::vector<int>> cartesianFactors(Mi_local.size());
    for (int i = 0; i < Mi_local.size(); i++)
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
    } while (icet::next_cartesian_product(cartesianFactors, firstVector));

    return allPossibleMCPermutations;
}
