#include "ManyBodyNeighborList.hpp"

/**
@details
    This function uses @a neighbor_lists to construct all possible
    neighbors up to the given order. The output will be:
    @code{.cpp}
    std::vector<std::pair<originalNeighbors, manyNeigbhors>>
    @endcode

    The many body neigbhors can be retrieved by doing:
    @code{.cpp}
    for (const auto nbr : manyBodyNeighborIndices)
    {
        std::vector<std::pair<int,Vector3d>> neighbors = nbr.first; // this are the first orignal neighbors
        for(const auto manynbr : nbr.second)
        {
            many_body_neigbhor = neighbors;
            many_body_neigbhor.append(manynbr);
        }
    }
    @endcode

    This means that if @a originalNeigbhors.size()==2 then for each lattice site in @a manyNeigbhors
    you can combine it with @a originalNeigbhors to get all triplets that have these first two original neighbors (lattice indices).

    @param neighborLists list of neighbor lists
    @param index
    @param saveBothWays if true then both @a i,j,k and @a j,i,k etc.. will be saved; otherwise only @a i,j,k will be saved if @a i<j<k.
*/

std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> ManyBodyNeighborList::build(const std::vector<NeighborList> &neighborLists,
                                                                                                       int index,
                                                                                                       bool saveBothWays)
{

    if (neighborLists.empty())
    {
        throw std::runtime_error("Error: neigbhorlist vector is empty in ManyBodyNeighborList::build");
    }
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> manyBodyNeighborIndices;

    addSinglet(index, manyBodyNeighborIndices);
    addPairs(index, neighborLists[0], manyBodyNeighborIndices, saveBothWays);

    for (size_t c = 2; c < neighborLists.size() + 2; c++)
    {
        auto Ni = neighborLists[c - 2].getNeighbors(index);
        Vector3d zeroVector = {0.0, 0.0, 0.0};
        std::vector<LatticeSite> currentOriginalNeighbors;
        currentOriginalNeighbors.push_back(LatticeSite(index, zeroVector));  // index is always first index

        combineToHigherOrder(neighborLists[c - 2], manyBodyNeighborIndices, Ni, currentOriginalNeighbors, saveBothWays, c);
    }
    _latticeNeighbors = manyBodyNeighborIndices;
    return manyBodyNeighborIndices;
}

/// Adds singlet from the index to manyBodyNeighborIndices
void ManyBodyNeighborList::addSinglet(const int index,
                                      std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> &manyBodyNeighborIndices) const
{
    Vector3d zeroVector = {0.0, 0.0, 0.0};
    LatticeSite latticeNeighborSinglet = LatticeSite(index, zeroVector);
    std::vector<LatticeSite> singletLatticeSites;
    singletLatticeSites.push_back(latticeNeighborSinglet);

    std::vector<LatticeSite> latticeSitesEmpty;
    manyBodyNeighborIndices.push_back(std::make_pair(singletLatticeSites, latticeSitesEmpty));
}

/// Add all pairs originating from index using neighbor_list
void ManyBodyNeighborList::addPairs(const int index,
                                    const NeighborList &neighborList,
                                    std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> &manyBodyNeighborIndices,
                                    bool saveBothWays) const

{
    Vector3d zeroVector = {0.0, 0.0, 0.0};
    LatticeSite latticeNeighborIndex = LatticeSite(index, zeroVector);

    std::vector<LatticeSite> firstSite = {latticeNeighborIndex};
    std::vector<LatticeSite> Ni = neighborList.getNeighbors(index);
    //exclude smaller neighbors
    if (!saveBothWays)
    {
        Ni = getFilteredNj(Ni, latticeNeighborIndex);
    }

    if (Ni.size() == 0)
    {
        return;
    }
    manyBodyNeighborIndices.push_back(std::make_pair(firstSite, Ni));
}

/**
@todo Revise the entire docstring.

This will use the matrix of equivalent sites @a manyBodyNeighborIndices together with neighbor list @a nl to construct the distinct and indistinct sets of points.

The output will be std::vector<std::vector<std::vector<LatticeSite>>>.
The next outer vector contains the set of indistinct set of lattice neighbors.


Algorithm
=========

The algorithm works by taking the first column of the matrix of equivalent sites
and it will take Ni (lattice neighbors of i) and find the intersection of Ni and col1, intersection(Ni, col1) = Ni_pm
all j in Ni_c1 are then within the cutoff of site i, then depending on the order all the pairs/triplets will be constructed from the
lattice neighbors in Ni_pm.

This will be repeated for all the sites in the neighbor_list. In the end all the pair/triplet terms will have been generated from col1.

Then you will take the first vector<LatticeSites> and find the rows of these LatNbrs in col1 of the permutation matrix.
then you traverse through all the columns in the permutation matrix. These vector<latticeNeighbors> are then indistinct from the original.
Note that here a validity check is made to ensure that atleast one LatticeNeigbhor originate in the original lattice (unitcell offset = [0,0,0])
otherwise we are including a  "ghost cluster".

The new vector<latticeNeighbors> found when traversing the columns are likely to have been found from the combinations in Ni_pm and these must then
be removed/overlooked when moving to the next vector<LatticeSite>.

    for each j in Ni construct the intersect of N_j and N_i = N_ij.
    all k in N_ij are then neighbors with i,j
    what is saved is then i,j and N_ij up to the desired order "maxorder"
*/
void ManyBodyNeighborList::combineToHigherOrder(const NeighborList &nl,
                                                std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> &manyBodyNeighborIndices,
                                                const std::vector<LatticeSite> &Ni,
                                                std::vector<LatticeSite> &currentOriginalNeighbors,
                                                bool saveBothWays,
                                                const size_t maxOrder)
{

    for (const auto &j : Ni)
    {
        //if j is smaller than last added site then continue
        // if bothways = True then don't compare to first
        bool cont = false;

        if (saveBothWays)
        {
            if (currentOriginalNeighbors.size() > 1)
            {
                if (j < currentOriginalNeighbors.back())
                {
                    cont = true;
                }
            }
        }
        else
        {
            if (j < currentOriginalNeighbors.back())
            {
                cont = true;
            }
        }
        if (cont)
        {
            continue;
        }
        // if ((!saveBothWays && currentOriginalNeighbors.size() == 1) && j < currentOriginalNeighbors.back())
        // {
        //     continue;
        // }

        auto originalNeighborCopy = currentOriginalNeighbors;
        originalNeighborCopy.push_back(j); // put j in originalNeigbhors

        auto Nj = nl.getNeighbors(j.index());

        //translate the neighbors
        translateAllNi(Nj, j.unitcellOffset());

        //exclude smaller neighbors
        if (!saveBothWays)
        {
            Nj = getFilteredNj(Nj, j);
        }

        //construct the intersection
        const auto intersection_N_ij = getIntersection(Ni, Nj);

        if (originalNeighborCopy.size() + 1 < maxOrder)
        {
            combineToHigherOrder(nl, manyBodyNeighborIndices, intersection_N_ij, originalNeighborCopy, saveBothWays, maxOrder);
        }

        if (intersection_N_ij.size() > 0 && originalNeighborCopy.size() == (maxOrder - 1))
        {
            manyBodyNeighborIndices.push_back(std::make_pair(originalNeighborCopy, intersection_N_ij));
        }
    }
}

/*
Since N_j is always sorted then simply search for first k in N_j that have k>= j
and then filtered are from indexof(k) to end()

*/
std::vector<LatticeSite> ManyBodyNeighborList::getFilteredNj(const std::vector<LatticeSite> &N_j, const LatticeSite &j) const
{
    auto first = std::upper_bound(N_j.begin(), N_j.end(), j);

    std::vector<LatticeSite> ret(first, N_j.end());
    return ret;
}

/**
    Offsets all indice, offsets pairs in Ni with the input offset, e.g:
    For all j in Ni:
     offset j.offset with "unitCellOffset"

*/
void ManyBodyNeighborList::translateAllNi(std::vector<LatticeSite> &Ni, const Vector3d &offset) const
{
    for (auto &latticeSite : Ni)
    {
        latticeSite.addUnitcellOffset(offset);
    }
}

/// Returns number of manybodies one can make from _latticeNeighbors[index]
size_t ManyBodyNeighborList::getNumberOfSites(const unsigned int index) const
{
    //std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> _latticeNeighbors;
    return _latticeNeighbors[index].second.size();
}

/** Return the many_body neighbor at "firstIndex"  and "secondIndex "
    in _latticeNeighbors and _latticeNeighbors[firstIndex] respectively
*/
std::vector<LatticeSite> ManyBodyNeighborList::getSites(const unsigned int &firstIndex,
                                                            const unsigned int &secondIndex) const
{
    std::vector<LatticeSite> sites = _latticeNeighbors[firstIndex].first;
    //If zero then this is a singlet
    if (getNumberOfSites(firstIndex) > 0)
    {
        sites.push_back(_latticeNeighbors[firstIndex].second[secondIndex]);
    }
    return sites;
}
