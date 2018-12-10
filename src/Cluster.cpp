#include "Cluster.hpp"

/**
@details Create an instance of a cluster.
@param structure icet structure object
@param latticeSites list of lattice sites that form the cluster
@param sorted cluster will be sorted if True
@param tag integer identifier
*/
Cluster::Cluster(const Structure &structure,
                 const std::vector<LatticeSite> &latticeSites,
                 const bool sorted,
                 const int tag)
{
    /// @todo remove class/function specific numerical tolerances.
    _symprec = 1e-6;

    size_t clusterSize = latticeSites.size();
    std::vector<int> sites(clusterSize);
    std::vector<double> distances;
    distances.reserve((clusterSize * (clusterSize - 1) / 2));
    for (size_t i = 0; i < latticeSites.size(); i++)
    {
        sites[i] = structure.getUniqueSite(latticeSites[i].index());
        for (size_t j = i + 1; j < latticeSites.size(); j++)
        {
            double distance = roundDouble(structure.getDistance(latticeSites[i].index(),
                                                                latticeSites[j].index(),
                                                                latticeSites[i].unitcellOffset(),
                                                                latticeSites[j].unitcellOffset()));

            distances.push_back(distance);
        }
    }
    _sites = sites;
    _distances = distances;
    _tag = tag;
    _radius = icet::getGeometricalRadius(latticeSites, structure);
    _sorted = sorted;
    if (_sorted) sort();
}

/**
@details This function sorts the sites and their corresponding distances. This
ensures the uniqueness of the cluster representation and enables sorting using
class specific operators.

Algorithm for sorting
---------------------
First the distances associated with each site \a i are sorted..

If there is one smallest distance then \a i is the first site and its first
neighbor is the second site and so on/

There are then some special cases which need to be distinguished.

@todo clean up docstring

Case 1:
    Two or more distances are equal

    Problem:
        Which site is picked as the first?

Case 2:
    Two or more distances are equal in \a i_dist.

    Problem:
        It is unclear which site is site 2, 3 (if \a i_dist[0]==i_dist[1]).

Case 3:
    Two or more distances are equal in \a j_dist where \a j is in \a i_nbrs.

    Problem:
        What is the problem here?

Solution for case 1:
    1) assume case 2 and 3 do not apply.
        a) get "getDistIndices" and getReorderedSites for each case
        b) compare to each other. The lowest should be the one to use
    2) if case 2 applies as well:
        a) for each smallest \a i_dist get solution from case 2
        b) compare case 2 solutions of \a i_dist to find minimum.

Solution case 2:
    for each \a j,k,l.. indices in \a i_nbr that have \a i_dists equal:
        @code
        for combination in all_combinations(j,k,l,...):
            get "getDistIndices" and getReorderedSites
        @endcode
        take the \a get_dist_indices and reordered sites that are the smallest
        and use the corresponding combination of indices.
*/
void Cluster::sort()
{
    if (_distances.size() == 0)
    {
        return;
    }

    std::vector<LocalEnvironment> localEnvironments(_sites.size());
    for (size_t i = 0; i < _sites.size(); i++)
    {
        localEnvironments[i] = getLocalEnvironment(i);
    }
    std::sort(localEnvironments.begin(), localEnvironments.end());

    std::vector<int> minimumOrder = localEnvironments[0].getClusterIndices();
    if (minimumOrder.size() != _sites.size())
    {
        throw("minimumOrder.size != _sites.size()");
    }

    std::vector<double> min_distance;
    std::vector<int> min_sites;
    std::vector<int> min_indices;

    auto equal_minimum_first_sites = getEqualMinimumFirstSites(localEnvironments);
    if (equal_minimum_first_sites.size() > 1)
    {
        // Case 1
        auto min_data = case1_min_indices(equal_minimum_first_sites);
        min_distance = std::get<0>(min_data);
        min_sites = std::get<1>(min_data);
        min_indices = std::get<2>(min_data);
    }
    else
    {
        // Case 2
        auto min_data = case2_min_indices(localEnvironments[0]);
        min_distance = std::get<0>(min_data);
        min_sites = std::get<1>(min_data);
        min_indices = std::get<2>(min_data);
    }

    // Some validation of algorithm and debugging
    if (min_distance.size() != _distances.size())
    {
        std::cout << "min_distances.size() = " << min_distance.size()
                  << " _distances.size() = " << _distances.size() << std::endl;
        throw std::runtime_error("Algorithm ended up with wrong size for distances");
    }

    if (min_sites.size() != _sites.size())
    {
        throw std::runtime_error("Algorithm ended up with wrong size for sites");
    }
    _distances = min_distance;
    _sites = min_sites;
}


// CONTINUE CLEAN-UP HERE


/// Brute force attempt to find minimum clusters
std::tuple<std::vector<int>, std::vector<double>, std::vector<int>> Cluster::findMinimumClusterByBruteForce()
{

    std::vector<int> atomic_order(_sites.size());
    for (size_t i = 0; i < atomic_order.size(); i++)
    {
        atomic_order[i] = i;
    }
    auto min_order = atomic_order;
    auto min_dists = getReorderedDistances(atomic_order);
    auto min_sites = getReorderedSites(atomic_order);

    do
    {
        auto dists = getReorderedDistances(atomic_order);
        auto sites = getReorderedSites(atomic_order);
        if (compare_sites_dists(dists, sites, min_dists, min_sites))
        {
            min_dists = dists;
            min_sites = sites;
            min_order = atomic_order;
        }

    } while (std::next_permutation(atomic_order.begin(), atomic_order.end()));

    return std::make_tuple(min_order, min_dists, min_sites);
}


/**
This validates the sorting by testing all possible
combinations of sites and sees if there is a lower state of the cluster.
*/
void Cluster::validateSorting()
{

    auto minBruteForce = findMinimumClusterByBruteForce();
    auto bruteforce_order = std::get<0>(minBruteForce);
    auto bruteForceDists = std::get<1>(minBruteForce);
    auto bruteForceSites = std::get<2>(minBruteForce);

    if (compare_sites_dists(bruteForceDists, bruteForceSites, _distances, _sites))
    {
        std::cout << " bruteforce dists, sites and order:" << std::endl;
        for (auto d : bruteForceDists)
        {
            std::cout << d << " ";
        }
        std::cout << std::endl;
        for (auto d : bruteForceSites)
        {
            std::cout << d << " ";
        }
        std::cout << std::endl;
        for (auto d : bruteforce_order)
        {
            std::cout << d << " ";
        }
        std::cout << std::endl;
        std::cout << " algorithm result for dists and sites:" << std::endl;
        for (auto d : _distances)
        {
            std::cout << d << " ";
        }
        std::cout << std::endl;
        for (auto d : _sites)
        {
            std::cout << d << " ";
        }
        std::cout << std::endl;

        throw std::runtime_error("Error: brute force found a smaller");
    }
}


/**
Get all equal nearestNeighborDistances that are identical to the minimum one
Equality means all distances and sites are equal.
*/
std::vector<LocalEnvironment> Cluster::getEqualMinimumFirstSites(const std::vector<LocalEnvironment> &i_neighbors) const
{
    std::vector<LocalEnvironment> equalFirstDists;
    equalFirstDists.push_back(i_neighbors[0]);

    for (size_t i = 1; i < i_neighbors.size(); i++)
    {

        if (i_neighbors[i] == i_neighbors[0])
        {
            equalFirstDists.push_back(i_neighbors[i]);
        }
        else
        {
            break;
        }
    }

    // std::cout << " Found " << equalFirstDists.size() << " equal first dists" << std::endl;
    // std::cout << "Minimum and second minimum: " << std::endl;
    // for (size_t i = 0; i < i_neighbors.size(); i++)
    // {
    //     i_neighbors[i].print();
    // }

    return equalFirstDists;
}

/**
Returns true if dist_index1 is equal to dist_index2

THe hitch is that dist_index1[i].second is a index in the cluster and has to be checked what site is on that
index when comparing

*/
bool Cluster::isEqualFirstDists(const std::vector<std::pair<double, int>> &dist_index1, const std::vector<std::pair<double, int>> &dist_index2) const
{
    for (size_t i = 0; i < dist_index1.size(); i++)
    {
        if (!(dist_index1[i].first == dist_index2[i].first && _sites[dist_index1[i].second] == _sites[dist_index2[i].second]))
        {
            return false;
        }
    }
    return true;
}


/**


Arguments:
    equal_minimum_i_neighbors
    a vector of all the minimum first dists
Do case 1:
    solutions case 1:
    1) assume case2 and case3 is not active.
        a) get "getDistIndices" and getReorderedSites for each case
        b) compare to each other. The lowest should be the one to use
    2) if case2 is active also:
       a )for each smallest i_dist:
            get solution from case 2
        b) compare case2 solutions of i_dist to find minimum.

*/

std::tuple<std::vector<double>, std::vector<int>, std::vector<int>> Cluster::case1_min_indices(const std::vector<LocalEnvironment> &equal_minimum_i_neighbors) const
{

    auto minOrder = equal_minimum_i_neighbors[0].getClusterIndices();
    auto minDistance = getReorderedDistances(minOrder);
    auto minSites = getReorderedSites(minOrder);

    if (isCase2(equal_minimum_i_neighbors[0]))
    {
        // std::cout << "case 2 inside case1" << std::endl;
        for (size_t i = 0; i < equal_minimum_i_neighbors.size(); i++)
        {
            //case2_min_indices(const int i_index, const std::vector<std::pair<double, int>> &i_dist)
            auto case_2_solution = case2_min_indices(equal_minimum_i_neighbors[i]);

            auto min_distances_trial = std::get<0>(case_2_solution);
            auto min_sites_trial = std::get<1>(case_2_solution);
            auto min_order_trial = std::get<2>(case_2_solution);

            if (compare_sites_dists(min_distances_trial, min_sites_trial, minDistance, minSites))
            {
                minOrder = min_order_trial;
                minDistance = min_distances_trial;
                minSites = min_sites_trial;
            }
        }
    }
    else
    {
        // std::cout << "NOT case 2 inside case1" << std::endl;
        for (size_t i = 0; i < equal_minimum_i_neighbors.size(); i++)
        {

            auto min_order_trial = equal_minimum_i_neighbors[i].getClusterIndices();
            auto min_distances_trial = getReorderedDistances(min_order_trial);
            auto min_sites_trial = getReorderedSites(min_order_trial);

            if (compare_sites_dists(min_distances_trial, min_sites_trial, minDistance, minSites))
            {
                minOrder = min_order_trial;
                minDistance = min_distances_trial;
                minSites = min_sites_trial;
            }
        }
    }
    return std::make_tuple(minDistance, minSites, minOrder);
}


/**
@details Rearranges distances and sites to the order given in minimumOrder.

example:
-------
minimumOrder[0] = current index which should be first place
minimumOrder[1] = current index which should be second place
and so on ...

This is a slightly non trivial operation since swapping two sites changes
the order of the distances in a complex way.

The distances are ordered by:
 d_12, d_13, d_14, ... , d_23, d_24, ...


 solution:
 ---------
 The algorithm creates a vector of tuples with i, j, distance

 the current state is: 1, 2, d_12
                       1, 3, d_13


which is then mapped to:
                       minOrder.index_of(1), minOrder.index_of(2), d_12
                       minOrder.index_of(1), minOrder.index_of(3), d_13


where minOrder.index_of(1) = index of 1 in minOrder.
this vector is then sorted and the positions will align to the new order.

*/
void Cluster::setThisOrder(const std::vector<int> &minimumOrder)
{
    _sites = getReorderedSites(minimumOrder);
    _distances = getReorderedDistances(minimumOrder);
}

//get count of a specific element vector
// int Cluster::getCount(const std::vector<int> &elements) const
// {
//     const auto find = _element_counts.find(elements);
//     if (find == _element_counts.end())
//     {
//         return 0;
//     }
//     else
//     {
//         return _element_counts.at(elements);
//     }
// }

/**
Gets the min order of indices as given by this "first dists"

this format is used throughout this code to simpler extract the indices from a given first dist (i_dists)

The format is
nearestNeighborDistances.first = a vector of distances and indices to the corresponding site the distance is to
nearestNeighborDistances.second.first = the site of this index
nearestNeighborDistances.second.second the index of this nearestNeighborDistances, i.e. the i in i_dists
*/

std::vector<int> Cluster::getOrderFromFirstDists(const std::pair<std::vector<std::pair<double, int>>, std::pair<int, int>> &nearestNeighborDistances) const
{
    std::vector<int> minOrder(_sites.size());

    int counter = 0;
    minOrder[counter++] = nearestNeighborDistances.second.second;
    for (const auto &dist_index_pair : nearestNeighborDistances.first)
    {
        minOrder[counter++] = dist_index_pair.second;
    }
    return minOrder;
}

/**
    Checks if these distance, sites vector has some equal elements
     and is thus subject to case2
*/
bool Cluster::isCase2(const LocalEnvironment &i_neighbors) const
{
    for (size_t i = 0; i < i_neighbors.neighborDistances().size() - 1; i++)
    {
    if (i_neighbors.neighborDistances()[i] == i_neighbors.neighborDistances()[i + 1] && i_neighbors.neighborSites()[i] == i_neighbors.neighborSites()[i + 1])
        {
            return true;
        }
    }
    return false;
    //return std::adjacent_find(nearestNeighborDistances.begin(), nearestNeighborDistances.end()) == nearestNeighborDistances.end();
}

/**
The distances are ordered by:

 d_12, d_13, d_14, ... , d_23, d_24, ...

 return a tuple of the current state:
                       1, 2, d_12
                       1, 3, d_13

*/
std::vector<std::tuple<int, int, double>> Cluster::getDistIndices() const
{

    std::vector<std::tuple<int, int, double>> dist_indices;
    int counter = 0;
    for (size_t k = 0; k < _sites.size(); k++)
    {
        for (size_t l = k + 1; l < _sites.size(); l++)
        {
            dist_indices.push_back(std::make_tuple(k, l, _distances[counter++]));
        }
    }
    return dist_indices;
}

/**
 The algorithm creates a vector of tuples with i, j, distance

 the current state is: 1, 2, d_12
                       1, 3, d_13


which is then mapped to:
                       indiceOrder.index_of(1), indiceOrder.index_of(2), d_12
                       indiceOrder.index_of(1), indiceOrder.index_of(3), d_13


where indiceOrder.index_of(1) = index of 1 in indiceOrder.
this vector is then sorted and the positions will align to the new order.


*/
std::vector<std::tuple<int, int, double>> Cluster::getDistIndices(const std::vector<int> &indiceOrder) const
{

    std::vector<std::tuple<int, int, double>> dist_indices;
    dist_indices.reserve(_distances.size());
    int counter = 0;
    for (size_t k = 0; k < _sites.size(); k++)
    {
        for (size_t l = k + 1; l < _sites.size(); l++)
        {

            auto find_first = std::find(indiceOrder.begin(), indiceOrder.end(), k);
            auto find_second = std::find(indiceOrder.begin(), indiceOrder.end(), l);

            int first = std::distance(indiceOrder.begin(), find_first);
            int second = std::distance(indiceOrder.begin(), find_second);

            if (second < first)
            {
                auto temp = first;
                first = second;
                second = temp;
            }
            dist_indices.push_back(std::make_tuple(first, second, _distances[counter++]));
        }
    }

    std::sort(dist_indices.begin(), dist_indices.end());

    return dist_indices;
}

/**
Get the distances if the sites would have been rearranged according to indiceOrder

returns a double (distance ) vector
*/
std::vector<double> Cluster::getReorderedDistances(const std::vector<int> &indiceOrder) const
{
    auto dist_indices = getDistIndices(indiceOrder);
    int counter = 0;
    std::vector<double> reorderedDistances(_distances.size());
    for (const auto &tup : dist_indices)
    {
        reorderedDistances[counter++] = std::get<2>(tup);
    }
    return reorderedDistances;
}

/**
Return the sites if the order would have been as is given in indiceOrder
*/
std::vector<int> Cluster::getReorderedSites(const std::vector<int> &indiceOrder) const
{
    std::vector<int> tempSites(_sites.size());
    for (size_t i = 0; i < indiceOrder.size(); i++)
    {
        tempSites[i] = _sites[indiceOrder[i]];
    }
    return tempSites;
}

/**
solutions case 2:
    for each j,k,l.. indices in i_nbr that have i_dists equal:
        for combination in all_combinations(j,k,l,...):
            get "getDistIndices" and getReorderedSites
        take the get_dist_indices and reordered sites that are smallest and
        use the corresponding combination of indices.

returns min_indices, correspong distances and corresponding sites
*/
std::tuple<std::vector<double>, std::vector<int>, std::vector<int>> Cluster::case2_min_indices(const LocalEnvironment &i_neighbor) const
{

    std::vector<int> minimumOrder = i_neighbor.getClusterIndices();

    std::vector<double> min_distances = getReorderedDistances(minimumOrder);
    std::vector<int> min_sites = getReorderedSites(minimumOrder);
    std::vector<int> min_indices = minimumOrder;
    // std::cout << "original distances " << std::endl;
    // for (auto d : min_distances)
    // {
    //     std::cout << d << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "printing i_neighbor " << std::endl;
    // i_neighbor.print();
    // std::cout << " Min order" << std::endl;
    // for (auto d : minimumOrder)
    // {
    //     std::cout << d << " ";
    // }
    // std::cout << std::endl;

    //identical indices is a vector of vectors
    // the identical indices are given in the global indices in the cluster
    std::vector<std::vector<int>> identicalIndices = i_neighbor.getEquivalentIndices();

    //return if no dists, sites are equal
    if (identicalIndices.size() == 0)
    {
        // std::cout << "no dists equal:" << std::endl;
        // i_neighbor.print();
        return std::make_tuple(min_distances, min_sites, min_indices);
    }

    // for (auto identVec : identicalIndices)
    // {
    //     for (auto d : identVec)
    //     {
    //         std::cout << d << " ";
    //     }
    //     std::cout << std::endl;
    // }

    if (minimumOrder.size() != _sites.size())
    {
        throw std::runtime_error("Minimumorder in case 2 is not equal in size to _sites");
    }
    //sort identical indices so they appear in order in terms of indice in minimumOrder:
    //this is needed when taking the permutations of the identical sites
    for (auto &vec : identicalIndices)
    {
        std::sort(vec.begin(), vec.end());
    }

    //do all permutations of the identical dists, sites
    // identIndices = std::vector<std::pair<int, int>>

    //this i_neighbor has the right first indice and site

    auto trial_order = minimumOrder;
    findMinimumIndexPermutation(0, identicalIndices, minimumOrder, trial_order, min_distances, min_sites, min_indices);

    return std::make_tuple(min_distances, min_sites, min_indices);
}

void Cluster::findMinimumIndexPermutation(size_t currentIndexSet,
					  const std::vector<std::vector<int>> &identicalIndices,
					  const std::vector<int> &minimumOrder,
					  std::vector<int> trial_order,
					  std::vector<double> &min_distances,
					  std::vector<int> &min_sites,
					  std::vector<int> &min_indices) const
{
    std::vector<int> identicalIndexSet = identicalIndices[currentIndexSet];
    //trial_order = minimumOrder;
    do
    {
        // trial_order = minimumOrder;
        for (size_t i = 0; i < identicalIndexSet.size(); i++)
        {
            trial_order[identicalIndexSet[i]] = minimumOrder[identicalIndices[currentIndexSet][i]];
        }

        auto distances_trial = getReorderedDistances(trial_order);
        auto trial_sites = getReorderedSites(trial_order);

        if (compare_sites_dists(distances_trial, trial_sites, min_distances, min_sites))
        {
            min_distances = distances_trial;
            min_sites = trial_sites;
            min_indices = trial_order;
        }
        if (currentIndexSet + 1 < identicalIndices.size())
        {
            findMinimumIndexPermutation(currentIndexSet + 1, identicalIndices, minimumOrder, trial_order, min_distances, min_sites, min_indices);
        }

    } while (std::next_permutation(identicalIndexSet.begin(), identicalIndexSet.end()));
}

/**
compare distances and sites if dist1 and sites1 < dists2 and sites2
(first compare distances, if they are equal compare sites)
*/
bool Cluster::compare_sites_dists(const std::vector<double> &dist1, const std::vector<int> &sites1, const std::vector<double> &dist2, const std::vector<int> &sites2) const
{

    for (size_t i = 0; i < dist1.size(); i++)
    {
        if (dist1[i] < dist2[i])
        {
            // std::cout<<"true "<< dist1[i]<< " "<< dist2[i]<<std::endl;
            return true;
        }
        else if (dist1[i] > dist2[i])
        {
            // std::cout<<"false "<< dist1[i]<< " "<< dist2[i]<<std::endl;
            return false;
        }
    }
    if (dist1 < dist2)
    {
        // std::cout<<"true: dist1< dist2"<<std::endl;
        return true;
    }
    if (dist1 > dist2)
    {
        // std::cout<<"false: dist1 > dist2"<<std::endl;
        return false;
    }

    // std::cout<<"return sites sites1 < sites2 "<< std::boolalpha<< (sites1 < sites2)<<std::endl;

    return sites1 < sites2;
}

/*
Get all dists, sites and indices that origin from site i

Returns a I_Neighbors structure
*/
LocalEnvironment Cluster::getLocalEnvironment(const size_t i_index)
{
    std::vector<std::tuple<double, int, int>> dists_site_index;
    size_t counter = 0;
    for (size_t k = 0; k < _sites.size(); k++)
    {
        for (size_t l = k + 1; l < _sites.size(); l++)
        {
            if (k == i_index or l == i_index)
            {
                if (k != i_index)
                {
                    dists_site_index.push_back(std::make_tuple(_distances[counter], _sites[k], k)); //Change to sites here or else it will mess up the sorting
                }
                else
                {
                    dists_site_index.push_back(std::make_tuple(_distances[counter], _sites[l], l));
                }
            }
            counter++;
        }
    }
    if (counter != _distances.size())
    {
        std::cout << "Error: count not equal to distance size " << _distances.size() << " counter " << counter << std::endl;
        std::cout << "Sites size: " << _sites.size() << std::endl;
        throw std::out_of_range(" count not equal to distance size");
    }
    std::sort(dists_site_index.begin(), dists_site_index.end());
    std::vector<double> distances(dists_site_index.size());
    std::vector<int> sites(dists_site_index.size());
    std::vector<int> indices(dists_site_index.size());

    for (size_t i = 0; i < dists_site_index.size(); i++)
    {
        distances[i] = std::get<0>(dists_site_index[i]);
        sites[i] = std::get<1>(dists_site_index[i]);
        indices[i] = std::get<2>(dists_site_index[i]);
    }
    LocalEnvironment i_nbrs = LocalEnvironment(i_index, _sites[i_index], distances, sites, indices);

    return i_nbrs;
}

/*
Swap sites of site i and site j and also changes the corresponding distances.
*/
void Cluster::swapSites(const int i, const int j)
{
    //swap sites
    std::iter_swap(_sites.begin() + i, _sites.begin() + j);

    std::vector<std::tuple<int, int, double>> dist_indices;
    int counter = 0;
    for (size_t k = 0; k < _sites.size(); k++)
    {
        for (size_t l = k; l < _sites.size(); l++)
        {
            int first = k;
            int second = l;
            if (first == i)
            {
                first = j;
            }
            if (first == j)
            {
                first = i;
            }
            if (second == i)
            {
                second = j;
            }
            if (second == j)
            {
                second = i;
            }

            if (second < first)
            {
                auto temp = first;
                first = second;
                second = temp;
            }
            dist_indices.push_back(std::make_tuple(first, second, _distances[counter++]));
        }
    }

    std::sort(dist_indices.begin(), dist_indices.end());
    counter = 0;
    for (const auto &tup : dist_indices)
    {
        _distances[counter++] = std::get<2>(tup);
    }
}

/**
Print the cluster to standard out.

Format is first distances then sites
*/
void Cluster::print() const
{
    for (const auto d : _distances)
    {
        std::cout << d << " ";
    }
    std::cout << " :: ";
    for (const auto s : _sites)
    {
        std::cout << s << " ";
    }
    std::cout << _radius << " ";
    std::cout << std::endl;

}
