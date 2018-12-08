#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <pybind11/stl.h>
#include <map>
#include <unordered_map>
#include <algorithm>
#include "Structure.hpp"
#include "LatticeSite.hpp"
#include "Geometry.hpp"
#include "LocalEnvironment.hpp"

#include <boost/functional/hash.hpp>
using boost::hash;
using boost::hash_combine;
using boost::hash_value;

using namespace Eigen;

namespace py = pybind11;

/**
@brief This class handles information pertaining to a single cluster.
*/
class Cluster
{
  public:

    // Empty constructor
    Cluster() { }

    /// Create cluster from a structure and a set of lattice sites.
    Cluster(const Structure &structure,
            const std::vector<LatticeSite> &latticeSites,
            const bool sorted = true,
            const int tag = 0);

    /// Sort cluster.
    void sort();

    /// Validate sorting of clusters.
    void validateSorting();

    /// Swap two sites.
    void swapSites(const int, const int);

    /// Print the cluster to stdout.
    void print() const;

  public:

    /// Returns the order (i.e., the number of sites) of the cluster.
    unsigned int order() const { return _sites.size(); }

    /// Returns the radius of the cluster.
    double radius() const { return _radius; }

    // Returns the sites in the cluster.
    std::vector<int> sites() const { return _sites; }

    // Returns distances between points in the  cluster.
    std::vector<double> distances() const { return _distances; }

    /// Returns the local environment of a site (point) in the cluster.
    LocalEnvironment getLocalEnvironment(const size_t);

    /// Returns true if this is a sorted cluster
    bool isSorted() const { return _sorted; }

    /// Returns the cluster tag used for identification (if distances/sites are not used for distinction).
    int tag() const { return _tag; }

    /// Set the cluster tag and mark the cluster as unsorted.
    void setTag(const int tag)
    {
        _sorted = false;
        _tag = tag;
    }

// CONTINUE CLEAN UP HERE

    /// Brute force attempt to find minimum clusters
    std::tuple<std::vector<int>, std::vector<double>, std::vector<int>> findMinimumClusterByBruteForce();

    /// Find all equal nearestNeighborDistances that are identical to the minimum one.
    std::vector<LocalEnvironment> getEqualMinimumFirstSites(const std::vector<LocalEnvironment> &i_neighbors) const;

    /// Returns true if dist_index1 is equal to dist_index2
    bool isEqualFirstDists(const std::vector<std::pair<double, int>> &, const std::vector<std::pair<double, int>> &) const;

    std::tuple<std::vector<double>, std::vector<int>, std::vector<int>> case1_min_indices(const std::vector<LocalEnvironment> &) const;

    std::vector<int> getOrderFromFirstDists(const std::pair<std::vector<std::pair<double, int>>, std::pair<int, int>> &) const;

    bool isCase2(const LocalEnvironment &) const;

    void setThisOrder(const std::vector<int> &);

    std::vector<std::tuple<int, int, double>> getDistIndices() const;

    std::vector<std::tuple<int, int, double>> getDistIndices(const std::vector<int> &) const;

    /// Get the distances if the sites would have been rearranged according to indiceOrder.
    std::vector<double> getReorderedDistances(const std::vector<int> &) const;

    /// Return the sites if the order would have been as is given in indiceOrder
    std::vector<int> getReorderedSites(const std::vector<int> &) const;

    std::tuple<std::vector<double>, std::vector<int>, std::vector<int>> case2_min_indices(const LocalEnvironment &) const;

    void findMinimumIndexPermutation(size_t,
				     const std::vector<std::vector<int>> &,
				     const std::vector<int> &,
				     std::vector<int>,
				     std::vector<double> &,
				     std::vector<int> &,
				     std::vector<int> &) const;

    /// Compare distances and sites if dist1 and sites1 < dists2 and sites2
    bool compare_sites_dists(const std::vector<double> &,
                             const std::vector<int> &,
                             const std::vector<double> &,
                             const std::vector<int> &) const;

public:

    /// Comparison operator for equality.
    friend bool operator==(const Cluster &c1, const Cluster &c2)
    {
        if (c1.isSorted() != c2.isSorted())
        {
            throw std::runtime_error("Undefined behavior: comparing sorted with unsorted cluster (==)");
        }

        // Unsorted clusters uses clustertag for comparison
        if (!c1.isSorted())
        {
            return c1.tag() == c2.tag();
        }

        // Sorted clusters.
        /// @todo Establish whether the following test is sufficient.
        if (c1.sites() != c2.sites())
        {
            return false;
        }
        for (size_t i =0; i < c1.distances().size(); i++)
        {
            if(fabs(c1.distances()[i] - c2.distances()[i])>1e-5)
            {
                return false;
            }
        }
        return true;
    }

    /// Comparison operator for inequality.
    friend bool operator!=(const Cluster &c1, const Cluster &c2) { return !(c1 == c2); }

    /// Comparison operator for "smaller than" in the case of sortable clusters.
    friend bool operator<(const Cluster &c1, const Cluster &c2)
    {
        if (c1.isSorted() != c2.isSorted())
        {
            throw std::runtime_error("Undefined behavior: comparing sorted with unsorted cluster (<)");
        }

        if (!c1.isSorted())
        {
            return c1.tag() < c2.tag();
        }

        // 1) compare number of bodies in cluster
        if (c1.order() < c2.order())
        {
            return true;
        }
        if (c1.order() > c2.order())
        {
            return false;
        }

        // 2) compare distances
        for (size_t i = 0; i < c1.order(); i++)
        {
            if (c1.distances()[i] < c2.distances()[i])
            {
                return true;
            }
            if (c2.distances()[i] > c2.distances()[i])
            {
                return false;
            }
        }

        // 3) compare sites
        for (size_t i = 0; i < c1.order(); i++)
        {
            if (c1.sites()[i] < c2.sites()[i])
            {
                return true;
            }
            if (c1.sites()[i] > c2.sites()[i])
            {
                return false;
            }
        }

        // 4) if we are here then everything is equal so return false
        return false;
    }

private:

    /// List of lattice sites.
    std::vector<int> _sites;

    /// List of distances between points in cluster.
    std::vector<double> _distances;

    /// @todo what is this? The _element_counts member was deactivated since it is not in use.
    // std::map<std::vector<int>, int> _element_counts;

    /// Cluster radius.
    double _radius;

    /// True/False if this is a sorted/unsorted cluster.
    bool _sorted;

    /// Cluster tag.
    int _tag;

    /// Numerical precision imposed for rounding floats.
    /// @todo move to central place and harmonize handling of numerical precision.
    double _symprec;

    /// Round float number.
    /// @todo move to a more general location.
    double roundDouble(const double &double_value)
    {
        return round(double_value / _symprec) * _symprec;
    }
};

namespace std
{
/**
@brief Compute hash for a cluster
@details The hash is obtained by computing individual hash values for tag,
distances, and sites, and then combining them using XOR and bit shifting.
*/
template <>
struct hash<Cluster>
{
    size_t
    operator()(const Cluster &k) const
    {

        size_t seed = 0;

        // If unsorted just use the tag as seed
        if (!k.isSorted())
        {
            hash_combine(seed, k.tag());
            return seed;
        }

        for (const auto &distance : k.distances())
        {
            hash_combine(seed, hash_value(distance));
        }

        for (const auto &site : k.sites())
        {
            hash_combine(seed, hash_value(site));
        }

        return seed;
    }
};
}
