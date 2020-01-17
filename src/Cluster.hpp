#pragma once

#include <boost/functional/hash.hpp>
#include "FloatType.hpp"
#include "Geometry.hpp"
#include "LatticeSite.hpp"
#include "Structure.hpp"

using boost::hash;
using boost::hash_combine;

/// This class handles information pertaining to a single cluster.
class Cluster
{
public:

    // Empty constructor
    Cluster() { }

    /// Creates cluster from a structure and a set of lattice sites.
    Cluster(const Structure &structure,
            const std::vector<LatticeSite> &latticeSites,
            const int tag = 0);

public:

    /// Returns the order (i.e., the number of sites) of the cluster.
    unsigned int order() const { return _sites.size(); }

    /// Returns the radius of the cluster.
    double radius() const { return _radius; }

    /// Returns the sites in the cluster.
    std::vector<int> sites() const { return _sites; }

    /// Returns distances between points in the cluster.
    std::vector<double> distances() const { return _distances; }

    /// Returns the cluster tag used for identification.
    int tag() const { return _tag; }

    /// Set the cluster tag.
    void setTag(const int tag) { _tag = tag; }

public:

    /// Comparison operator for equality.
    friend bool operator==(const Cluster &lhs, const Cluster &rhs) { return lhs.tag() == rhs.tag(); }

    /// Comparison operator for less than.
    friend bool operator<(const Cluster &lhs, const Cluster &rhs) { return lhs.tag() < rhs.tag(); }

private:

    /// List of lattice sites.
    std::vector<int> _sites;

    /// List of distances between points in cluster.
    std::vector<double> _distances;

    /// Cluster radius.
    double _radius;

    /// Cluster tag.
    int _tag;

};

namespace std
{
    /**
    @brief Computes hash for a cluster.
    @details The hash is obtained by computing the hash value for the tag.
    */
    template <>
    struct hash<Cluster>
    {
        /// Hash operator.
        size_t operator()(const Cluster &cluster) const
        {
            size_t seed = 0;
            hash_combine(seed, cluster.tag());
            return seed;
        }
    };

    /// Stream operator.
    ostream& operator<<(ostream&, const Cluster&);
}
