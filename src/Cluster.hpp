#pragma once

#include <boost/functional/hash.hpp>
#include <iostream>
#include "LatticeSite.hpp"
#include "Structure.hpp"

using boost::hash;
using boost::hash_combine;

/// This class handles information pertaining to a single cluster.
class Cluster
{
public:
    /// Empty constructor.
    Cluster() {}

    /// Creates cluster from a structure and a set of lattice sites.
    Cluster(const std::vector<LatticeSite> &,
            std::shared_ptr<const Structure>);

    /// Returns the lattice sites of this cluster.
    const std::vector<LatticeSite> &latticeSites() const { return _latticeSites; }

    /// Returns the order (i.e., the number of sites) of the cluster.
    unsigned int order() const { return _latticeSites.size(); }

    /// Returns the radius of the cluster.
    double radius() const;

    /// Returns the distances between the points in the cluster
    std::vector<float> distances() const;

    /// Returns the positions of the sites in this cluster in Cartesian coordinates.
    std::vector<Vector3d> getPositions() const;

    /// Comparison operator for automatic sorting.
    friend bool operator<(const Cluster &cluster1, const Cluster &cluster2)
    {
        return cluster1.latticeSites() < cluster2.latticeSites();
    }

    /// Translate the sites of the cluster by a constant vector.
    void translate(const Eigen::Vector3i &);

    /// Transform this cluster to a new structure
    void transformToSupercell(std::shared_ptr<const Structure>,
                              std::unordered_map<LatticeSite, LatticeSite> &,
                              const double);

    /// Check whether a site index is included with a zero offset.
    bool isSiteIndexIncludedWithZeroOffset(const size_t index) const;

    /// Returns the number of allowed components on each site in this cluster.
    std::vector<int> getNumberOfAllowedSpeciesPerSite() const;

    /// Count the number of occurences of a site index among the sites in this cluster
    unsigned int getCountOfOccurencesOfSiteIndex(const size_t) const;

    /// Stream operator
    friend std::ostream &operator<<(std::ostream &os, const Cluster &cluster);

private:
    /// The lattice sites in the cluster.
    std::vector<LatticeSite> _latticeSites;

    /**
    @brief The structure that the lattice sites of this cluster refers to. 
    @details
        We use a shared pointer to make this work with PyBind; if a Cluster
        is initilized from the Python side, the structure will get deleted
        (causing a segmentation fault) if a regular pointer rather than
        a share pointer is used. 
     */
    std::shared_ptr<const Structure> _structure;
};
