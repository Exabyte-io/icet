#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "Cluster.hpp"
#include "LatticeSite.hpp"
#include "Symmetry.hpp"
#include "VectorOperations.hpp"

using namespace Eigen;

/**
Class Orbit

contains equivalent vector<LatticeSites>
contains a sorted Cluster for representation

Can be compared to other orbits

*/

class Orbit
{
public:
    /// Constructor.
    Orbit(const Cluster &cluster) : _representativeCluster(cluster) {}

    /// Adds a group of sites that are equivalent to this orbit.
    void addEquivalentSites(const std::vector<LatticeSite> &, bool = false);

    /// Adds several groups of sites that are equivalent to this orbit.
    void addEquivalentSites(const std::vector<std::vector<LatticeSite>> &, bool = false);

    /// Returns the number of equivalent sites in this orbit.
    size_t size() const { return _equivalentSites.size(); }

    /// Returns the radius of the representative cluster in this orbit.
    double radius() const { return _representativeCluster.radius(); }

    /// Returns the sorted, representative cluster for this orbit.
    Cluster getRepresentativeCluster() const { return _representativeCluster; }

    /// Returns the equivalent sites.
    std::vector<LatticeSite> getSitesByIndex(unsigned int) const;

    /// Returns the permuted equivalent sites.
    std::vector<LatticeSite> getPermutedSitesByIndex(unsigned int) const;

    /// Returns all equivalent sites.
    std::vector<std::vector<LatticeSite>> getEquivalentSites() const { return _equivalentSites; }

    /// Returns all permuted equivalent sites.
    std::vector<std::vector<LatticeSite>> getPermutedEquivalentSites() const;

    /// Sets the equivalent sites.
    void setEquivalentSites(const std::vector<std::vector<LatticeSite>> &equivalentSites) { _equivalentSites = equivalentSites; }

    /// Sorts sites.
    void sort() { std::sort(_equivalentSites.begin(), _equivalentSites.end()); }

    /// Returns the number of bodies of the cluster that represent this orbit.
    unsigned int getClusterSize() const { return _representativeCluster.order(); }

    /// Returns the equivalent sites permutations.
    std::vector<std::vector<int>> getPermutationsOfEquivalentSites() const { return _equivalentSitesPermutations; }

    /// Assigns the equivalent sites permutations.
    void setEquivalentSitesPermutations(std::vector<std::vector<int>> &permutations) { _equivalentSitesPermutations = permutations; }

    /// Assigns the allowed sites permutations.
    void setAllowedSitesPermutations(std::set<std::vector<int>> &permutations) { _allowedSitesPermutations = permutations; }

    /// Gets the allowed sites permutations.
    std::set<std::vector<int>> getAllowedSitesPermutations() const { return _allowedSitesPermutations; }

    /// Returns the representative sites of this orbit (if any equivalentSites permutations exists it is to these sites they refer to).
    std::vector<LatticeSite> getRepresentativeSites() const { return _equivalentSites[0]; }

    /**
    Returns the number of exactly equal sites in equivalent sites vector
    This is used among other things to debug orbits when duplicates are not expected
    */
    int getNumberOfDuplicates(int verbosity = 0) const;

    /// Returns the relevant multicomponent vectors of this orbit given the number of allowed components.
    std::vector<std::vector<int>> getMultiComponentVectors(const std::vector<int> &Mi_local) const;

    std::vector<std::vector<int>> getAllPossibleMultiComponentVectorPermutations(const std::vector<int> &Mi_local) const;

    /// Returns true if the input sites exists in _equivalentSites, order does not matter if sorted=false.
    bool contains(const std::vector<LatticeSite>, bool) const;

    /// Removes all elements i in equivalent sites if any sites in _equivalentSites[i] have the input index.
    void removeSitesWithIndex(const size_t, bool);

    /// Removes all elements i in equivalent sites if no sites in _equivalentSites[i] have the input index.
    void removeSitesNotWithIndex(const size_t index, bool);

    /// Remove a specific set of sites with corresponding equivalent site permutation.
    void removeSites(std::vector<LatticeSite>);

    /// Comparison operator for automatic sorting in containers.
    friend bool operator==(const Orbit &orbit1, const Orbit &orbit2)
    {
        throw std::logic_error("Reached equal operator in Orbit");
    }

    /// Comparison operator for automatic sorting in containers.
    friend bool operator<(const Orbit &orbit1, const Orbit &orbit2)
    {
        throw std::logic_error("Reached < operator in Orbit");
    }

    /**
    Creates a copy of this orbit and translates all LatticeSite offsets in equivalent sites.
    This will also transfer any existing permutations directly, which should be fine since an offset does not change permutations to the prototype sites.
    */
    friend Orbit operator+(const Orbit &orbit, const Eigen::Vector3d &offset)
    {
        Orbit orbitOffset = orbit;
        for (auto &latticeSites : orbitOffset._equivalentSites)
        {
            for (auto &latticeSite : latticeSites)
            {
                latticeSite = latticeSite + offset;
            }
        }
        return orbitOffset;
    }

    /// Appends an orbit to this orbit.
    Orbit &operator+=(const Orbit &orbit_rhs)
    {
        // This orbit does not have any eq. sites permutations: check that orbit_rhs also doesn't have them
        if (_equivalentSitesPermutations.size() == 0)
        {
            if (orbit_rhs.getPermutationsOfEquivalentSites().size() != 0)
            {
                throw std::runtime_error("One orbit has equivalent site permutations and one does not (Orbit &operator+=)");
            }
        }
        else // This orbit has some eq. sites permutations: check that orbit_rhs also has them
        {
            if (orbit_rhs.getPermutationsOfEquivalentSites().size() == 0)
            {
                throw std::runtime_error("One orbit has equivalent site permutations and one does not (Orbit &operator+=)");
            }
        }

        // Get representative sites
        auto rep_sites_rhs = orbit_rhs.getRepresentativeSites();
        auto rep_sites_this = getRepresentativeSites();

        if (rep_sites_this.size() != rep_sites_rhs.size())
        {
            throw std::runtime_error("Error: Orbit order is not equal in function: Orbit &operator+= ");
        }
        // The offsets between the offsets of the two rep. eq. sites
        Vector3d offsetOfOffsets;

        const auto rhsEquivalentSites = orbit_rhs.getEquivalentSites();
        const auto rhsEquivalentSitesPermutations = orbit_rhs.getPermutationsOfEquivalentSites();

        // Insert rhs eq sites and corresponding permutations
        _equivalentSites.insert(_equivalentSites.end(), rhsEquivalentSites.begin(), rhsEquivalentSites.end());
        _equivalentSitesPermutations.insert(_equivalentSitesPermutations.end(), rhsEquivalentSitesPermutations.begin(), rhsEquivalentSitesPermutations.end());
        return *this;
    }

public:
    /// Container of equivalent sites for this orbit
    std::vector<std::vector<LatticeSite>> _equivalentSites;

    /// Representative sorted cluster for this orbit
    Cluster _representativeCluster;

private:
    /// Contains the permutations of the equivalent sites which takes it to the order of the reference cluster
    std::vector<std::vector<int>> _equivalentSitesPermutations;

    /// Contains the allowed sites permutations. i.e. if 0,2,1 is in this set then 0,1,0 is the same MC vector as 0,0,1
    std::set<std::vector<int>> _allowedSitesPermutations;
};


namespace std
{
    /// Stream operator.
    ostream& operator<<(ostream&, const Orbit&);
}
