#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <vector>
#include <string>
#include "LatticeSite.hpp"
#include "Cluster.hpp"
#include "VectorHash.hpp"
#include "Symmetry.hpp"
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
    Orbit(const Cluster &);

    /// Adds a group sites that are equivalent to the ones in this orbit.
    void addEquivalentSites(const std::vector<LatticeSite> &latNbrs, bool sort = false)
    {
        _equivalentSites.push_back(latNbrs);
        if (sort)
        {
            sortOrbit();
        }
    }
    /// Adds equilvalent sites.
    void addEquivalentSites(const std::vector<std::vector<LatticeSite>> &LatticeSites, bool sort = false)
    {
        _equivalentSites.insert(_equivalentSites.end(), LatticeSites.begin(), LatticeSites.end());
        if (sort)
        {
            sortOrbit();
        }
    }

    /// Returns number of equivalent sites in this orbit.
    size_t size() const
    {
        return _equivalentSites.size();
    }
    /// Returns the radius of the representative cluster in this orbit.
    double radius() const
    {
        return _representativeCluster.radius();
    }
    /// Returns the sorted, representative cluster for this orbit.
    Cluster getRepresentativeCluster() const
    {
        return _representativeCluster;
    }

    /// Returns equivalent sites.
    std::vector<std::vector<LatticeSite>> getEquivalentSites() const
    {
        return _equivalentSites;
    }

    /// Returns equivalent sites.
    std::vector<std::vector<LatticeSite>> getPermutedEquivalentSites() const
    {
        std::vector<std::vector<LatticeSite>> permutedSites(_equivalentSites.size());
        for (size_t i = 0; i < _equivalentSites.size(); i++)
        {
            permutedSites[i] = getSitesWithPermutation(i);
        }
        return permutedSites;
    }

    /// Returns the equivalent sites at position `index`.
    std::vector<LatticeSite> GetSitesOfIndex(unsigned int index) const
    {
        if (index >= _equivalentSites.size())
        {
            throw std::out_of_range("Index out of range in function Orbit::GetSitesOfIndex");
        }
        return _equivalentSites[index];
    }

    /// Returns the equivalent sites at position `index` by using the permutation to rep. cluster.
    std::vector<LatticeSite> getSitesWithPermutation(unsigned int index) const
    {
        if (index >= _equivalentSites.size())
        {
            throw std::out_of_range("Index out of range for _equivalentSites in function Orbit::getSitesWithPermutation");
        }
        if (index >= _equivalentSitesPermutations.size())
        {
            std::string errMSG = " size of orbit that failed " + std::to_string(_equivalentSites[0].size()) + "\n";
            errMSG += "Index out of range for _equivalentSitesPermutations in function Orbit::getSitesWithPermutation index: " + std::to_string(index) + " >= " + std::to_string(_equivalentSitesPermutations.size());
            throw std::out_of_range(errMSG);
        }

        return icet::getPermutedVector<LatticeSite>(_equivalentSites[index], _equivalentSitesPermutations[index]);
    }

    /// Sets the equivalent sites.
    void setEquivalentSites(const std::vector<std::vector<LatticeSite>> &equivalentSites)
    {
        _equivalentSites = equivalentSites;
    }

    void sortOrbit()
    {
        std::sort(_equivalentSites.begin(), _equivalentSites.end());
    }
    /// Returns the number of bodies of the cluster that represent this orbit.
    unsigned int getClusterSize() const
    {
        return _representativeCluster.order();
    }
    /// Comparison operator for automatic sorting in containers.
    friend bool operator==(const Orbit &orbit1, const Orbit &orbit2)
    {

        /// First test against number of bodies in cluster
        if (orbit1.getRepresentativeCluster().order() != orbit2.getRepresentativeCluster().order())
        {
            return false;
        }
        /// Not equal size: compare by radius
        if (fabs(orbit1.radius() - orbit2.radius()) > 1e-5) // @TODO: remove hard-coded tolerance and add tunable parameter
        {
            return false;
        }

        // Check size of vector of equivalent sites
        if (orbit1.size() != orbit2.size())
        {
            return false;
        }

        // Now size of  equivalent sites vector are the same, then check the individual equivalent sites
        return orbit1.getEquivalentSites() == orbit2.getEquivalentSites();
    }

    /// Comparison operator for automatic sorting in containers.
    friend bool operator<(const Orbit &orbit1, const Orbit &orbit2)
    {

        /// First test against number of bodies in cluster
        if (orbit1.getRepresentativeCluster().order() != orbit2.getRepresentativeCluster().order())
        {
            return orbit1.getRepresentativeCluster().order() < orbit2.getRepresentativeCluster().order();
        }
        /// Not equal size: compare by radius
        if (fabs(orbit1.radius() - orbit2.radius()) > 1e-5) // @TODO: remove 1e-4 and add tunable parameter
        {
            return orbit1.radius() < orbit2.radius();
        }

        // Check size of vector of equivalent sites
        if (orbit1.size() < orbit2.size())
        {
            return true;
        }
        if (orbit1.size() > orbit2.size())
        {
            return false;
        }
        // Now size of  equivalent sites vector are the same, then check the individual equivalent sites
        return orbit1.getEquivalentSites() < orbit2.getEquivalentSites();
    }
    /// Return the equivalent sites permutations.
    std::vector<std::vector<int>> getPermutationsOfEquivalentSites() const
    {
        return _equivalentSitesPermutations;
    }

    /// Assigns the equivalent sites permutations.
    void setEquivalentSitesPermutations(std::vector<std::vector<int>> &permutations)
    {
        _equivalentSitesPermutations = permutations;
    }

    /// Assigns the allowed sites permutations.
    void setAllowedSitesPermutations(std::unordered_set<std::vector<int>, VectorHash> &permutations)
    {
        _allowedSitesPermutations = permutations;
    }

    /// Gets the allowed sites permutations.
    std::unordered_set<std::vector<int>, VectorHash> getAllowedSitesPermutations() const
    {
        return _allowedSitesPermutations;
    }

    /// Returns the representative sites of this orbit (if any equivalentSites permutations exists it is to these sites they refer to).
    std::vector<LatticeSite> getRepresentativeSites() const
    {
        return _equivalentSites[0];
    }

    /**
    Returns the number of exactly equal sites in equivalent sites vector
    This is used among other things to debug orbits when duplicates are not expected
    */
    int getNumberOfDuplicates(int verbosity = 0) const;

    /**
    Creates a copy of this orbit and translates all LatticeSite offsets in equivalent sites.
    This will also transfer any existing permutations directly, which should be fine since an offset does not change permutations to the prototype sites.
    */
    friend Orbit operator+(const Orbit &orbit, const Eigen::Vector3d &offset)
    {
        Orbit orbitOffset = orbit;
        for (auto &latNbrs : orbitOffset._equivalentSites)
        {
            for (auto &latNbr : latNbrs)
            {
                latNbr = latNbr + offset;
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
                throw std::runtime_error("Error: one orbit has eq. site permutations and one doesn't in function: Orbit &operator+= ");
            }
        }
        else // This orbit has some eq. sites permutations: check that orbit_rhs also has them
        {
            if (orbit_rhs.getPermutationsOfEquivalentSites().size() == 0)
            {
                throw std::runtime_error("Error: one orbit has eq. site permutations and one doesn't in function: Orbit &operator+= ");
            }
        }

        // Assert that both orbits are referencing the same prototype sites with the difference that the offsets are offset by a constant

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

    /// Returns the relevant multicomponent vectors of this orbit given the number of allowed components.
    std::vector<std::vector<int>> getMultiComponentVectors(const std::vector<int> &Mi_local) const;

    std::vector<std::vector<int>> getAllPossibleMultiComponentVectorPermutations(const std::vector<int> &Mi_local) const;

    /// Returns true if the input sites exists in _equivalentSites, order doesn't matter if sorted= false.
    bool contains(const std::vector<LatticeSite> sites, bool sorted) const;

    /// Removes all elements i in equivalent sites if any sites in _equivalentSites[i] have the input index.
    void removeSitesWithIndex(const size_t indexRemove, bool);

    /// Removes all elements i in equivalent sites if no sites in _equivalentSites[i] have the input index.
    void removeSitesNotWithIndex(const size_t index, bool);

    /// Remove a specific set of sites with corresponding equivalent site permutation.
    void removeSites(std::vector<LatticeSite>);

    /// Container of equivalent sites for this orbit
    std::vector<std::vector<LatticeSite>> _equivalentSites;

    /// Representative sorted cluster for this orbit
    Cluster _representativeCluster;

  private:
    /// Contains the permutations of the equivalent sites which takes it to the order of the reference cluster
    std::vector<std::vector<int>> _equivalentSitesPermutations;

    /// Contains the allowed sites permutations. i.e. if 0,2,1 is in this set then 0,1,0 is the same MC vector as 0,0,1
    std::unordered_set<std::vector<int>, VectorHash> _allowedSitesPermutations;
};
