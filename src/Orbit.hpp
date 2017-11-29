#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <vector>
#include <string>
#include "LatticeSite.hpp"
#include "Cluster.hpp"
#include "hash_functions.hpp"
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

    ///Add a group sites that are equivalent to the ones in this orbit
    void addEquivalentSites(const std::vector<LatticeSite> &latNbrs, bool sort = false)
    {
        _equivalentSites.push_back(latNbrs);
        if (sort)
        {
            sortOrbit();
        }
    }
    ///add many lattice neigbhors
    void addEquivalentSites(const std::vector<std::vector<LatticeSite>> &LatticeSites, bool sort = false)
    {
        _equivalentSites.insert(_equivalentSites.end(), LatticeSites.begin(), LatticeSites.end());
        if (sort)
        {
            sortOrbit();
        }
    }

    ///Returns amount of equivalent sites in this orbit
    size_t size() const
    {
        return _equivalentSites.size();
    }
    ///Returns the geometric size of the orbit defines as the mean distance to the center of the
    double geometricalSize() const
    {
        return _representativeCluster.geometricalSize();
    }
    ///Return the sorted, reprasentative cluster for this orbit
    Cluster getRepresentativeCluster() const
    {
        return _representativeCluster;
    }

    ///Returns equivalent sites
    std::vector<std::vector<LatticeSite>> getEquivalentSites() const
    {
        return _equivalentSites;
    }

    ///Returns equivalent sites
    std::vector<std::vector<LatticeSite>> getPermutatedEquivalentSites() const
    {
        std::vector<std::vector<LatticeSite>> permutatedSites(_equivalentSites.size());
        for(size_t i =0; i < _equivalentSites.size(); i++)
        {
            permutatedSites[i] = getSitesWithPermutation(i);
        }
        return permutatedSites;
    }



    ///Return the equivalent sites at position `index`
    std::vector<LatticeSite> GetSitesOfIndex(unsigned int index) const
    {
        if (index >= _equivalentSites.size())
        {
            throw std::out_of_range("Index out of range in function Orbit::GetSitesOfIndex");
        }
        return _equivalentSites[index];
    }

    ///Return the equivalent sites at position `index` by using the permutation to rep. cluster
    std::vector<LatticeSite> getSitesWithPermutation(unsigned int index) const
    {
        if (index >= _equivalentSites.size())
        {
            throw std::out_of_range("Index out of range for _equivalentSites in function Orbit::getSitesWithPermutation");
        }
        if (index >= _equivalentSitesPermutations.size())
        {
            std::cout<<" size of orbit that failed "<<_equivalentSites[0].size()<<std::endl;
            std::string errMSG = "Index out of range for _equivalentSitesPermutations in function Orbit::getSitesWithPermutation index: " + std::to_string(index) + " >= "  + std::to_string( _equivalentSitesPermutations.size());
            throw std::out_of_range(errMSG);
        }

        return icet::getPermutatedVector<LatticeSite>( _equivalentSites[index],_equivalentSitesPermutations[index]);
    }

        ///This sets the equivalent sites
        void setEquivalentSites(const std::vector<std::vector<LatticeSite>> &equivalentSites)
        {
            _equivalentSites = equivalentSites;
        }

        void sortOrbit()
        {
            std::sort(_equivalentSites.begin(), _equivalentSites.end());
        }
        ///Return the number of bodies of the cluster that represent this orbit
        unsigned int getClusterSize() const
        {
            return _representativeCluster.order();
        }
        ///Compare operator for automatic sorting in containers
        friend bool operator==(const Orbit &orbit1, const Orbit &orbit2)
        {

            /// First test against number of bodies in cluster
            if (orbit1.getRepresentativeCluster().order() != orbit2.getRepresentativeCluster().order())
            {
                return false;
            }
            ///not equal size: compare by geometrical size
            if (fabs(orbit1.geometricalSize() - orbit2.geometricalSize()) > 1e-5) // @TODO: remove 1e-4 and add tunable parameter
            {
                return false;
            }

            // check size of vector of equivalent sites
            if (orbit1.size() != orbit2.size())
            {
                return false;
            }

            //Now size of  equivalent sites vector are the same, then check the individual equivalent sites
            return orbit1.getEquivalentSites() == orbit2.getEquivalentSites();
        }

        ///Compare operator for automatic sorting in containers
        friend bool operator<(const Orbit &orbit1, const Orbit &orbit2)
        {

            /// First test against number of bodies in cluster
            if (orbit1.getRepresentativeCluster().order() != orbit2.getRepresentativeCluster().order())
            {
                return orbit1.getRepresentativeCluster().order() < orbit2.getRepresentativeCluster().order();
            }
            ///not equal size: compare by geometrical size
            if (fabs(orbit1.geometricalSize() - orbit2.geometricalSize()) > 1e-5) // @TODO: remove 1e-4 and add tunable parameter
            {
                return orbit1.geometricalSize() < orbit2.geometricalSize();
            }

            // check size of vector of equivalent sites
            if (orbit1.size() < orbit2.size())
            {
                return true;
            }
            if (orbit1.size() > orbit2.size())
            {
                return false;
            }
            //Now size of  equivalent sites vector are the same, then check the individual equivalent sites
            return orbit1.getEquivalentSites() < orbit2.getEquivalentSites();
        }
        ///Return the equivalent sites permutations
        std::vector<std::vector<int>> getEquivalentSitesPermutations() const
        {
            return _equivalentSitesPermutations;
        }

        /// Assigns the equivalent sites permutations
        void setEquivalentSitesPermutations(std::vector<std::vector<int>> & permutations)
        {
            _equivalentSitesPermutations = permutations;
        }

        /// Assigns the allowed sites permutations
        void setAllowedSitesPermutations(std::unordered_set<std::vector<int>, VectorHash> & permutations)
        {
            _allowedSitesPermutations = permutations;
        }

        ///Return the representative sites of this orbit (if any equivalentSites permutations exists it is to these sites they refer to)
        std::vector<LatticeSite> getRepresentativeSites() const
        {
            return _equivalentSites[0];
        }

        /**
    Returns the number of exactly equal sites in equivalent sites vector
    This is used among other things to debug orbits when duplicates is not expected
    */
        int getNumberOfDuplicates(int verbosity = 0) const;

        /**
        Creates a copy of this orbit and translates all LatticeSite offsets in equivalent sites
        this will also transfer any existing permutations directly which should be fine since an offset doesn't change permutations to the prototype sites)
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
        /// append an orbit to this orbit.
        Orbit &operator+=(const Orbit &orbit_rhs)
        {
            // this orbit doesn't have any eq. sites permutations: check that orbit_rhs also doesn't have them
            if (_equivalentSitesPermutations.size() == 0)
            {
                if (orbit_rhs.getEquivalentSitesPermutations().size() != 0)
                {
                    throw std::runtime_error("Error: one orbit has eq. site permutations and one doesn't in function: Orbit &operator+= ");
                }
            }
            else //this orbit has some eq. sites permutations: check that orbit_rhs also has them
            {
                if (orbit_rhs.getEquivalentSitesPermutations().size() == 0)
                {
                    throw std::runtime_error("Error: one orbit has eq. site permutations and one doesn't in function: Orbit &operator+= ");
                }
            }

            //Assert that both orbits are referencing the same prototype sites with the difference that the offsets are offset by a constant

            //Get representative sites
            auto rep_sites_rhs = orbit_rhs.getRepresentativeSites();
            auto rep_sites_this = getRepresentativeSites();

            if (rep_sites_this.size() != rep_sites_rhs.size())
            {
                throw std::runtime_error("Error: Orbit order is not equal in function: Orbit &operator+= ");
            }
            //The offsets between the offsets of the two rep. eq. sites
            Vector3d offsetOfOffsets;

            for (size_t i = 0; i < rep_sites_this.size(); i++)
            {
                // 
                // if (rep_sites_this[i].index() != rep_sites_rhs[i].index())
                // {
                //     std::cout<< rep_sites_this[i].index()<< " "<< rep_sites_rhs[i].index()<<std::endl;
                //     throw std::runtime_error("Error: this orbit and orbit_rhs do not have the same reference cluster in function: Orbit &operator+=");
                // }
                if (i == 0)
                {
                    offsetOfOffsets = rep_sites_this[i].unitcellOffset() - rep_sites_rhs[i].unitcellOffset();
                }
                else //check that the offsets between sites at position `i` is the same as `i-1`
                {
                    Vector3d newOffset = rep_sites_this[i].unitcellOffset() - rep_sites_rhs[i].unitcellOffset();
                    if ((newOffset - offsetOfOffsets).norm() > 0.1)
                    {
                        throw std::runtime_error("Error: this orbit and orbit_rhs do not have the same offsets between sites in function : Orbit &operator+=");
                    }
                    else
                    {
                        offsetOfOffsets = newOffset;
                    }
                }
            }

            //All tests passed, can now add equivalent sites and equivalent sites permutations

            //_equivalentSites.insert(_equivalentSites.end(), LatticeSites.begin(), LatticeSites.end());

            const auto rhsEquivalentSites = orbit_rhs.getEquivalentSites();
            const auto rhsEquivalentSitesPermutations = orbit_rhs.getEquivalentSitesPermutations();
            //Insert rhs eq sites and corresponding permutations
            _equivalentSites.insert(_equivalentSites.end(), rhsEquivalentSites.begin(), rhsEquivalentSites.end());
            _equivalentSitesPermutations.insert(_equivalentSitesPermutations.end(), rhsEquivalentSitesPermutations.begin(), rhsEquivalentSitesPermutations.end());
            return *this;
        }

        ///Mi_local are the same size as representative sites and details the allowed occupations on the representative sites
        std::vector<std::vector<int>> getMCVectors(const std::vector<int> & Mi_local) const;

        std::vector<std::vector<int>> getAllPossibleMCVectorPermutations(const std::vector<int> &Mi_local) const;

      private:
        ///Representative sorted cluster for this orbit
        Cluster _representativeCluster;

        ///Container of equivalent sites for this orbit
        std::vector<std::vector<LatticeSite>> _equivalentSites;

        ///Contains the permutations of the equivalent sites which takes it to the order of the reference cluster
        std::vector<std::vector<int>> _equivalentSitesPermutations;

        /// Contains the allowed sites permutations. i.e. if 0,2,1 is in this set then 0,1,0 is the same MC vector as 0,0,1
        std::unordered_set<std::vector<int>, VectorHash> _allowedSitesPermutations;
    };
