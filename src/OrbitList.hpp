#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include "Orbit.hpp"
#include "ManybodyNeighborlist.hpp"
#include "Structure.hpp"
#include "Cluster.hpp"
#include "Neighborlist.hpp"
#include <unordered_map>
#include <unordered_set>
#include "LatticeSite.hpp"
#include "hash_functions.hpp"
#include "Vector3dCompare.hpp"
#include "Symmetry.hpp"
#include "Geometry.hpp"
/**
Class OrbitList

contains a sorted vector or orbits


*/

class OrbitList
{
  public:
    OrbitList();
    OrbitList(const std::vector<Neighborlist> &neighborlists, const Structure &);
    OrbitList(const Structure &, const std::vector<std::vector<LatticeSite>> &, const std::vector<Neighborlist> &);

    /**
    The structure is a super cell
    the Vector3d is the offset you translate the orbit with
    the map maps primitive lattice neighbors to lattice neighbors in the supercell
    the const unsigned int is the index of the orbit

    strategy is to get the translated orbit and then map it using the map and that should be the partial supercell orbit of this site
    add together all sites and you get the full supercell porbot
    */
    Orbit getSuperCellOrbit(const Structure &, const Vector3d &, const unsigned int, std::unordered_map<LatticeSite, LatticeSite> &) const;

    OrbitList getLocalOrbitList(const Structure &, const Vector3d &, std::unordered_map<LatticeSite, LatticeSite> &) const;
    ///Add a group sites that are equivalent to the ones in this orbit
    void addOrbit(const Orbit &orbit)
    {
        _orbitList.push_back(orbit);
    }

    ///Returns number of orbits
    size_t size() const
    {
        return _orbitList.size();
    }

    /**
    Returns the number of orbits which are made up of N bodies
    */
    unsigned int getNumberOfNClusters(unsigned int N) const
    {
        unsigned int count = 0;
        for (const auto &orbit : _orbitList)
        {
            if (orbit.getRepresentativeCluster().order() == N)
            {
                count++;
            }
        }
        return count;
    }

    ///Return a copy of the orbit at position i in _orbitList
    Orbit getOrbit(unsigned int i) const
    {
        if (i >= size())
        {
            throw std::out_of_range("Error: Tried accessing orbit at out of bound index. Orbit OrbitList::getOrbit");
        }
        return _orbitList[i];
    }

    /// Clears the _orbitList
    void clear()
    {
        _orbitList.clear();
    }

    /// Sort the orbitlist
    void sort()
    {
        std::sort(_orbitList.begin(), _orbitList.end());
    }

    ///Returns orbitlist
    std::vector<Orbit> getOrbitList() const
    {
        return _orbitList;
    }

    int findOrbit(const Cluster &) const;

    /**
    Prints information about the orbitlist
    */

    void print(int verbosity = 0) const
    {
        int orbitCount = 0;
        for (const auto &orbit : _orbitList)
        {
            std::cout << "Orbit number: " << orbitCount++ << std::endl;
            std::cout << "Representative cluster " << std::endl;
            orbit.getRepresentativeCluster().print();

            std::cout << "Multiplicities: " << orbit.size() << std::endl;
            if (verbosity > 1)
            {
                std::cout << "Duplicates: " << orbit.getNumberOfDuplicates() << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void addClusterToOrbitlist(const Cluster &cluster, const std::vector<LatticeSite> &, std::unordered_map<Cluster, int> &);

    void addPermutationMatrixColumns(std::vector<std::vector<std::vector<LatticeSite>>> &lattice_neighbors, std::unordered_set<std::vector<int>, VectorHash> &taken_rows, const std::vector<LatticeSite> &lat_nbrs, const std::vector<int> &pm_rows,
                                     const std::vector<std::vector<LatticeSite>> &permutation_matrix, const std::vector<LatticeSite> &col1, bool) const;

    std::vector<LatticeSite> getColumn1FromPM(const std::vector<std::vector<LatticeSite>> &, bool sortIt = true) const;
    std::vector<int> findRowsFromCol1(const std::vector<LatticeSite> &col1, const std::vector<LatticeSite> &latNbrs, bool sortit = true) const;

    bool validatedCluster(const std::vector<LatticeSite> &) const;
    void addOrbitsFromPM(const Structure &, const std::vector<std::vector<std::vector<LatticeSite>>> &);
    void addOrbitFromPM(const Structure &, const std::vector<std::vector<LatticeSite>> &);
    void checkEquivalentClusters() const;

    std::vector<LatticeSite> translateSites(const std::vector<LatticeSite> &, const unsigned int) const;
    std::vector<std::vector<LatticeSite>> getSitesTranslatedToUnitcell(const std::vector<LatticeSite> &, bool sortit = true) const;
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> getMatchesInPM(const std::vector<std::vector<LatticeSite>> &, const std::vector<LatticeSite> &) const;

    void transformSiteToSupercell(LatticeSite &site, const Structure &superCell, std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap) const;
    void setPrimitiveStructure(const Structure &primitive)
    {
        _primitiveStructure = primitive;
    }

    ///Returns the primitive structure
    Structure getPrimitiveStructure() const
    {
        return _primitiveStructure;
    }
    /// += a orbitlist to another, first assert that they have the same number of orbits or that this is empty and then add equivalent sites of orbit i of rhs to orbit i to ->this
    OrbitList &operator+=(const OrbitList &rhs_ol)
    {
        if (size() == 0)
        {
            _orbitList = rhs_ol.getOrbitList();
            return *this;
        }

        if (size() != rhs_ol.size())
        {
            std::string errorMsg = "Error: lhs.size() and rhs.size() are not equal in  OrbitList& operator+= " + std::to_string(size()) + " != " + std::to_string(rhs_ol.size());
            throw std::runtime_error(errorMsg);
        }

        for (size_t i = 0; i < rhs_ol.size(); i++)
        {
            _orbitList[i] += rhs_ol.getOrbit(i); // .addEquivalentSites(.getEquivalentSites());
        }
        return *this;
    }

    // OrbitList getSupercellOrbitlist(const Structure &superCell) const;

    ///Adds the permutation information to the orbits
    void addPermutationInformationToOrbits(const std::vector<LatticeSite> &, const std::vector<std::vector<LatticeSite>> &);

    ///Returns all columns from the given rows in permutation matrix
    std::vector<std::vector<LatticeSite>> getAllColumnsFromRow(const std::vector<int> &, const std::vector<std::vector<LatticeSite>> &, bool, bool sortIt=true ) const;

    ///First construct rows_sort = sorted(rows)  then returns true/false if rows_sort exists in taken_rows
    bool isRowsTaken(const std::unordered_set<std::vector<int>, VectorHash> &taken_rows, std::vector<int> rows) const;

    ///Will find the sites in col1, extract and return all columns along with their unit cell translated indistinguishable sites
    std::vector<std::vector<LatticeSite>> getAllColumnsFromSites(const std::vector<LatticeSite> &,
        const std::vector<LatticeSite> &,
        const std::vector<std::vector<LatticeSite>> & ) const;

    ///Check that the lattice neighbors do not have any unitcell offsets in a pbc=false direction
    bool isSitesPBCCorrect(const std::vector<LatticeSite> &sites) const;

  private:
    int findOrbit(const Cluster &, const std::unordered_map<Cluster, int> &) const;
    Structure _primitiveStructure;
    std::vector<Orbit> _orbitList;
};
