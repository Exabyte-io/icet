#pragma once

#include "Structure.hpp"
#include "OrbitList.hpp"
#include "LocalOrbitListGenerator.hpp"
#include "ClusterCounts.hpp"
#include "PeriodicTable.hpp"

//namespace icet {

/**
@brief This class handles the cluster space.
@details It provides functionality for setting up a cluster space, calculating cluster vectors as well as retrieving various types of associated information.
@todo Add mathematical definitions.
*/

class ClusterSpace
{
  public:

    /// Constructor.
    ClusterSpace(std::vector<int>, std::vector<std::string>, const OrbitList);

    /// Returns the cluster vector corresponding to the input structure.
    std::vector<double> getClusterVector(const Structure &) const;

    /// Returns information concerning the cluster space.
    std::pair<int, std::vector<int>> getClusterSpaceInfo(const unsigned int);

    /// Returns the entire orbit list.
    OrbitList getOrbitList() const { return _orbitList; }

    /// Returns an orbit from the orbit list.
    Orbit getOrbit(const size_t index) const { return _orbitList.getOrbit(index); }

    /// Returns the native clusters.
    /// @todo What is a native cluster? Partial answer: clusters within the unit cell?
    ClusterCounts getNativeClusters(const Structure &structure) const;

    /// Returns the multi-component (MC) vector permutations for each MC vector in the set of MC vectors.
    /// @todo Clean up this description.
    std::vector<std::vector<std::vector<int>>> getMultiComponentVectorPermutations(const std::vector<std::vector<int>> &, const int ) const;


  public:

    /// Returns the cutoff for each order.
    std::vector<double> getCutoffs() const { return _clusterCutoffs; }

    /// Returns the primitive structure.
    Structure getPrimitiveStructure() const { return _primitiveStructure; }

    /// Returns the number of allowed components for each site.
    std::vector<int> getNumberOfAllowedSpeciesForEachSite(const Structure &, const std::vector<LatticeSite> &) const;

    /// Returns a list of elements associated with cluster space as chemical symbols.
    std::vector<std::string> getChemicalSymbols() const
    {
        std::vector<std::string> elements;
        for (const auto &elem : _elements)
            elements.push_back(PeriodicTable::intStr[elem]);
        return elements;
    }

    /// Returns the cluster space size, i.e. the length of a cluster vector.
    size_t getClusterSpaceSize()
    {
        if (!_isClusterSpaceInitialized)
            collectClusterSpaceInfo();
        return _clusterSpaceInfo.size();
    }

    /// Returns the mapping between atomic numbers and the internal species enumeration scheme.
    std::map<int, int> getElementMap() const { return _elementMap; }


  private:

    /// Returns the cluster product.
    /// @todo This function computes a specific term in the cluster vector. Can we find a more telling name?
    double getClusterProduct(const std::vector<int> &mcVector, const std::vector<int> &numberOfAllowedSpecies, const std::vector<int> &elements) const;

    /// Collect information about the cluster space.
    void collectClusterSpaceInfo();

    /// Returns the default cluster function.
    double defaultClusterFunction(const int numberOfAllowedSpecies, const int clusterFunction, const int element) const;


  private:

    /// True if cluster space has been initialized.
    bool _isClusterSpaceInitialized = false;

    /// Cluster space information.
    /// The first index (int) corresponds to the orbit index, the second index (vector of ints) refers to a multi-component vector.
    /// @todo Check description.
    /// @todo This function returns a very specific type of information. Consider giving it a more descriptive name.
    std::vector<std::pair<int, std::vector<int>>> _clusterSpaceInfo;

    /// Number of allowed components on each site of the primitive structure.
    std::vector<int> _numberOfAllowedSpecies;

    /// Primitive (prototype) structure.
    Structure _primitiveStructure;

    /// List of orbits based on primitive structure.
    OrbitList _orbitList;

    /// Radial cutoffs by cluster order starting with pairs.
    std::vector<double> _clusterCutoffs;

    /// Elements identified by atomic number considered in this cluster space.
    std::vector<int> _elements;

    /// Map between atomic numbers and the internal species enumeration scheme.
    std::map<int, int> _elementMap;

};

//}