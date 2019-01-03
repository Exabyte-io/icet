#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include <string>
#include <pybind11/stl.h>
#include <map>
#include <unordered_map>
#include <algorithm>
#include "Structure.hpp"
#include "LatticeSite.hpp"
#include "Geometry.hpp"

namespace py = pybind11;

/**
  @brief Helper structure which is internally used for sorting clusters.
  @details An instance of this structure stores
  * the index and site of the "first" (reference) point in a clusterand
  * with the distances, indices, and sites of its neighbors

  For example in the case of a four-point cluster the structure would contain
  @code
  (index, site)              --> reference point
  (distance1, site1, index1) --> neighbor 1
  (distance2, site2, index2) --> neighbor 2
  (distance3, site3, index3) --> neighbor 3
  @endcode

  The structure provides functions for comparing clusters in order to establish
  and ordering.
*/
struct LocalEnvironment
{

    /// Empty constructor.
    LocalEnvironment() { }

    /**
    @brief Constructor.
    @param index index of the reference point of the cluster
    @param site site of the reference point of the cluster
    @param neighborDistances list of distances to neigbors
    @param neighborSites list of sites of neighbors
    @param neighborIndices list of indices of neighbors
    **/
    LocalEnvironment(const int &index,
                const int &site,
                const std::vector<double> &neighborDistances,
                const std::vector<int> &neighborSites,
                const std::vector<int> &neighborIndices) :
                _index(index), _site(site),
                _neighborDistances(neighborDistances),
                _neighborSites(neighborSites),
                _neighborIndices(neighborIndices) { }

    /// Returns list of distances to neighbors.
    std::vector<double> getNeighborDistances() const { return _neighborDistances; }

    /// Set list of distances to neighbors.
    void setNeighborDistances(const std::vector<double> &neighborDistances) { _neighborDistances = neighborDistances; }

    /// Returns list of sites of neighbors.
    std::vector<int> getNeighborSites() const { return _neighborSites; }

    /// Set list of sites of neighbors.
    void setNeighborSites(const std::vector<int> &sites) { _neighborSites = sites; }

    /// Return list of indices of neighbors.
    std::vector<int> getNeighborIndices() const { return _neighborIndices; }

    /// Set list of _neighborIndices.
    void setNeighborIndices(const std::vector<int> &neighborIndices) { _neighborIndices = neighborIndices; }

    /// Returns indices of all points in the cluster including the reference point.
    std::vector<int> getClusterIndices() const
    {
        std::vector<int> fullIndices = {_index};
        fullIndices.reserve(_neighborIndices.size());
        for (const auto &index : _neighborIndices)
        {
            fullIndices.push_back(index);
        }
        return fullIndices;
    }

    /**
    @brief Identify which neighbor tuples (distance \f$d_i\f$, site \f$s_i\f$)
    combinations are identical.

    @details This function returns a vector of vectors of indices that identify
    the neighbor tuples \f$(d_i, s_i)\f$, for which
    \f$d_i=d_j, s_i=s_j\, \forall\, i,j\f$.

    For example assuming that \f$d_i=d_j, s_i=s_j\f$ for \f$i,j\in(1,3)\f$
    and \f$d_i=d_j, s_i=s_j\f$ for \f$i,j\in(2,4)\f$, the function will
    return \f$[1, 3], [2, 4]\f$.

    @returns vector of vectors
    */
    std::vector<std::vector<int>> getEquivalentIndices() const
    {

        if (_neighborDistances.size() != _neighborSites.size())
        {
            throw std::runtime_error("_neighborDistances and _neighborSites not equal in size in getEquivalentIndices");
        }
        if (_neighborIndices.size() != _neighborSites.size())
        {
            throw std::runtime_error("_neighborIndices and _neighborSites not equal in size in getEquivalentIndices");
        }

        // maps _neighborDistances, _neighborSites to vector of _neighborIndices
        std::map<std::pair<double, int>, std::vector<int>> equivalentIndicesMap;
        for (size_t i = 0; i < _neighborDistances.size(); i++)
        {
            equivalentIndicesMap[std::make_pair(_neighborDistances[i], _neighborSites[i])].push_back(i + 1);
        }

        std::vector<std::vector<int>> equivalentIndices;
        for (const auto &mapPair : equivalentIndicesMap)
        {
            if (mapPair.second.size() > 1)
            {
                equivalentIndices.push_back(mapPair.second);
            }
        }
        return equivalentIndices;
    }

    /// Smaller than comparison operator.
    friend bool operator<(const LocalEnvironment &locEnv1, const LocalEnvironment &locEnv2)
    {
        if (locEnv1._neighborDistances.size() < locEnv2._neighborDistances.size())
        {
            return true;
        }
        else if (locEnv1._neighborDistances.size() > locEnv2._neighborDistances.size())
        {
            return false;
        }

        for (size_t i = 0; i < locEnv1._neighborDistances.size(); i++)
        {

            if (locEnv1._neighborDistances[i] < locEnv2._neighborDistances[i])
            {
                return true;
            }
            if (locEnv1._neighborDistances[i] > locEnv2._neighborDistances[i])
            {
                return false;
            }
        }

        if (locEnv1.site() < locEnv2.site())
        {
            return true;
        }
        if (locEnv1.site() > locEnv2.site())
        {
            return false;
        }

        for (size_t i = 0; i < locEnv1._neighborSites.size(); i++)
        {
            if (locEnv1._neighborSites[i] < locEnv2._neighborSites[i])
            {
                return true;
            }
            else if (locEnv1._neighborSites[i] > locEnv2._neighborSites[i])
            {
                return false;
            }
        }
        return false;
    }

    /// Cast object as string.
    operator std::string () const
    {
        std::string str;
        for (auto d : _neighborDistances)
        {
            str += std::to_string(d) + " ";
        }
        str += " : ";
        str += std::to_string(_site) + " ";
        for (auto d : _neighborSites)
        {
            str += std::to_string(d) + " ";
        }
        str += std::to_string(_index) + " ";
        for (auto d : _neighborIndices)
        {
            str += std::to_string(d) + " ";
        }
        return str;
    }

    /// Write class information to stdout.
    void print() const
    {
        std::string str;
        for (auto d : _neighborDistances)
        {
            str += std::to_string(d) + " ";
        }
        str += " : ";
        str += std::to_string(_site) + " ";
        for (auto d : _neighborSites)
        {
            str += std::to_string(d) + " ";
        }
        str += std::to_string(_index) + " ";
        for (auto d : _neighborIndices)
        {
            str += std::to_string(d) + " ";
        }
        std::cout << str << std::endl;
    }

    /// Equality comparison operator.
    friend bool operator==(const LocalEnvironment &locEnv1, const LocalEnvironment &locEnv2)
    {
        if (locEnv1._neighborDistances != locEnv2._neighborDistances)
        {
            return false;
        }
        else if (locEnv1._neighborSites != locEnv2._neighborSites)
        {
            return false;
        }
        return true;
    }

public:

    /// Returns index of reference point.
    int index() const { return _index; }

    /// Set index of reference point.
    void setIndex(int index) { _index = index; }

    /// Returns site of reference point.
    int site() const { return _site; }

    /// Set site of reference point.
    void setSite(int site) { _site = site; }

    /// Returns neighbor sites.
    std::vector<int> neighborSites() const { return _neighborSites; }

    /// Returns neighbor distances.
    std::vector<double> neighborDistances() const { return _neighborDistances; }

private:

    /// Index of point.
    int _index;

    /// Site of point.
    int _site;

    /// List of distances to neighbors.
    std::vector<double> _neighborDistances;

    /// List of sites of neighbors.
    std::vector<int> _neighborSites;

    /// List of indices of neighbors..
    std::vector<int> _neighborIndices;
};
