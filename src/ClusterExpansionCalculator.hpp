#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <string>
#include <math.h>
#include "Structure.hpp"
#include "ClusterSpace.hpp"
#include "OrbitList.hpp"
#include "LocalOrbitListGenerator.hpp"
#include "ClusterCounts.hpp"
#include "PeriodicTable.hpp"
#include "Vector3dCompare.hpp"
using namespace Eigen;


class ClusterExpansionCalculator
{
    public:
    
    /// Constructor.
    ClusterExpansionCalculator(const ClusterSpace &, const Structure &);

    /// Returns the local cluster vector.
    std::vector<double> getLocalClusterVector(const std::vector<int>&, const int, const std::vector<size_t>);

    private:
  
    /// Maps offsets to local orbit lists.
    std::unordered_map<Vector3d, OrbitList, Vector3dHash> _localOrbitlists;

    /// Internal cluster space.
    ClusterSpace _clusterSpace;
    
    /// The supercell the calculator is created for.
    Structure _superCell;
    
    /// The full primitive orbit list, contains all clusters for the primitive cell.
    OrbitList _fullPrimitiveOrbitList;

    /// Maps a lattice site from the primitive and get the equivalent in the supercell.
    std::unordered_map<LatticeSite, LatticeSite> _primToSupercellMap;

    /// Maps supercell index to its corresponding primitive cell offset.
    std::map<int, Vector3d> _indexToOffset;
};
