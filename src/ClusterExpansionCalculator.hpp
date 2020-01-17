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
#include "VectorOperations.hpp"
using namespace Eigen;

/**
@details This class provides a cluster expansion calculator. A cluster
    expansion calculator is specific for a certain supercell. Upon
    initialization various quantities specific to the given supercell are
    precomputed. This greatly speeds up subsequent calculations and enables one
    to carry out e.g., Monte Carlo simulations in a computationally efficient
    manner.
**/
class ClusterExpansionCalculator
{
public:

    /// Constructor.
    ClusterExpansionCalculator(const ClusterSpace &, const Structure &, const double);

    /// Returns the local cluster vector.
    std::vector<double> getLocalClusterVector(const std::vector<int>&, const int, const std::vector<size_t>);

private:

    /// Maps offsets to local orbit lists.
    std::unordered_map<Vector3d, OrbitList, Vector3dHash> _localOrbitlists;

    /// Internal cluster space.
    ClusterSpace _clusterSpace;

    /// The supercell the calculator is created for.
    Structure _supercell;

    /// The full primitive orbit list, contains all clusters for the primitive cell.
    OrbitList _fullPrimitiveOrbitList;

    /// Maps a lattice site from the primitive and get the equivalent in the supercell.
    std::unordered_map<LatticeSite, LatticeSite> _primToSupercellMap;

    /// Maps supercell index to its corresponding primitive cell offset.
    std::map<int, Vector3d> _indexToOffset;
};
