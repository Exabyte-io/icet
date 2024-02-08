#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "Structure.hpp"
#include "ClusterSpace.hpp"
#include "OrbitList.hpp"
#include "LocalOrbitListGenerator.hpp"
#include "VectorOperations.hpp"
using namespace Eigen;

/**
@brief This class provides a cluster expansion calculator.

@details
    A cluster expansion calculator is specific for a certain supercell. Upon
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

    /// Returns change in cluster vector upon flipping occupation of one site
    std::vector<double> getClusterVectorChange(const py::array_t<int> &, const size_t, const size_t);

    /// Returns the full cluster vector.
    std::vector<double> getClusterVector(const py::array_t<int> &);

    /// Returns a local cluster vector; the contribution to the cluster vector from one site.
    std::vector<double> getLocalClusterVector(const py::array_t<int> &, int);

private:
    /// The full orbit list used when calculating full cluster vector
    OrbitList _fullOrbitList;

    /// Maps offsets to local orbit lists.
    std::unordered_map<Vector3i, OrbitList, Vector3iHash> _localOrbitlists;

    /// Internal cluster space.
    ClusterSpace _clusterSpace;

    /// The supercell the calculator is created for.
    std::shared_ptr<Structure> _supercell;

    /// Maps supercell index to its corresponding primitive cell offset.
    std::map<int, Vector3i> _indexToOffset;
};
