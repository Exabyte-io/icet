#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <string>
#include "Structure.hpp"
#include "LatticeSite.hpp"

using namespace Eigen;

/**
@brief Class for maintaining a neighbor list.

@details This class provides functionality for building and storing a
(conventional) pair-wise neighbor list (as opposed to the specialized many-body
neighbor list defined by ManyBodyNeighborList).

@see ManyBodyNeighborList

@todo get rid of local DISTTOL definition
*/
class NeighborList
{
public:

    /**
    @brief Initialize a neighbor list instance.
    @param cutoff cutoff to be used for constructing the neighbor list.
    */
    NeighborList(const double cutoff) : _cutoff(cutoff) { };

    /// Build a neighbor list based on the given structure.
    void build(const Structure &);

    /// Update a neighbor list based on the given structure.
    void update(const Structure &);

    /// Check if two sites are neighbors.
    bool isNeighbor(const size_t, const size_t, const Vector3d) const;

    /// Returns list of neighbors.
    std::vector<LatticeSite> getNeighbors(size_t) const;

    /// Returns size of neighbor list.
    size_t size() const { return _neighbors.size(); }

    /// Returns cutoff radius used for building the neighbor list.
    double cutoff() const { return _cutoff; }

  private:
    /**
    @brief Neighbor list.

    @details Each entry in the outer vector corresponds to one position in the
    structure used to build the neighbor list; each inner vector specifies the
    neighbors of said position in the form of lattice sites.
    */
    std::vector<std::vector<LatticeSite>> _neighbors;

    /// Cutoff radius used for building the neighbor list.
    double _cutoff;

    /// Tolerance imposed for distance comparison.
    double DISTTOL = 1e-7;
};
