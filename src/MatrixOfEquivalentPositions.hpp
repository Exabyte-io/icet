#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

/**
@details This class handles a matrix of equivalent positions. Each row
corresponds to a set of symmetry equivalent positions. The entry in the
first column is commonly treated as the representative position.
**/
class MatrixOfEquivalentPositions
{
public:

    /**
    @details This class stores a matrix of (=symmetry equivalent) positions.
    @param translations translational symmetry operations
    @param rotations rotational symmetry operations
    **/
    MatrixOfEquivalentPositions(const std::vector<Eigen::Vector3d> &translations,
                                const std::vector<Eigen::Matrix3d> &rotations)
    {
        _translations = translations;
        _rotations = rotations;
    }

    /// Builds matrix of symmetry equivalent positions.
    void build(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> &);

    /// Returns the matrix of symmetry equivalent positions.
    std::vector<std::vector<Eigen::Vector3d>> getPermutedPositions() const { return _matrixOfEquivalentPositions; }

private:

    /// Translational symmetry operations
    std::vector<Eigen::Vector3d> _translations;

    /// Rotational symmetry operations
    std::vector<Eigen::Matrix3d> _rotations;

    /// Matrix of symmetry equivalent positions
    std::vector<std::vector<Eigen::Vector3d>> _matrixOfEquivalentPositions;
};
