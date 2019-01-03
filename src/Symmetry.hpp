#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include "LatticeSite.hpp"
#include "Structure.hpp"
namespace icet
{

/// Find the permutation vector that takes v_permutation to v_original
template <typename T>
std::vector<int> getPermutation(const std::vector<T> &v_original, const std::vector<T> &v_permutation)
{
    if (v_original.size() != v_permutation.size())
    {
        throw std::runtime_error("Error: vectors are not of the same size in function getPermutation in Symmetry.hpp");
    }

    std::vector<int> indices(v_original.size());

    for (size_t i = 0; i < v_original.size(); i++)
    {
        auto find = std::find(v_permutation.begin(), v_permutation.end(), v_original[i]);
        if (find == v_permutation.end())
        {
            throw std::runtime_error("Error: Permutation not possible since vectors do not contain same elements in function getPermutation in Symmetry.hpp ");
        }
        else
        {
            indices[i] = std::distance(v_permutation.begin(), find);
        }
    }

    return indices;
}

///Return the permutation of v using the permutation in indices
template <typename T>
std::vector<T>
getPermutedVector(
    const std::vector<T> &v,
    const std::vector<int> &indices)
{

    if (v.size() != indices.size())
    {
        throw std::runtime_error("Error: vectors are not of the same size in function getPermutedVector in Symmetry.hpp");
    }

    std::vector<T> v2(v.size());
    for (size_t i = 0; i < v.size(); i++)
    {
        v2[i] = v[indices[i]];
    }
    return v2;
}

///Return the permutation of v using the permutation in indices
template <typename T>
std::vector<std::vector<T>>
getAllPermutations(
    std::vector<T> v)
{
    std::vector<std::vector<T>> allPermutations;
    std::sort(v.begin(), v.end());

    do
    {
        allPermutations.push_back(v);

    } while (std::next_permutation(v.begin(), v.end()));

    return allPermutations;
}

/// Return the transformed position `position` using the input translation and rotation
/// @TODO whys is there a '2' here?
Eigen::Vector3d transformPosition(const Eigen::Vector3d &position, const Eigen::Vector3d &translation, const Eigen::Matrix3d &rotation);

///Returns the next cartesian product of currentProduct using the vector of vectors items items[0] is the possible combinations for element n
bool nextCartesianProduct(const std::vector<std::vector<int>> &items, std::vector<int> &currentProduct);

std::vector<Eigen::Vector3d> getOffsetPositions(const Structure&, const Eigen::Vector3d&);

}
