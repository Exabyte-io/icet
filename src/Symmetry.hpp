#pragma once

#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>

//#include "LatticeSite.hpp"
#include "Structure.hpp"

namespace icet
{

/// Find the permutation vector that takes v_permutation to v_original
template <typename T>
std::vector<int> getPermutation(const std::vector<T> &v_original, const std::vector<T> &v_permutation)
{
    if (v_original.size() != v_permutation.size())
    {
        throw std::runtime_error("Vectors are not of the same size (Symmetry/getPermutation)");
    }

    std::vector<int> indices(v_original.size());

    for (size_t i = 0; i < v_original.size(); i++)
    {
        auto find = std::find(v_permutation.begin(), v_permutation.end(), v_original[i]);
        if (find == v_permutation.end())
        {
            throw std::runtime_error("Permutation not possible since vectors do not contain the same elements (Symmetry/getPermutation)");
        }
        else
        {
            indices[i] = std::distance(v_permutation.begin(), find);
        }
    }

    return indices;
}

/// Returns the permutation of v using the permutation in indices.
template <typename T>
std::vector<T> getPermutedVector(const std::vector<T> &v,
                                 const std::vector<int> &indices)
{
    if (v.size() != indices.size())
    {
        throw std::runtime_error("Sizes of vectors do not match (Symmetry/getPermutedVector)");
    }

    std::vector<T> v2(v.size());
    for (size_t i = 0; i < v.size(); i++)
    {
        v2[i] = v[indices[i]];
    }
    return v2;
}

/// Returns the permutation of v using the permutation in indices.
template <typename T>
std::vector<std::vector<T>> getAllPermutations(std::vector<T> v)
{
    std::vector<std::vector<T>> allPermutations;
    std::sort(v.begin(), v.end());

    do
    {
        allPermutations.push_back(v);

    } while (std::next_permutation(v.begin(), v.end()));

    return allPermutations;
}

/// Returns the next cartesian product.
bool nextCartesianProduct(const std::vector<std::vector<int>> &items, std::vector<int> &currentProduct);

} // namespace icet
