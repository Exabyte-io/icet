#pragma once
#include <vector>
#include <boost/functional/hash.hpp>
using boost::hash;
using boost::hash_combine;
using boost::hash_value;

/// Hash function for a vector of ints.
struct VectorHash {
    /// Hash operator.
    size_t operator()(const std::vector<int>& v) const
    {
        size_t seed = 0;
        for (const int &i : v) {
            hash_combine(seed, hash_value(i));
        }
        return seed;
    }
};

/// Hash function for a three-dimensional vector.
struct Vector3dHash {
    /// Hash operator.
    size_t operator()(const Vector3d& v) const
    {
        size_t seed = 0;
        for (size_t i =0; i<3; i++) {
            hash_combine(seed, hash_value(v[i]));
        }
        return seed;
    }
};

/// Comparison operation for two three-dimensional vectors.
struct Vector3dCompare
{
    /// Comparison operator.
    bool operator()(const Vector3d &lhs, const Vector3d &rhs) const
    {
        for (size_t i = 0; i < 3; i++)
        {
            if (lhs[i] == rhs[i])
            {
                continue;
            }
            else
            {
                return lhs[i] < rhs[i];
            }
        }
        return false;
    }
};
