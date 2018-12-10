#pragma once
#include <Eigen/Dense>
#include <boost/functional/hash.hpp>
#include <iostream>
using boost::hash;
using boost::hash_combine;
using boost::hash_value;

/**
@brief Struct for storing information concerning a lattice site.
@details This structure provides functionality for handling lattice sites
most notably including comparison operators.
*/
struct LatticeSite
{

public:
    /// Empty constructor.
    LatticeSite() { }

    /**
    @brief Constructor.
    @param index site index
    @param unitcellOffset offset of site relative to unit cell at origin in units of lattice vectors
    */
    LatticeSite(const size_t index, const Eigen::Vector3d unitcellOffset)
    {
        _index = index;
        _unitcellOffset = unitcellOffset;
    }

    /// Return index of site.
    size_t index() const { return _index; }

    /// Set index of site.
    void setIndex(size_t index) { _index = index; }

    /// Return offset relative to unit cell at origin in units of lattice vectors.
    Eigen::Vector3d unitcellOffset() const { return _unitcellOffset; }

    /// Set offset relative to unit cell at origin in units of lattice vectors.
    void setUnitcellOffset(Eigen::Vector3d offset) { _unitcellOffset = offset; }

    /// Add offset relative to unit cell at origin in units of lattice vectors.
    void addUnitcellOffset(Eigen::Vector3d offset) { _unitcellOffset += offset; }

    /// Smaller than operator.
    bool operator<(const LatticeSite &other) const
    {
        if (_index == other.index())
        {
            for (size_t i = 0; i < 3; i++)
            {
                if (_unitcellOffset[i] != other.unitcellOffset()[i])
                {
                    return _unitcellOffset[i] < other.unitcellOffset()[i];
                }
            }
        }
        return _index < other.index();
    }

    /// Equality operator.
    bool operator==(const LatticeSite &other) const
    {
        if (_index != other.index())
        {
            return false;
        }

        for (size_t i = 0; i < 3; i++)
        {
            if (_unitcellOffset[i] != other.unitcellOffset()[i])
            {
                return false;
            }
        }
        return true;
    }

    /// Addition operator.
    friend LatticeSite operator+(const LatticeSite &latticeSite, const Eigen::Vector3d &offset)
    {
        LatticeSite latnbr = LatticeSite(latticeSite.index(), latticeSite.unitcellOffset() + offset);
        return latnbr;
    }

    /// Substraction operator.
    friend LatticeSite operator-(const LatticeSite &latticeSite, const Eigen::Vector3d &offset)
    {
        LatticeSite latnbr = LatticeSite(latticeSite.index(), latticeSite.unitcellOffset() - offset);
        return latnbr;
    }

    /// Addition and assignment operator.
    friend LatticeSite operator+=(const LatticeSite &latticeSite, const Eigen::Vector3d &offset)
    {
        LatticeSite latnbr = LatticeSite(latticeSite.index(), latticeSite.unitcellOffset() + offset);
        return latnbr;
    }

    /// Cast object as string.
    operator std::string () const
    {
        std::string str = std::to_string(_index) + " :";
        for (size_t i = 0; i < 3; i++)
        {
            str += " " + std::to_string(_unitcellOffset[i]);
        }
        return str;
    }

    /// Write class information to stdout.
    void print() const
    {
        std::string str = std::to_string(_index) + " :";
        for (size_t i = 0; i < 3; i++)
        {
            str += " " + std::to_string(_unitcellOffset[i]);
        }
        std::cout << str << std::endl;
    }

private:

    /// Site index.
    size_t _index;

    /// Offset relative to the unit cell at the origin in units of lattice vectors.
    Eigen::Vector3d _unitcellOffset;

};

namespace std
{

    /// Compute hash for an individual lattice site.
    template <>
    struct hash<LatticeSite>
    {
        size_t
        operator()(const LatticeSite &k) const
        {
            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:
            size_t seed = 0;
            hash_combine(seed, hash_value(k.index()));

            for (size_t i = 0; i < 3; i++)
            {
                hash_combine(seed, hash_value(k.unitcellOffset()[i]));
            }
            return seed;
        }
    };

    /// Compute hash for a list of lattice sites.
    template <>
    struct hash<std::vector<LatticeSite>>
    {
        size_t
        operator()(const std::vector<LatticeSite> &k) const
        {
            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:
            size_t seed = 0;
            for (const auto &latNbr : k)
            {
                hash_combine(seed, std::hash<LatticeSite>{}(latNbr));
            }
            return seed;
        }
    };
}
