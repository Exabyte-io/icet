#include "ClusterExpansionCalculator.hpp"
#include <pybind11/stl.h>

ClusterExpansionCalculator::ClusterExpansionCalculator(const ClusterSpace &clusterSpace,
                                                       const Structure &structure,
                                                       const double fractionalPositionTolerance)
{
    _clusterSpace = clusterSpace;
    _supercell = std::make_shared<Structure>(structure);
    LocalOrbitListGenerator LOLG = LocalOrbitListGenerator(clusterSpace.getPrimitiveOrbitList(), _supercell, fractionalPositionTolerance);

    // Create a full orbit list, used to calculate full cluster vectors.
    _fullOrbitList = LOLG.getFullOrbitList();

    // Map indices in the supercell to offsets in the primitive cell,
    // and for each unique offset, precompute all possible
    // self-contained local orbit lists for the supercell.
    // The latter can be used to calculate local cluster vectors
    // and cluster vector differences.
    for (size_t i = 0; i < _supercell->size(); i++)
    {
        // Find offset of this site in terms of the primitive structure
        Vector3d position = _supercell->positionByIndex(i);
        Vector3i offset = _clusterSpace.primitiveStructure().findLatticeSiteByPosition(position, fractionalPositionTolerance).unitcellOffset();

        // Create map from atom index to offset
        _indexToOffset[i] = offset;

        // If we still have not created a local orbit list for this offset, we should make one
        if (_localOrbitlists.find(offset) == _localOrbitlists.end())
        {
            _localOrbitlists[offset] = LOLG.getLocalOrbitList(offset, true);
        }
    }
}

/**
@details Calculate change in cluster vector upon change in occupation on one site.
@param occupationsBefore Occupation vector for the supercell before the flip
@param flipIndex         Index in the supercell where occupation has changed
@param newOccupation     New atomic number on site index
*/
std::vector<double> ClusterExpansionCalculator::getClusterVectorChange(const py::array_t<int> &occupationsBefore,
                                                                       size_t flipIndex,
                                                                       size_t newOccupation)
{
    if ((size_t)occupationsBefore.size() != _supercell->size())
    {
        throw std::runtime_error("Input occupations and internal supercell structure mismatch in size (ClusterExpansionCalculator::getClusterVectorChange)");
    }
    _supercell->setAtomicNumbers(occupationsBefore);

    if (flipIndex >= _supercell->size())
    {
        throw std::runtime_error("flipIndex larger than the length of the structure (ClusterExpansionCalculator::getClusterVectorChange)");
    }

    return _clusterSpace.getClusterVectorFromOrbitList(_localOrbitlists[_indexToOffset[flipIndex]], _supercell, flipIndex, newOccupation);
}

/**
@details This constructs a cluster vector that only includes clusters that contain the input index.
@param occupations the occupation vector for the supercell
@param index the local index of the supercell
*/
std::vector<double> ClusterExpansionCalculator::getLocalClusterVector(const py::array_t<int> &occupations, int index)
{
    if ((size_t)occupations.size() != _supercell->size())
    {
        throw std::runtime_error("Input occupations and internal supercell structure mismatch in size (ClusterExpansionCalculator::getLocalClusterVector)");
    }
    _supercell->setAtomicNumbers(occupations);
    return _clusterSpace.getClusterVectorFromOrbitList(_localOrbitlists[_indexToOffset[index]], _supercell, index);
}

/**
@details Calculate the cluster vector for a supercell.
@param occupations the occupation vector of the supercell
*/
std::vector<double> ClusterExpansionCalculator::getClusterVector(const py::array_t<int> &occupations)
{
    if ((size_t)occupations.size() != _supercell->size())
    {
        throw std::runtime_error("Input occupations and internal supercell structure mismatch in size (ClusterExpansionCalculator::getClusterVector)");
    }
    _supercell->setAtomicNumbers(occupations);
    return _clusterSpace.getClusterVectorFromOrbitList(_fullOrbitList, _supercell);
}
