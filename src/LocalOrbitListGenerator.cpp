#include "LocalOrbitListGenerator.hpp"

LocalOrbitListGenerator::LocalOrbitListGenerator(const OrbitList &orbitList, const Structure &superCell) : _orbitList(orbitList), _supercell(superCell)
{
    _positionClosestToOrigin = getClosestToOrigin();
    mapSitesAndFindCellOffsets();

    // generateSmartOffsets();
}

/**
    Maps supercell positions to reference to the primitive cell and find unique primitive cell offsets
    Will loop through all sites in supercell and map them to the primitive structures cell
    and find the unique primitive cell offsets
    */
void LocalOrbitListGenerator::mapSitesAndFindCellOffsets()
{
    _primToSupercellMap.clear();

    std::set<Vector3d, Vector3dCompare> uniqueCellOffsets;

    //map all sites
    for (size_t i = 0; i < _supercell.size(); i++)
    {
        Vector3d position_i = _supercell.getPositions().row(i);

        LatticeSite primitive_site = _orbitList.getPrimitiveStructure().findLatticeSiteByPosition(position_i);
        Vector3d primitive_position = _orbitList.getPrimitiveStructure().getPositions().row(primitive_site.index());
        // Basically only append offsets to indices that correspond to the atom in the origin
        if ((primitive_position - _positionClosestToOrigin).norm() < 1e-5)
        {
            uniqueCellOffsets.insert(primitive_site.unitcellOffset());
        }
    }

    _uniquePrimcellOffsets.clear();

    _uniquePrimcellOffsets.assign(uniqueCellOffsets.begin(), uniqueCellOffsets.end());

    std::sort(_uniquePrimcellOffsets.begin(), _uniquePrimcellOffsets.end(), Vector3dCompare());
}

///generate and returns the local orbit list with the input index
OrbitList LocalOrbitListGenerator::getLocalOrbitList(const size_t index)
{
    if (index >= _uniquePrimcellOffsets.size())
    {
        std::string errMsg;
        errMsg += "Error: attempting to getLocalOrbitList with index " + std::to_string(index);
        errMsg += " when size of unique offsets are: " + std::to_string(_uniquePrimcellOffsets.size()) + ".";
        throw std::out_of_range(errMsg);
    }

    return _orbitList.getLocalOrbitList(_supercell, _uniquePrimcellOffsets[index], _primToSupercellMap);
}

///generate and returns the local orbit list with the input offset (require that the offset is in uniquecell offset?)
OrbitList LocalOrbitListGenerator::getLocalOrbitList(const Vector3d &primOffset)
{
    auto find = std::find(_uniquePrimcellOffsets.begin(), _uniquePrimcellOffsets.end(), primOffset);
    if (find == _uniquePrimcellOffsets.end())
    {
        std::cout << "Warning: generating local orbit list with offset not found in _uniquePrimcellOffsets" << std::endl;
    }

    return _orbitList.getLocalOrbitList(_supercell, primOffset, _primToSupercellMap);
}

/// Generate the complete orbit list (the sum of all local orbit lists)
OrbitList LocalOrbitListGenerator::getFullOrbitList()
{
    OrbitList orbitList = OrbitList();
    for (size_t i = 0; i < getNumberOfUniqueOffsets(); i++)
    {
        orbitList += getLocalOrbitList(i);
    }
    return orbitList;
}


/// Find the indices of the supercell that are within 1e-3 of the position argument
std::vector<int> LocalOrbitListGenerator::findMatchingSupercellPositions(const Vector3d &position) const
{
    std::vector<int> matchedIndices;
    for (size_t i = 0; i < (size_t) _supercell.getPositions().rows(); i++)
    {
        if ((position - _supercell.getPositionByIndex(i)).norm() < 1e-3)
        {
            matchedIndices.push_back(i);
        }
    }
    return matchedIndices;
}

//clears the unordered_map and the vector
void LocalOrbitListGenerator::clear()
{
    _primToSupercellMap.clear();
    _uniquePrimcellOffsets.clear();
}
