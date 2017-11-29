#include "LocalOrbitlistGenerator.hpp"

LocalOrbitlistGenerator::LocalOrbitlistGenerator(const OrbitList &primitiveOrbitlist, const Structure &superCell) : _orbitlist(primitiveOrbitlist), _supercell(superCell)
{
    mapSitesAndFindCellOffsets();
}

/**
    Maps supercell positions to reference to the primitive cell and find unique primitive cell offsets
    Will loop through all sites in supercell and map them to the primitive structures cell
    and find the unique primitive cell offsets
    */
void LocalOrbitlistGenerator::mapSitesAndFindCellOffsets()
{
    _primToSupercellMap.clear();

    std::set<Vector3d, Vector3dCompare> uniqueCellOffsets;

    //map all sites
    for (size_t i = 0; i < _supercell.size(); i++)
    {
        Vector3d position_i = _supercell.getPositions().row(i);

        LatticeSite primitive_site = _orbitlist.getPrimitiveStructure().findLatticeSiteByPosition(position_i);
        // LatticeSite super_site = _supercell.findLatticeSiteByPosition(position_i);

        // _primToSupercellMap[primitive_site] = super_site;
        uniqueCellOffsets.insert(primitive_site.unitcellOffset());
    }

    _uniquePrimcellOffsets.clear();

    _uniquePrimcellOffsets.assign(uniqueCellOffsets.begin(), uniqueCellOffsets.end());

    std::sort(_uniquePrimcellOffsets.begin(), _uniquePrimcellOffsets.end(), Vector3dCompare());
}

///generate and returns the local orbitlist with the input index
OrbitList LocalOrbitlistGenerator::generateLocalOrbitlist(const unsigned int index)
{
    if (index >= _uniquePrimcellOffsets.size())
    {
        std::string errMsg = "Error: attempting to generateLocalOrbitlist with index " + std::to_string(index) + " when size of unique offsets are: " + std::to_string(_uniquePrimcellOffsets.size()) + ".";
        throw std::out_of_range(errMsg);
    }

    return _orbitlist.getLocalOrbitList(_supercell, _uniquePrimcellOffsets[index], _primToSupercellMap);
}

///generate and returns the local orbitlist with the input offset (require that the offset is in uniquecell offset?)
OrbitList LocalOrbitlistGenerator::generateLocalOrbitlist(const Vector3d &primOffset)
{
    auto find = std::find(_uniquePrimcellOffsets.begin(), _uniquePrimcellOffsets.end(), primOffset);
    if (find == _uniquePrimcellOffsets.end())
    {
        std::cout << "Warning: generating local orbitlist with offset not found in _uniquePrimcellOffsets" << std::endl;
    }

    return _orbitlist.getLocalOrbitList(_supercell, primOffset, _primToSupercellMap);
}


/// Generate the complete orbitlist (the sum of all local orbitlists)
OrbitList LocalOrbitlistGenerator::generateFullOrbitlist()
{
    OrbitList orbitList = OrbitList();
    for(int i = 0; i < getUniqueOffsetsCount(); i++)
    {
        orbitList += generateLocalOrbitlist(i);
    }
    return orbitList;


}


//clears the unordered_map and the vector
void LocalOrbitlistGenerator::clear()
{
    _primToSupercellMap.clear();
    _uniquePrimcellOffsets.clear();
}
