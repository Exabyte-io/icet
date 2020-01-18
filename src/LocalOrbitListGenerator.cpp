#include "LocalOrbitListGenerator.hpp"

/**
@param orbitList orbit list for the underlying primitive cell
@param supercell supercell structure for which to set up the local orbit list generation
@param fractionalPositionTolerance tolerance applied when comparing positions in fractional coordinates
*/
LocalOrbitListGenerator::LocalOrbitListGenerator(const OrbitList &orbitList,
                                                 const Structure &supercell,
                                                 const double fractionalPositionTolerance)
                                                 : _orbitList(orbitList),
                                                   _supercell(supercell),
                                                   _fractionalPositionTolerance(fractionalPositionTolerance)
{
    _positionClosestToOrigin = getClosestToOrigin();
    mapSitesAndFindCellOffsets();
}

/// @details This position is used for extracting unit cell offsets later on.
Vector3d LocalOrbitListGenerator::getClosestToOrigin()
{
    Vector3d closestToOrigin;
    double distanceToOrigin = 1e6;
    for (size_t i = 0; i < _orbitList.getPrimitiveStructure().size(); i++)
    {
        Vector3d position_i = _orbitList.getPrimitiveStructure().getPositions().row(i);
        LatticeSite lattice_site = _orbitList.getPrimitiveStructure().findLatticeSiteByPosition(position_i, _fractionalPositionTolerance);
        // @todo Can this be removed?
        if (lattice_site.unitcellOffset().norm() > FLOATTYPE_EPSILON)
        {
            continue;
        }
        if (position_i.norm() < distanceToOrigin)
        {
            distanceToOrigin = position_i.norm();
            closestToOrigin = position_i;
            _indexToClosestAtom = i;
        }
    }
    return closestToOrigin;
}

/**
@details Maps supercell positions to reference to the primitive cell and find
unique primitive cell offsets. Loops through all sites in supercell and
map them to the primitive structures cell and find the unique primitive cell
offsets.
*/
void LocalOrbitListGenerator::mapSitesAndFindCellOffsets()
{
    _primToSupercellMap.clear();

    std::set<Vector3d, Vector3dCompare> uniqueCellOffsets;

    // Map all sites
    for (size_t i = 0; i < _supercell.size(); i++)
    {
        Vector3d position_i = _supercell.getPositions().row(i);

        LatticeSite primitive_site = _orbitList.getPrimitiveStructure().findLatticeSiteByPosition(position_i, _fractionalPositionTolerance);

        if (primitive_site.index() == _indexToClosestAtom)
        {
            uniqueCellOffsets.insert(primitive_site.unitcellOffset());
        }
    }

    // If empty: add zero offset
    if (uniqueCellOffsets.size() == 0)
    {
        Vector3d zeroVector = {0.0, 0.0, 0.0};
        uniqueCellOffsets.insert(zeroVector);
    }

    _uniquePrimcellOffsets.clear();

    _uniquePrimcellOffsets.assign(uniqueCellOffsets.begin(), uniqueCellOffsets.end());

    if (_uniquePrimcellOffsets.size() != _supercell.size() / _orbitList.getPrimitiveStructure().size())
    {
        std::ostringstream msg;
        msg << "Wrong number of unitcell offsets found (LocalOrbitListGenerator::mapSitesAndFindCellOffsets)." << std::endl;
        msg << "Expected: " << _supercell.size() / _orbitList.getPrimitiveStructure().size() << std::endl;
        msg << "Found:    " << _uniquePrimcellOffsets.size();
        throw std::runtime_error(msg.str());
    }
    std::sort(_uniquePrimcellOffsets.begin(), _uniquePrimcellOffsets.end(), Vector3dCompare());
}

/**
@details Generates and returns the local orbit list with the input index.
@param index index in unique primitive cell offsets
*/
OrbitList LocalOrbitListGenerator::getLocalOrbitList(const size_t index)
{
    if (index >= _uniquePrimcellOffsets.size())
    {
        std::ostringstream msg;
        msg << "Failed to run with index " << index << " (LocalOrbitListGenerator::getLocalOrbitList)" << std::endl;
        msg << " Size of _uniquePrimcellOffsets: " << _uniquePrimcellOffsets.size();
        throw std::out_of_range(msg.str());
    }
    return _orbitList.getLocalOrbitList(_supercell, _uniquePrimcellOffsets[index], _primToSupercellMap, _fractionalPositionTolerance);
}

/**
@details Generates and returns the local orbit list.
@param primOffset translate the unitcell by this offset and then generate the local orbit list
*/
OrbitList LocalOrbitListGenerator::getLocalOrbitList(const Vector3d &primOffset)
{
    auto find = std::find(_uniquePrimcellOffsets.begin(), _uniquePrimcellOffsets.end(), primOffset);
    if (find == _uniquePrimcellOffsets.end())
    {
        std::cout << "Warning: Generating local orbit list with offset not found in _uniquePrimcellOffsets (LocalOrbitListGenerator::getLocalOrbitList)" << std::endl;
    }
    return _orbitList.getLocalOrbitList(_supercell, primOffset, _primToSupercellMap, _fractionalPositionTolerance);
}

/// Generates the complete orbit list (the sum of all local orbit lists).
OrbitList LocalOrbitListGenerator::getFullOrbitList()
{
    OrbitList orbitList = OrbitList();
    for (size_t i = 0; i < getNumberOfUniqueOffsets(); i++)
    {
        orbitList += getLocalOrbitList(i);
    }

    if (orbitList.size() != _orbitList.size())
    {
        std::ostringstream msg;
        msg << "Full orbitlist size is not the same as local orbitlist size (LocalOrbitListGenerator::getFullOrbitList)" << std::endl;
        msg << " full orbitlist size: " << orbitList.size() << std::endl;
        msg << " local orbitlist size: " << _orbitList.size() << std::endl;
        throw std::runtime_error(msg.str());
    }
    return orbitList;
}

// Clears the unordered_map and the vector.
void LocalOrbitListGenerator::clear()
{
    _primToSupercellMap.clear();
    _uniquePrimcellOffsets.clear();
}
