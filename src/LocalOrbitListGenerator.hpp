#pragma once

#include <iomanip>
#include <unordered_map>
#include <vector>

#include "LatticeSite.hpp"
#include "OrbitList.hpp"
#include "Structure.hpp"
#include "VectorOperations.hpp"

/**

This is a small class that has a:

orbit list (from primitive structure)
supercell
list of unique primitive cell offsets that the supercell span
the primToSupercellMap


you can query this object with

///Generate the orbit list from the primitive offset with count i
OrbitList getLocalOrbitList(int i);

std::vector<Vector3d> getUniqueOffsets() const;

std::vector<Vector3d> primToSupercellMap() const;

etc...
*/

class LocalOrbitListGenerator
{
public:
    /// Constructor.
    LocalOrbitListGenerator(){};
    LocalOrbitListGenerator(const OrbitList &, const Structure &, const double);

    /// Generates and returns the local orbit list with the input index.
    OrbitList getLocalOrbitList(const size_t);

    /// Generates the full orbit list from this structure.
    OrbitList getFullOrbitList();

    /// Returns the number of unique offsets.
    size_t getNumberOfUniqueOffsets() const { return _uniquePrimcellOffsets.size(); }

    /// Returns the primitive lattice neighbor to supercell lattice neigbhor map.
    std::unordered_map<LatticeSite, LatticeSite> getMapFromPrimitiveToSupercell() const { return _primToSupercellMap; }

    /// Returns the unique primitive cells
    std::vector<Vector3d> getUniquePrimitiveCellOffsets() const { return _uniquePrimcellOffsets; }

private:
    /// Maps supercell positions to reference.
    void mapSitesAndFindCellOffsets();

    /// Primitive orbit list.
    OrbitList _orbitList;

    /// Supercell structure from which the local orbit list will be based upon.
    Structure _supercell;

    /// Maps a latticeNeighbor from the primitive and get the equivalent in supercell.
    std::unordered_map<LatticeSite, LatticeSite> _primToSupercellMap;

    /// Finds the position of the atom that is closest to the origin.
    Vector3d getClosestToOrigin();

    size_t _indexToClosestAtom;
    /// The unique offsets of the primitive cell required to "cover" the supercell.
    std::vector<Vector3d> _uniquePrimcellOffsets;

    /// Tolerance applied when comparing positions in Cartesian coordinates.
    double _fractionalPositionTolerance;
};
