#pragma once
#include <vector>
#include <unordered_map>
#include "LatticeSite.hpp"
#include "Structure.hpp"
#include "OrbitList.hpp"
#include "Vector3dCompare.hpp"

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

///clears primToSupercellMap and unique offsets
void reset();

etc...
*/

class LocalOrbitListGenerator
{
  public:
    LocalOrbitListGenerator(){};

    LocalOrbitListGenerator(const OrbitList &, const Structure &);

    ///generate and returns the local orbit list with the input index
    OrbitList getLocalOrbitList(const size_t);

    ///generate and returns the local orbit list with the input offset (require that the offset is in uniquecell offset?)
    OrbitList getLocalOrbitList(const Vector3d &);

    /// Generate the full orbit list from this structure
    OrbitList getFullOrbitList();

    //clears the unordered_map and the vector
    void clear();

    ///Returns the number of unique offsets
    size_t getNumberOfUniqueOffsets() const
    {
        return _uniquePrimcellOffsets.size();
    }

    ///Return the primitive lattice neighbor to supercell lattice neigbhor map
    std::unordered_map<LatticeSite, LatticeSite> getMapFromPrimitiveToSupercell() const
    {
        return _primToSupercellMap;
    }

    ///Returns the unique primitive cells
    std::vector<Vector3d> getUniquePrimitiveCellOffsets() const
    {
        return _uniquePrimcellOffsets;
    }

  private:
    /**
    Maps supercell positions to reference to the primitive cell and find unique primitive cell offsets
    Will loop through all sites in supercell and map them to the primitive structures cell
    and find the unique primitive cell offsets
    */
    void mapSitesAndFindCellOffsets();

    ///Primitive orbit list
    OrbitList _orbitList;

    ///supercell structure from which the local orbit list will be based upon
    Structure _supercell;

    ///this maps a latticeNeighbor from the primitive and get the equivalent in supercell
    std::unordered_map<LatticeSite, LatticeSite> _primToSupercellMap;

    /// Find the position of the atom that is closest to the origin.
    /// This position is used to extracting unit cell offsets later on
    Vector3d getClosestToOrigin() const
    {
        Vector3d closestToOrigin;
        double distanceToOrigin = 1e6;
        for (size_t i = 0; i < _orbitList.getPrimitiveStructure().size(); i++)
        {
            Vector3d position_i = _orbitList.getPrimitiveStructure().getPositions().row(i);
            if ((position_i.norm()) < distanceToOrigin)
            {
                distanceToOrigin = position_i.norm();
                closestToOrigin = position_i;
            }
        }
        return closestToOrigin;
    }

    ///
    Vector3d _positionClosestToOrigin;
    ///The unique offsets of the primitive cell required to "cover" the supercell
    std::vector<Vector3d> _uniquePrimcellOffsets;

    /// The sub permutation matrices that will together map the basis atoms unto the supercell.
    std::vector<Matrix3i> _subPermutationMatrices;

    /// Find the indices of the supercell that are within 1e-3 of the position argument
    std::vector<int> findMatchingSupercellPositions(const Vector3d &position) const;
};
