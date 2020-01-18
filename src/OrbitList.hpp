#pragma once

#include <vector>

#include "Cluster.hpp"
#include "Geometry.hpp"
#include "LatticeSite.hpp"
#include "ManyBodyNeighborList.hpp"
#include "NeighborList.hpp"
#include "Orbit.hpp"
#include "Structure.hpp"
#include "Symmetry.hpp"
#include "VectorOperations.hpp"
#include "VectorOperations.hpp"

/**
container for a sorted list of orbits.
@details Instances of OrbitList serve as a containers for a list of orbits and
provide associated functionality.
*/

class OrbitList
{
  public:

    /// Empty constructor.
    OrbitList() { };

    /// Constructs orbit list from a set of neighbor lists, a permutation matrix, and a structure.
    OrbitList(const Structure &,
              const std::vector<std::vector<LatticeSite>> &,
              const std::vector<NeighborList> &,
              const double);

    /// Sort orbit list.
    void sort(const double);

    /// Adds an orbit.
    void addOrbit(const Orbit &orbit);

    /// Adding-to-existing (+=) operator.
    OrbitList &operator+=(const OrbitList &);

    /// Returns a copy of the orbit of the given index.
    Orbit getOrbit(unsigned int) const;

    /// Returns an orbit in the given (supercell) structure.
    Orbit getSuperCellOrbit(const Structure &,
                            const Vector3d &,
                            const unsigned int,
                            std::unordered_map<LatticeSite, LatticeSite> &,
                            const double) const;

    /// Returns the local orbit list for a site.
    OrbitList getLocalOrbitList(const Structure &,
                                const Vector3d &,
                                std::unordered_map<LatticeSite, LatticeSite> &,
                                const double) const;

    // @todo Add description.
    void addColumnsFromMatrixOfEquivalentSites(std::vector<std::vector<std::vector<LatticeSite>>> &,
                                                   std::unordered_set<std::vector<int>, VectorHash> &,
                                                   const std::vector<LatticeSite> &,
                                                   const std::vector<int> &,
                                                   const std::vector<std::vector<LatticeSite>> &,
                                                   const std::vector<LatticeSite> &, bool) const;

    /// Clears the orbit list.
    void clear() { _orbits.clear(); }

    /// Returns the number of orbits.
    size_t size() const { return _orbits.size(); }

    /// Returns the number of orbits which are made up of a certain number of bodies.
    unsigned int getNumberOfNBodyClusters(unsigned int) const;

    // Returns the first column of the permutation matrix.
    std::vector<LatticeSite> getReferenceLatticeSites(const std::vector<std::vector<LatticeSite>> &,
                                              bool = true) const;

    // Returns rows of the permutation matrix that match the lattice sites.
    std::vector<int> getIndicesOfEquivalentLatticeSites(const std::vector<LatticeSite> &,
                                      const std::vector<LatticeSite> &,
                                      bool = true) const;

    // Returns true if the cluster includes at least on site from the unit cell at the origin.
    bool validCluster(const std::vector<LatticeSite> &) const;

    // @todo Add description.
    void addOrbitsFromPM(const Structure &,
                         const std::vector<std::vector<std::vector<LatticeSite>>> &);

    // @todo Add description.
    void addOrbitFromPM(const Structure &,
                        const std::vector<std::vector<LatticeSite>> &);

    // @todo Add description.
    void checkEquivalentClusters(const double) const;

    // @todo Add description.
    std::vector<LatticeSite> translateSites(const std::vector<LatticeSite> &,
                                            const unsigned int) const;

    /// @todo Add description.
    std::vector<std::vector<LatticeSite>> getSitesTranslatedToUnitcell(const std::vector<LatticeSite> &,
                                                                       bool sortit = true) const;

    /// @todo Add description.
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> getMatchesInPM(const std::vector<std::vector<LatticeSite>> &,
                                                                                      const std::vector<LatticeSite> &) const;

    /// @todo Add description.
    void transformSiteToSupercell(LatticeSite &,
                                  const Structure &,
                                  std::unordered_map<LatticeSite, LatticeSite> &,
                                  const double) const;

    /// Adds the permutation information to the orbits.
    void addPermutationInformationToOrbits(const std::vector<LatticeSite> &, const std::vector<std::vector<LatticeSite>> &);

    /// Returns all columns from the given rows in permutation matrix
    std::vector<std::vector<LatticeSite>> getAllColumnsFromRow(const std::vector<int> &,
                                                               const std::vector<std::vector<LatticeSite>> &,
                                                               bool,
                                                               bool) const;

    /// @todo Clarify description.
    /// First construct rows_sort = sorted(rows)  then returns true/false if rows_sort exists in taken_rows
    bool isRowsTaken(const std::unordered_set<std::vector<int>,
                     VectorHash> &,
                     std::vector<int>) const;

    /// Finds and returns sites in first column of matrix of equivalent sites along with their unit cell translated indistinguishable sites.
    std::vector<std::vector<LatticeSite>> getAllColumnsFromSites(const std::vector<LatticeSite> &,
                                                                 const std::vector<LatticeSite> &,
                                                                 const std::vector<std::vector<LatticeSite>> &) const;

    /// Checks that the lattice neighbors do not have any unitcell offsets in a non-periodic direction.
    bool isSitesPBCCorrect(const std::vector<LatticeSite> &) const;

    /// Removes from each orbit all sites in equivalent sites that involve the given site.
    void removeSitesContainingIndex(const int,
                                    bool);

    /// Removes from each orbit all sites in equivalent sites that _do not_ involve the given site.
    void removeSitesNotContainingIndex(const int, bool);

    /// Returns the first column of the permutation matrix used to construct the orbit list.
    std::vector<LatticeSite> getFirstColumnOfMatrixOfEquivalentSites() const { return _referenceLatticeSites; }

    /// Returns the permutation matrix used to construct the orbit list.
    std::vector<std::vector<LatticeSite>> getMatrixOfEquivalentSites() const { return _matrixOfEquivalentSites; }

    /// Removes all equivalent sites that exist both in this orbit list and the input orbit list.
    void subtractSitesFromOrbitList(const OrbitList &);

    /// Removes an orbit identified by its index.
    void removeOrbit(const size_t);

    /// Removes all orbits that have inactive sites.
    void removeInactiveOrbits(const Structure &);

    /// Returns the orbits in this orbit list.
    std::vector<Orbit> getOrbits() const { return _orbits; }

    /// Returns the primitive structure.
    Structure getPrimitiveStructure() const { return _primitiveStructure; }

    /// Sets primitive structure.
    void setPrimitiveStructure(const Structure &primitive) { _primitiveStructure = primitive; }

public:

    /// Contains all the orbits in the orbit list.
    /// @todo why is this not private?
    std::vector<Orbit> _orbits;

private:

    /// @todo Add description.
    std::vector<LatticeSite> _referenceLatticeSites;

    /// Permutation matrix.
    std::vector<std::vector<LatticeSite>> _matrixOfEquivalentSites;

    /// @todo Add description.
    /// @todo Why is this method private but its overloaded buddy is not?
    int findOrbitIndex(const Cluster &, const std::unordered_map<Cluster, int> &) const;

    /// Primitive structure for which orbit list was constructed.
    Structure _primitiveStructure;
};
