#pragma once
#include <iostream>
#include <vector>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "LatticeSite.hpp"
#include "NeighborList.hpp"
#include "VectorOperations.hpp"

/**
Design approach:
    input pair neighbors and calculate higher order neighbors
    using set intersection.
*/

class ManyBodyNeighborList
{
  public:
    ManyBodyNeighborList()
    {
        //empty...
    }

    std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> build(const std::vector<NeighborList> &, int index, bool);

    void combineToHigherOrder(const NeighborList &nl,
                              std::vector<std::pair<std::vector<LatticeSite>,
			      std::vector<LatticeSite>>> &many_bodyNeighborIndex,
                              const std::vector<LatticeSite> &Ni,
			      std::vector<LatticeSite> &currentOriginalNeighbrs,
			      bool saveBothWays,
			      const size_t);

    /**
    @details Return the lattice sites that appear in two list of lattice sites.
    @param Ni list of lattice sites
    @param Nj another list of lattice sites
    **/
    std::vector<LatticeSite> getIntersection(const std::vector<LatticeSite> &Ni, const std::vector<LatticeSite> &Nj)
    {
        std::vector<LatticeSite> N_intersection;
        N_intersection.reserve(Ni.size());
        std::set_intersection(Ni.begin(), Ni.end(),
                              Nj.begin(), Nj.end(),
                              std::back_inserter(N_intersection));
        return N_intersection;
    }


    void translateAllNi(std::vector<LatticeSite> &Ni, const Vector3d &unitCellOffset) const;

    size_t getNumberOfSites(const unsigned int index) const;

    std::vector<LatticeSite> getSites(const unsigned int &,
                                      const unsigned int &) const;

    void addSinglet(const int, std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> &) const;
    void addPairs(const int, const NeighborList &, std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> &, bool) const;

  private:
    std::vector<LatticeSite> getFilteredNj(const std::vector<LatticeSite> &, const LatticeSite &) const;
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> _latticeNeighbors;
};
