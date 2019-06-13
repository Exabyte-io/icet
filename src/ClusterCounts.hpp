#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <unordered_set>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "ManyBodyNeighborList.hpp"
#include "Structure.hpp"
#include "Cluster.hpp"
#include "LatticeSite.hpp"
#include "PeriodicTable.hpp"
#include "OrbitList.hpp"

using namespace Eigen;

namespace py = pybind11;

class ClusterCounts
{
public:
  ClusterCounts()
  {
    symprec = 1e-5;
    //empty constructor
  }
  void countLatticeSites(const Structure &,
                         const std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> &);
  void count(const Structure &,
             const std::vector<LatticeSite> &);
  void count(const Structure &, const std::vector<std::vector<LatticeSite>> &,
             const Cluster &, bool);
  void countCluster(const Cluster &, const std::vector<int> &, bool);
  void countOrbitList(const Structure &, const OrbitList &, bool orderIntact, bool permuteSites = false);

  std::unordered_map<Cluster, std::map<std::vector<int>, int>> getClusterCounts() const
  {
    return _clusterCounts;
  }

  /// Returns the cluster counts size i.e. the total number of clusters
  size_t size() const
  {
    return _clusterCounts.size();
  }
  /// Reset cluster counts
  void reset()
  {
    _clusterCounts.clear();
  }

  /**
 Helpful function that prints the cluster counts
  */
  void print()
  {
    for (const auto &map_pair : _clusterCounts)
    {
      int total = 0;
      map_pair.first.print();
      std::cout << "==============" << std::endl;
      for (const auto &element_count_pair : map_pair.second)
      {
        for (auto el : element_count_pair.first)
        {
          std::cout << PeriodicTable::intStr[el] << " ";
        }
        std::cout << map_pair.second.at(element_count_pair.first) << std::endl;
        total += map_pair.second.at(element_count_pair.first);
      }
      std::cout << "Total: " << total << std::endl;
      std::cout << std::endl;
    }
  }
  std::unordered_map<Cluster, std::map<std::vector<int>, int>> _clusterCounts;

private:
  double roundDouble(const double &double_value)
  {
    return round(double_value / symprec) * symprec;
  }
  double symprec;

};
