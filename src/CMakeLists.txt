# Require a modern version of CMake.
CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0 FATAL_ERROR)
PROJECT(icet)
SET(CMAKE_BUILD_TYPE Release)

# This library depends on the pybind11 framework.
ADD_SUBDIRECTORY("3rdparty/pybind11" pybind11)

# List of source files that need to be compiled:
SET(SourceFiles
  Structure.cpp
  PyBinding.cpp
  NeighborList.cpp
  ManyBodyNeighborList.cpp
  Cluster.cpp
  PermutationMap.cpp
  ClusterCounts.cpp
  Orbit.cpp
  OrbitList.cpp
  LocalOrbitListGenerator.cpp
  ClusterSpace.cpp
  Symmetry.cpp
  Geometry.cpp
)

PYBIND11_ADD_MODULE(_icet ${SourceFiles})
SET_TARGET_PROPERTIES(_icet PROPERTIES OUTPUT_NAME "_icet")


