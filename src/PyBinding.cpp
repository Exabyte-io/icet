#include "Structure.hpp"
#include "NeighborList.hpp"
#include "ManyBodyNeighborList.hpp"
#include "Cluster.hpp"
#include "PermutationMap.hpp"
#include "LatticeSite.hpp"
#include "ClusterCounts.hpp"
#include "LocalOrbitListGenerator.hpp"
#include "ClusterSpace.hpp"
#include <pybind11/pybind11.h>
#include "Symmetry.hpp"
#include "Orbit.hpp"
#include "OrbitList.hpp"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <pybind11/operators.h>

PYBIND11_MODULE(_icet, m)
{

    m.doc() = R"pbdoc(
        Python interface
        ================

        This is the Python interface generated via pybind11 from the C++
        core classes and methods.

        .. toctree::
           :maxdepth: 2

        .. currentmodule:: _icet

        Cluster
        -------
        .. autoclass:: Cluster
           :members:
           :undoc-members:

        ClusterCounts
        -------------
        .. autoclass:: ClusterCounts
           :members:
           :undoc-members:

        ClusterSpace
        ------------
        .. autoclass:: ClusterSpace
           :members:
           :undoc-members:

        LatticeSite
        -----------
        .. autoclass:: LatticeSite
           :members:
           :undoc-members:

        LocalOrbitListGenerator
        -----------------------
        .. autoclass:: LocalOrbitListGenerator
           :members:
           :undoc-members:

        ManyBodyNeighborList
        --------------------
        .. autoclass:: ManyBodyNeighborList
           :members:
           :undoc-members:

        NeighborList
        ------------
        .. autoclass:: NeighborList
           :members:
           :undoc-members:

        Orbit
        -----
        .. autoclass:: Orbit
           :members:
           :undoc-members:

        OrbitList
        ---------
        .. autoclass:: OrbitList
           :members:
           :undoc-members:

        PermutationMap
        --------------
        .. autoclass:: PermutationMap
           :members:
           :undoc-members:

        Structure
        ---------
        .. autoclass:: Structure
           :members:
           :undoc-members:
    )pbdoc";

    // Disable the automatically generated signatures that prepend the
    // docstrings by default.
    py::options options;
    options.disable_function_signatures();

    py::class_<Structure>(m, "Structure",
        R"pbdoc(
             This class stores the cell metric, positions, chemical symbols,
             and periodic boundary conditions that describe a structure. It
             also holds information pertaining to the components that are
             allowed on each site and provides functionality for computing
             distances between sites.
         )pbdoc")
    .def(py::init<>())
    .def(py::init<const Eigen::Matrix<double, Dynamic, 3, Eigen::RowMajor> &,
                  const std::vector<std::string> &,
                  const Eigen::Matrix3d &,
                  const std::vector<bool> &,
                  double>(),
         R"pbdoc(
             Initializes an icet Structure instance.

             Parameters
             ----------
             positions : list of vector
                 list of positions in Cartesian coordinates
             chemical_symbols : list of strings
                 chemical symbol of each case
             cell : 3x3 array
                  cell metric
             pbc : list of booleans
                 periodic boundary conditions
             tolerance : float
                 numerical tolerance imposed when testing for equality of
                 positions and distances
         )pbdoc",
         py::arg("positions"),
         py::arg("chemical_symbols"),
         py::arg("cell"),
         py::arg("pbc"),
         py::arg("tolerance") = 1e-5)
    .def("get_pbc", &Structure::getPBC,
         "Returns the periodic boundary conditions.")
    .def("set_pbc", &Structure::setPBC,
         "Sets the periodic boundary conditions.")
    .def_property("pbc",
                  &Structure::getPBC,
                  &Structure::setPBC,
                  "3-dimensional vector : periodic boundary conditions")
    .def("get_cell", &Structure::getCell,
         "Returns the cell metric.")
    .def("set_cell", &Structure::setCell,
         "Sets the cell metric.")
    .def_property("cell",
                  &Structure::getCell,
                  &Structure::setCell,
                  "3x3 array : cell metric")
    .def("get_positions",
         &Structure::getPositions,
         R"pbdoc(
             Returns the positions in Cartesian coordinates.

             Returns
             -------
             list of NumPy arrays
         )pbdoc")
    .def("set_positions",
         &Structure::setPositions,
         py::arg("positions"),
         R"pbdoc(
             Set the positions in Cartesian coordinates.

             Parameters
             ----------
             positions : list of NumPy arrays\
                 new positions in Cartesian coordinates
         )pbdoc")
    .def_property("positions",
                  &Structure::getPositions,
                  &Structure::setPositions,
                  "list of lists : atomic positions in Cartesian coordinates")
    .def("get_atomic_numbers",
         &Structure::getAtomicNumbers,
         "Returns a list of the species occupying each site by atomic number.")
    .def("set_atomic_numbers",
         &Structure::setAtomicNumbers,
         py::arg("atomic_numbers"),
         R"pbdoc(
             Sets the species occupying each site by atomic number.

             Parameters
             ----------
             atomic_numbers : list of ints
                new species by atomic number
         )pbdoc")
    .def_property("atomic_numbers",
                  &Structure::getAtomicNumbers,
                  &Structure::setAtomicNumbers,
                  "list of ints : atomic numbers of species on each site")
    .def("get_chemical_symbols",
         &Structure::getChemicalSymbols,
         "Returns a list of the species occupying each site by chemical symbol.")
    .def("set_chemical_symbols",
         &Structure::setChemicalSymbols,
         py::arg("chemical_symbols"),
         R"pbdoc(
             Sets the species occupying each site by chemical symbol.

             Parameters
             ----------
             chemical_symbols : list of strings
                new species by chemical symbol
         )pbdoc")
    .def_property("chemical_symbols",
                  &Structure::getChemicalSymbols,
                  &Structure::setChemicalSymbols,
                  "list of strings : chemical symbols of species on each site")
    .def("set_unique_sites",
         &Structure::setUniqueSites,
         py::arg("unique_sites"),
         R"pbdoc(
             Set unique sites.

             This method allows one to specify for each site in the structure
             the unique site it is related to.

             Parameters
             ----------
             unique_sites : list of ints
                site of interest
         )pbdoc")
    .def("get_unique_sites",
         &Structure::getUniqueSites,
         R"pbdoc(
             Returns the unique sites.

             Returns
             -------
             list of ints
         )pbdoc")
    .def_property("unique_sites",
                  &Structure::getUniqueSites,
                  &Structure::setUniqueSites,
                  "list of ints : unique sites")
    .def("get_unique_site",
         &Structure::getUniqueSite,
         py::arg("index"),
         R"pbdoc(
             Returns the unique site.

             Parameters
             ----------
             index : int
                 index of site of interest

             Returns
             -------
             int
                 index of unique site
         )pbdoc")
    .def("get_position",
         &Structure::getPosition,
         py::arg("site"),
         R"pbdoc(
             Returns the position of a specified site

             Parameters
             ----------
             site : LatticeSite object
                site of interest

             Returns
             -------
             vector
                 position in Cartesian coordinates
         )pbdoc")
    .def("get_distance",
         &Structure::getDistance,
         py::arg("index1"),
         py::arg("index2"),
         py::arg("offset1") = Vector3d(0, 0, 0),
         py::arg("offset2") = Vector3d(0, 0, 0),
         R"pbdoc(
             Returns the distance between two sites

             Parameters
             ----------
             index1 : int
                 index of the first site
             index2 : int
                 index of the second site
             offset1 : vector
                 offset to be applied to the first site
             offset2 : vector
                 offset to be applied to the second site

             Returns
             -------
             float
                 distance in length units
         )pbdoc")
    .def("find_site_by_position",
         &Structure::findSiteByPosition,
         py::arg("position"),
         R"pbdoc(
             Returns the index of the site that matches the position.

             Parameters
             ----------
             position : list/NumPy array
                 position in Cartesian coordinates

             Returns
             -------
             int
                 site index
         )pbdoc")
    .def("find_lattice_site_by_position",
         &Structure::findLatticeSiteByPosition,
         py::arg("position"),
         R"pbdoc(
             Returns the lattice site that matches the position.

             Parameters
             ----------
             position : list/NumPy array
                 position in Cartesian coordinates

             Returns
             -------
             LatticeSite object
                 lattice site
         )pbdoc")
    .def("find_lattice_sites_by_positions",
         &Structure::findLatticeSitesByPositions,
         py::arg("positions"),
         R"pbdoc(
             Returns the lattice sites that match the positions.

             Parameters
             ----------
             positions : list of lists/NumPy arrays
                 list of positions in Cartesian coordinates
             Returns
             -------
             list of LatticeSite object
                 list of lattice sites
         )pbdoc")
    .def("__len__", &Structure::size);

py::class_<NeighborList>(m, "NeighborList")
    .def(py::init<const double>(),
         py::arg("cutoff"),
         R"pbdoc(
             Initialize a neighbor list instance.

             Parameters
             ----------
             cutoff : float
                 cutoff to be used for constructing the neighbor list
         )pbdoc")
    .def("build",
         &NeighborList::build,
         py::arg("structure"),
         R"pbdoc(
             Build a neighbor list for the given atomic configuration.

             Parameters
             ----------
             structure : icet Structure instance
                 atomic configuration
         )pbdoc")
    .def("is_neighbor",
         &NeighborList::isNeighbor,
         py::arg("index1"),
         py::arg("index2"),
         py::arg("offset"),
         R"pbdoc(
             Check if two sites are neighbors.

             Parameters
             ----------
             index1 : int
                 index of the first site
             index2 : int
                 index of the second site
             offset : NumPy array
                 offset between the two sites in units of lattice vectors
             Returns
             -------
                 boolean
         )pbdoc")
    .def("get_neighbors",
         &NeighborList::getNeighbors,
         py::arg("index"),
         R"pbdoc(
             Returns a vector of lattice sites that identify the neighbors of site in question.

             Parameters
             ----------
             index : int
                 index of site in structure for which neighbor list was build
             Returns
             -------
             list of LatticeSite instances
         )pbdoc")
    .def("__len__", &NeighborList::size);

py::class_<ManyBodyNeighborList>(m, "ManyBodyNeighborList")
    .def(py::init<>())
    .def("calculate_intersection", &ManyBodyNeighborList::getIntersection)
    .def("build", &ManyBodyNeighborList::build);

py::class_<Cluster>(m, "Cluster")
    .def(py::init<const Structure &,
         const std::vector<LatticeSite> &,
         const bool, const int>(),
         pybind11::arg("structure"),
         pybind11::arg("lattice_sites"),
         pybind11::arg("sorted_cluster") = true,
         pybind11::arg("tag") = 0,
         R"pbdoc(
             Initialize a cluster instance.

             Parameters
             ----------
             structure : icet Structure instance
                 atomic configuration
             lattice_sites : list of int
                 list of lattice sites that form the cluster
             sorted_cluster : boolean
                 True if the cluster is sorted
             tag : int
                 cluster tag
         )pbdoc")
    //.def("count", &Cluster::count)
    //.def("get_count", &Cluster::getCount)
    .def("print", &Cluster::print)
    .def_property_readonly("sites",
                           &Cluster::sites,
                           "list of ints : site indices")
    .def_property_readonly("distances",
                           &Cluster::distances,
                           "list of floats : list of distances between sites")
    .def_property_readonly("sorted",
                           &Cluster::isSorted,
                           "boolean : True/False if the cluster is sorted/unsorted")
    .def_property_readonly("tag",
                           &Cluster::tag,
                           "int : cluster tag (defined for sorted cluster)")
    .def_property_readonly("geometrical_size",
                           &Cluster::geometricalSize,
                           "float : the geometrical size of the cluster")
    .def_property_readonly("order",
                           &Cluster::order,
                           "int : order of the cluster (= number of sites)")
    .def("__hash__", [](const Cluster &cluster) { return std::hash<Cluster>{}(cluster); })
    .def("__len__", &Cluster::order)
    .def(py::self < py::self)
    .def(py::self == py::self);
    ;

py::class_<PermutationMap>(m, "PermutationMap")
    .def(py::init<const std::vector<Vector3d> &,
                  const std::vector<Matrix3d> &>())
    .def("build", &PermutationMap::build)
    .def("get_permutated_positions", &PermutationMap::getPermutatedPositions)
    .def("get_indiced_positions", &PermutationMap::getIndicedPermutatedPositions)

    ;

py::class_<LatticeSite>(m, "LatticeSite")
    .def(py::init<const int, const Vector3d &>())
    .def("print", &LatticeSite::print)
    .def_property("index", &LatticeSite::index, &LatticeSite::setIndex)
    .def_property("unitcell_offset", &LatticeSite::unitcellOffset, &LatticeSite::setUnitcellOffset)
    .def(py::self < py::self)
    .def(py::self == py::self)
    .def(py::self + Eigen::Vector3d())
    .def("__hash__", [](const LatticeSite &latticeNeighbor) { return std::hash<LatticeSite>{}(latticeNeighbor); })

    ;

py::class_<ClusterCounts>(m, "ClusterCounts")
    .def(py::init<>())
    .def("count_lattice_neighbors", &ClusterCounts::countLatticeSites)
    .def("count", (void (ClusterCounts::*)(const Structure &, const std::vector<LatticeSite> &)) & ClusterCounts::count)
    .def("count", (void (ClusterCounts::*)(const Structure &, const std::vector<std::vector<LatticeSite>> &, const Cluster &)) & ClusterCounts::count)
    .def("count_orbit_list",&ClusterCounts::countOrbitList)
    .def("__len__", &ClusterCounts::size)
    .def("reset", &ClusterCounts::reset)
    .def("get_cluster_counts", [](const ClusterCounts &clusterCounts) {
        //&ClusterCounts::getClusterCounts
        py::dict clusterCountDict;
        for (const auto &mapPair : clusterCounts.getClusterCounts())
        {
            py::dict d;
            for (const auto &vecInt_intPair : mapPair.second)
            {
                d[py::tuple(py::cast(vecInt_intPair.first))] = vecInt_intPair.second;
            }
            clusterCountDict[py::cast(mapPair.first)] = d;
        }
        return clusterCountDict;
    })
    .def("print", &ClusterCounts::print);

py::class_<Orbit>(m, "Orbit")
    .def(py::init<const Cluster &>())
    .def("add_equivalent_sites",
         (void (Orbit::*)(const std::vector<LatticeSite> &, bool)) & Orbit::addEquivalentSites,
         py::arg("lattice_neighbors"),
         py::arg("sort")=false)
    .def("add_equivalent_sites",
         (void (Orbit::*)(const std::vector<std::vector<LatticeSite>> &, bool)) & Orbit::addEquivalentSites,
         py::arg("lattice_neighbors"),
         py::arg("sort")=false)
    .def("get_representative_cluster", &Orbit::getRepresentativeCluster)
    .def("get_equivalent_sites", &Orbit::getEquivalentSites)
    .def("get_representative_sites", &Orbit::getRepresentativeSites)
    .def("get_equivalent_sites_permutations", &Orbit::getEquivalentSitesPermutations)
    .def("get_sites_with_permutation", &Orbit::getSitesWithPermutation)
    .def("__len__", &Orbit::size)
    .def("get_number_of_duplicates", &Orbit::getNumberOfDuplicates, py::arg("verbosity") = 0)
    .def("get_MC_vectors",&Orbit::getMCVectors)
    .def(py::self < py::self)
    .def(py::self == py::self)
    .def(py::self + Eigen::Vector3d());

py::class_<OrbitList>(m, "OrbitList")
    .def(py::init<>())
    .def(py::init<const std::vector<NeighborList> &, const Structure &>())
    .def(py::init<const Structure &, const std::vector<std::vector<LatticeSite>> &, const std::vector<NeighborList> &>())
    .def("add_orbit", &OrbitList::addOrbit)
    .def("get_number_of_NClusters", &OrbitList::getNumberOfNClusters)
    .def("get_orbit", &OrbitList::getOrbit)
    .def("clear", &OrbitList::clear)
    .def("sort", &OrbitList::sort)
    .def("get_orbit_list", &OrbitList::getOrbitList)
    .def("get_primitive_structure",&OrbitList::getPrimitiveStructure)
    .def("__len__", &OrbitList::size)
    .def("print", &OrbitList::print, py::arg("verbosity") = 0)
    // .def("get_supercell_orbit_list", &OrbitList::getSupercellOrbitList)
    ;

py::class_<LocalOrbitListGenerator>(m, "LocalOrbitListGenerator")
    .def(py::init<const OrbitList &, const Structure &>())
    .def("generate_local_orbit_list", (OrbitList(LocalOrbitListGenerator::*)(const unsigned int)) & LocalOrbitListGenerator::generateLocalOrbitList)
    .def("generate_local_orbit_list", (OrbitList(LocalOrbitListGenerator::*)(const Vector3d &)) & LocalOrbitListGenerator::generateLocalOrbitList)
    .def("generate_full_orbit_list",  &LocalOrbitListGenerator::generateFullOrbitList)
    .def("clear", &LocalOrbitListGenerator::clear)
    .def("get_unique_offsets_count", &LocalOrbitListGenerator::getUniqueOffsetsCount)
    .def("get_prim_to_supercell_map", &LocalOrbitListGenerator::getPrimToSupercellMap)
    .def("get_unique_primcell_offsets", &LocalOrbitListGenerator::getUniquePrimcellOffsets);

py::class_<ClusterSpace>(m, "ClusterSpace",py::dynamic_attr())
    .def(py::init<std::vector<int>, std::vector<std::string>, const OrbitList &>())
    .def("get_cluster_vector", [](const ClusterSpace &ClusterSpace, const Structure &structure) {
        auto cv = ClusterSpace.generateClusterVector(structure);
        return py::array(cv.size(), cv.data());    
    })
    .def("get_orbit_list",&ClusterSpace::getOrbitList)
    .def("get_orbit", &ClusterSpace::getOrbit)
    .def("get_cluster_product", &ClusterSpace::getClusterProduct)
    .def("get_cluster_space_info", &ClusterSpace::getClusterSpaceInfo)
    .def("get_cluster_space_size", &ClusterSpace::getClusterSpaceSize)
    .def("get_atomic_numbers", &ClusterSpace::getAtomicNumbers)
    .def("get_cutoffs",&ClusterSpace::getCutoffs)
    .def("get_primitive_structure",&ClusterSpace::getPrimitiveStructure)
    .def("get_native_clusters",&ClusterSpace::getNativeClusters)
    .def("__len__", &ClusterSpace::getClusterSpaceSize)
    ;

}
