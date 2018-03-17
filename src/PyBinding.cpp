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
        .def("get_permuted_positions", &PermutationMap::getPermutedPositions)
        .def("get_indiced_positions", &PermutationMap::getIndicedPermutedPositions)

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
        .def("count", (void (ClusterCounts::*)(const Structure &, const std::vector<std::vector<LatticeSite>> &, const Cluster &,bool)) & ClusterCounts::count)
        .def("count_orbit_list", &ClusterCounts::countOrbitList)
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

        /* 
        @TODO Remove the usage of these functions
            in favor of the property versions.

            It should mostly be used in clusterspace.
            ------------ START removal -----------------------
        */

        .def("add_equivalent_sites",
             (void (Orbit::*)(const std::vector<LatticeSite> &, bool)) & Orbit::addEquivalentSites,
             py::arg("lattice_neighbors"),
             py::arg("sort") = false)
        .def("add_equivalent_sites",
             (void (Orbit::*)(const std::vector<std::vector<LatticeSite>> &, bool)) & Orbit::addEquivalentSites,
             py::arg("lattice_neighbors"),
             py::arg("sort") = false)
        .def("get_equivalent_sites", &Orbit::getEquivalentSites)
        .def("get_allowed_sites_permutations",&Orbit::getAllowedSitesPermutations)
        .def("get_representative_sites", &Orbit::getRepresentativeSites)
        .def("get_equivalent_sites_permutations", &Orbit::getEquivalentSitesPermutations)
        
        .def("get_representative_cluster", &Orbit::getRepresentativeCluster,
        R"pbdoc(
        The representative cluster
        represents the geometrical
        version of what this orbit is.
        )pbdoc")
        /*
        ------------ END removal -----------------------
        */
        .def_property_readonly("representative_cluster", &Orbit::getRepresentativeCluster,
        R"pbdoc(
        The representative cluster
        represents the geometrical
        version of what this orbit is.
        )pbdoc")
        .def_property("permutations_to_representative", &Orbit::getEquivalentSitesPermutations,&Orbit::setEquivalentSitesPermutations,
               R"pbdoc(
        Get the list of permutations.
        Where permutations_to_representative[i]
        takes self.equivalent_sites[i] to
        the same order as self.representative_sites.

        This can be used if you for example want to
        count elements and are interested in difference
        between ABB, BAB, BBA and so on. If you count the
        lattice sites that are permuted according to
        these permutations then you will get the correct
       counts. )pbdoc")
        .def_property_readonly("order", [](const Orbit &orbit) { return orbit.getRepresentativeCluster().order(); },
        R"pbdoc(
        Returns the order of the orbit.
        The order is the same as the number
        of bodies in the representative cluster
        or the number of lattice sites per element
        in equivalent_sites.
        )pbdoc")
        .def_property_readonly("geometrical_size", [](const Orbit &orbit) { return orbit.getRepresentativeCluster().geometricalSize(); },
        R"pbdoc(        Returns the geometrical size of the
        representative cluster.
        )pbdoc")
        .def_property_readonly("permuted_sites", &Orbit::getPermutedEquivalentSites,
        R"pbdoc(Get the equivalent sites but permuted
        to representative site.)pbdoc")
        .def_property_readonly("representative_sites",&Orbit::getRepresentativeSites,        
        R"pbdoc(
        The representative sites
        is a list of lattice sites
        that are uniquely picked out
        for this orbit which can be
        used to represent and distinguish
        between orbits.
        )pbdoc")
        .def_property("equivalent_sites",&Orbit::getEquivalentSites, &Orbit::setEquivalentSites,
        R"pbdoc(
        List of equivalent Lattice Sites
        )pbdoc")
        .def("get_sites_with_permutation", &Orbit::getSitesWithPermutation,
        R"pbdoc(Return the permuted to representative
        sites of equivalent_sites[index].)pbdoc")
        // .def("get_number_of_duplicates", &Orbit::getNumberOfDuplicates, py::arg("verbosity") = 0)
        .def("get_mc_vectors", &Orbit::getMCVectors,
                R"pbdoc(
        Return the mc vectors for this orbit given the allowed components.
        The mc vectors are returned as a list of tuples

        parameters
       ----------
        allowed_components : list of int
           The allowed components for the lattice sites,
           allowed_components[i] correspond to the number
           of allowed compoments at lattice site
           orbit.representative_sites[i].)pbdoc")
        .def("sort", &Orbit::sortOrbit,
        R"pbdoc(
        Sort the equivalent sites list.
        )pbdoc")
        .def("get_all_possible_mc_vector_permutations", &Orbit::getAllPossibleMCVectorPermutations,
        R"pbdoc(
        Similar to get all permutations but
       needs to be filtered through the
       number of allowed elements.

       parameters
       ----------
       allowed_components : list of int
           The allowed components for the lattice sites,
           allowed_components[i] correspond to the lattice site
           self.representative_sites[i].

        returns all_mc_vectors : list of tuples of int
        )pbdoc")
        .def_property("allowed_permutations", [](const Orbit &orbit) {
             auto permutationSet = orbit.getAllowedSitesPermutations(); 
             std::vector<std::vector<int>> vectorPermutations;
            for(auto vector : permutationSet)
            {
                vectorPermutations.push_back(vector);
            }
             return vectorPermutations; }, [](Orbit &orbit, const std::vector<std::vector<int>> &permutations) {
                
                std::unordered_set<std::vector<int>, VectorHash> setPermutations;
                setPermutations.insert(permutations.begin(),permutations.end());
                 orbit.setAllowedSitesPermutations(setPermutations); },
        R"pbdoc(Get the list of equivalent permutations
        for this orbit.

        If this orbit is a triplet
        and the permutation [0,2,1]
        exists this means that
        The lattice sites [s1, s2, s3]
        are equivalent to [s1, s3, s2]
        This will have the effect that
        for a ternary CE the cluster
        functions (0,1,0) will not
        be considered since it is
        equivalent to (0,0,1).)pbdoc")
        .def_property("permutations_to_representative", &Orbit::getEquivalentSitesPermutations, &Orbit::setEquivalentSitesPermutations,
        R"pbdoc(
        Get the list of permutations.
        Where permutations_to_representative[i]
        takes self.equivalent_sites[i] to
        the same order as self.representative_sites.
    
        This can be used if you for example want to
        count elements and are interested in difference
        between ABB, BAB, BBA and so on. If you count the
        lattice sites that are permuted according to
        these permutations then you will get the correct
        counts.
        )pbdoc")
        .def("__len__", &Orbit::size)

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
        .def("get_primitive_structure", &OrbitList::getPrimitiveStructure)
        .def("__len__", &OrbitList::size)
        .def("print", &OrbitList::print, py::arg("verbosity") = 0)
        // .def("get_supercell_orbit_list", &OrbitList::getSupercellOrbitList)
        ;

    py::class_<LocalOrbitListGenerator>(m, "LocalOrbitListGenerator")
        .def(py::init<const OrbitList &, const Structure &>())
        .def("generate_local_orbit_list", (OrbitList(LocalOrbitListGenerator::*)(const unsigned int)) & LocalOrbitListGenerator::generateLocalOrbitList)
        .def("generate_local_orbit_list", (OrbitList(LocalOrbitListGenerator::*)(const Vector3d &)) & LocalOrbitListGenerator::generateLocalOrbitList)
        .def("generate_full_orbit_list", &LocalOrbitListGenerator::generateFullOrbitList)
        .def("clear", &LocalOrbitListGenerator::clear)
        .def("get_unique_offsets_count", &LocalOrbitListGenerator::getUniqueOffsetsCount)
        .def("get_prim_to_supercell_map", &LocalOrbitListGenerator::getPrimToSupercellMap)
        .def("get_unique_primcell_offsets", &LocalOrbitListGenerator::getUniquePrimcellOffsets);

    py::class_<ClusterSpace>(m, "ClusterSpace", py::dynamic_attr())
        .def(py::init<std::vector<int>, std::vector<std::string>, const OrbitList &>())
        .def("get_cluster_vector", [](const ClusterSpace &ClusterSpace, const Structure &structure) {
            auto cv = ClusterSpace.generateClusterVector(structure);
            return py::array(cv.size(), cv.data());
        })
        .def("get_orbit_list", &ClusterSpace::getOrbitList)
        .def("get_orbit", &ClusterSpace::getOrbit)
        .def("get_cluster_product", &ClusterSpace::getClusterProduct)
        .def("get_cluster_space_info", &ClusterSpace::getClusterSpaceInfo)
        .def("get_cluster_space_size", &ClusterSpace::getClusterSpaceSize)
        .def("get_atomic_numbers", &ClusterSpace::getAtomicNumbers)
        .def("get_cutoffs", &ClusterSpace::getCutoffs)
        .def("_get_primitive_structure", &ClusterSpace::getPrimitiveStructure)
        .def("get_native_clusters", &ClusterSpace::getNativeClusters)
        .def("get_mc_vector_permutations",&ClusterSpace::getMCVectorPermutations)
        .def("get_allowed_occupations",&ClusterSpace::getAllowedOccupations)
        .def("__len__", &ClusterSpace::getClusterSpaceSize);
}
