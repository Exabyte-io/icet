#include <iostream>
#include <sstream>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

#include "Cluster.hpp"
#include "ClusterCounts.hpp"
#include "ClusterExpansionCalculator.hpp"
#include "ClusterSpace.hpp"
#include "LatticeSite.hpp"
#include "LocalOrbitListGenerator.hpp"
#include "ManyBodyNeighborList.hpp"
#include "NeighborList.hpp"
#include "Orbit.hpp"
#include "OrbitList.hpp"
#include "PeriodicTable.hpp"
#include "MatrixOfEquivalentPositions.hpp"
#include "Structure.hpp"
#include "Symmetry.hpp"

PYBIND11_MODULE(_icet, m)
{

    m.doc() = R"pbdoc(
        Python-C++ interface
        ====================

        This is the Python interface generated via pybind11 from the C++
        core classes and methods.

        .. toctree::
           :maxdepth: 2

        .. currentmodule:: _icet

        Cluster
        -------
        .. autoclass:: Cluster
           :members:

        ClusterCounts
        -------------
        .. autoclass:: ClusterCounts
           :members:

        ClusterSpace
        ------------
        .. autoclass:: ClusterSpace
           :members:

        LatticeSite
        -----------
        .. autoclass:: LatticeSite
           :members:

        LocalOrbitListGenerator
        -----------------------
        .. autoclass:: LocalOrbitListGenerator
           :members:

        ManyBodyNeighborList
        --------------------
        .. autoclass:: ManyBodyNeighborList
           :members:

        MatrixOfEquivalentPositions
        ---------------------------
        .. autoclass:: MatrixOfEquivalentPositions
           :members:

        NeighborList
        ------------
        .. autoclass:: NeighborList
           :members:

        Orbit
        -----
        .. autoclass:: Orbit
           :members:

        _OrbitList
        ----------
        .. autoclass:: _OrbitList
           :members:

        Structure
        ---------
        .. autoclass:: Structure
           :members:
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

        Parameters
        ----------
        positions : list(list(float))
            list of positions in Cartesian coordinates
        chemical_symbols : list(str)
            chemical symbol of each case
        cell : list(list(float))
             cell metric
        pbc : list(bool)
            periodic boundary conditions
        )pbdoc")
        .def(py::init<>())
        .def(py::init<const Eigen::Matrix<double, Dynamic, 3, Eigen::RowMajor> &,
                      const std::vector<std::string> &,
                      const Eigen::Matrix3d &,
                      const std::vector<bool> &>(),
             "Initializes an icet Structure instance.",
             py::arg("positions"),
             py::arg("chemical_symbols"),
             py::arg("cell"),
             py::arg("pbc"))
        .def_property(
            "pbc",
            &Structure::getPBC,
            &Structure::setPBC,
            "list(int) : periodic boundary conditions")
        .def_property(
            "cell",
            &Structure::getCell,
            &Structure::setCell,
            "list(list(float)) : cell metric")
        .def_property(
            "positions",
            &Structure::getPositions,
            &Structure::setPositions,
            "list(list(float)) : atomic positions in Cartesian coordinates")
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
             atomic_numbers : list(int)
                new species by atomic number
         )pbdoc")
        .def_property("atomic_numbers",
                      &Structure::getAtomicNumbers,
                      &Structure::setAtomicNumbers,
                      "list(int) : atomic numbers of species on each site")
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
             chemical_symbols : list(str)
                new species by chemical symbol
         )pbdoc")
        .def_property("chemical_symbols",
                      &Structure::getChemicalSymbols,
                      &Structure::setChemicalSymbols,
                      "list(str) : chemical symbols of species on each site")
        .def("set_unique_sites",
             &Structure::setUniqueSites,
             py::arg("unique_sites"),
             R"pbdoc(
             Sets the unique sites.

             This method allows one to specify for each site in the structure
             the unique site it is related to.

             Parameters
             ----------
             unique_sites : list(int)
                site of interest
         )pbdoc")
        .def("get_unique_sites",
             &Structure::getUniqueSites,
             R"pbdoc(
             Returns the unique sites.

             Returns
             -------
             list(int)
         )pbdoc")
        .def("set_number_of_allowed_species", (void (Structure::*)(const std::vector<int> &)) & Structure::setNumberOfAllowedSpecies,
             py::arg("numbersOfAllowedSpecies"),
             R"pbdoc(
             Sets the number of allowed species on each site.

             This method allows one to specify for each site in the structure
             the number of species allowed on that site.

             Parameters
             ----------
             numbersOfAllowedSpecies : list(int)
             )pbdoc")
        .def("set_number_of_allowed_species",
             (void (Structure::*)(const int)) & Structure::setNumberOfAllowedSpecies,
             py::arg("numbersOfAllowedSpecies"),
             R"pbdoc(
             Sets the number of allowed species on each site.

             This method allows one to specify for each site in the structure
             the number of species allowed on that site.

             Parameters
             ----------
             numbersOfAllowedSpecies : int
             )pbdoc")
        .def_property(
             "unique_sites",
             &Structure::getUniqueSites,
             &Structure::setUniqueSites,
             "list(int) : unique sites")
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
        .def("find_lattice_site_by_position",
             &Structure::findLatticeSiteByPosition,
             py::arg("position"),
             py::arg("fractional_position_tolerance"),
             R"pbdoc(
             Returns the lattice site that matches the position.

             Parameters
             ----------
             position : list or ndarray
                 position in Cartesian coordinates
             fractional_position_tolerance : float
                 tolerance for positions in fractional coordinates

             Returns
             -------
             _icet.LatticeSite
                 lattice site
             )pbdoc")
        .def("find_lattice_sites_by_positions",
             &Structure::findLatticeSitesByPositions,
             py::arg("positions"),
             py::arg("fractional_position_tolerance"),
             R"pbdoc(
             Returns the lattice sites that match the positions.

             Parameters
             ----------
             positions : list(list) or list(ndarray)
                 list of positions in Cartesian coordinates
             fractional_position_tolerance : float
                 tolerance for positions in fractional coordinates

             Returns
             -------
             list(_icet.LatticeSite)
                 list of lattice sites
             )pbdoc")
        .def("__len__", &Structure::size);

    py::class_<NeighborList>(m, "NeighborList")
        .def(py::init<const double>(),
             py::arg("cutoff"),
             R"pbdoc(
             Initializes a neighbor list instance.

             Parameters
             ----------
             cutoff : float
                 cutoff to be used for constructing the neighbor list
             )pbdoc")
        .def("build",
             &NeighborList::build,
             py::arg("structure"),
             py::arg("position_tolerance"),
             R"pbdoc(
             Builds a neighbor list for the given atomic configuration.

             Parameters
             ----------
             structure : icet.Structure
                 atomic configuration
             position_tolerance : float
                 tolerance applied when evaluating positions in Cartesian coordinates
             )pbdoc")
        .def("get_neighbors",
             &NeighborList::getNeighbors,
             py::arg("index"),
             R"pbdoc(
             Returns a list of lattice sites that identify the neighbors of site in question.

             Parameters
             ----------
             index : int
                 index of site in structure for which neighbor list was build

             Returns
             -------
             list(_icet.LatticeSite)
                list of lattice sites
             )pbdoc")
        .def("__len__", &NeighborList::size);

    // @todo document ManyBodyNeighborList in pybindings
    py::class_<ManyBodyNeighborList>(m, "ManyBodyNeighborList")
        .def(py::init<>())
        .def("calculate_intersection", &ManyBodyNeighborList::getIntersection)
        .def("build", &ManyBodyNeighborList::build);

    py::class_<Cluster>(m, "Cluster")
        .def(py::init<const Structure &,
                      const std::vector<LatticeSite> &,
                      const int>(),
             py::arg("structure"),
             py::arg("lattice_sites"),
             py::arg("tag") = 0,
             R"pbdoc(
             Initializes a cluster instance.

             Parameters
             ----------
             structure : icet Structure instance
                 atomic configuration
             lattice_sites : list(int)
                 list of lattice sites that form the cluster
             tag : int
                 cluster tag
             )pbdoc")
        .def_property_readonly(
             "distances",
             &Cluster::distances,
             "list(float) : list of distances between sites")
        .def_property_readonly(
             "sites",
             &Cluster::sites,
             "list(int) : list of distances between sites")
        .def_property(
             "tag",
             &Cluster::tag, &Cluster::setTag,
             "int : cluster tag (defined for sorted cluster)")
        .def_property_readonly(
             "radius",
             &Cluster::radius,
             "float : the radius of the cluster")
        .def_property_readonly(
             "order",
             &Cluster::order,
             "int : order of the cluster (= number of sites)")
        .def("__hash__",
             [](const Cluster &cluster)
             {
                 return std::hash<Cluster>{}(cluster);
             })
        .def("__len__",
             &Cluster::order)
        .def("__str__",
             [](const Cluster &cluster)
             {
                 std::ostringstream msg;
                 msg << "radius: " << cluster.radius();
                 msg << " vertex distances:";
                 for (const auto dist : cluster.distances())
                 {
                     msg << " " << std::to_string(dist);
                 }
                 return msg.str();
             })
        .def(py::self == py::self)
        .def(py::self < py::self);
    ;

    py::class_<::MatrixOfEquivalentPositions>(m, "MatrixOfEquivalentPositions",
             R"pbdoc(
             This class handles a matrix of equivalent positions. Each row
             corresponds to a set of symmetry equivalent positions. The entry in the
             first column is commonly treated as the representative position.
    
             Parameters
             ----------
             translations : list(list(float))
                 translational symmetry operations
             rotations : list(list(float))
                 rotational symmetry operations
             )pbdoc")
        .def(py::init<const std::vector<Vector3d> &,
                      const std::vector<Matrix3d> &>(),
             py::arg("translations"),
             py::arg("rotations"),
             R"pbdoc(Initializes a matrix of equivalent positions.)pbdoc")
        .def("build",
             &MatrixOfEquivalentPositions::build,
             py::arg("fractional_positions"),
             R"pbdoc(Generates the matrix of symmetry equivalent positions given a set of input
             coordinates. To this end, the function uses the rotational and translational
             symmetries provided when initializing the object.

             Parameters
             ----------
			 fractional_positions : list(list(float))
			 	positions of sites in fractional coordinates

             )pbdoc")
        .def("get_equivalent_positions",
             &MatrixOfEquivalentPositions::getEquivalentPositions,
        	 "Returns the matrix of symmetry equivalent positions.")
    ;

    py::class_<LatticeSite>(m, "LatticeSite")
        .def(py::init<const int, const Vector3d &>())
        .def_property(
             "index",
             &LatticeSite::index,
             &LatticeSite::setIndex,
             "int : site index")
        .def_property(
             "unitcell_offset",
             &LatticeSite::unitcellOffset,
             &LatticeSite::setUnitcellOffset,
             "list(int) : unit cell offset (in units of the cell vectors)")
        .def(py::self < py::self)
        .def(py::self == py::self)
        .def(py::self + Eigen::Vector3d())
        .def("__hash__", [](const LatticeSite &latticeNeighbor) { return std::hash<LatticeSite>{}(latticeNeighbor); })
    ;

    // @todo document ClusterCounts in pybindings
    py::class_<ClusterCounts>(m, "ClusterCounts")
        .def(py::init<>())
        .def("count",
             (void (ClusterCounts::*)(const Structure &,
                                      const std::vector<std::vector<LatticeSite>> &,
                                      const Cluster &,
                                      bool)) & ClusterCounts::count,
             R"pbdoc(
             Counts the vectors in `lattice_sites` assuming these sets of sites are
             represented by the cluster `cluster`.

             Parameters
             ----------
             structure : _icet.Structure
                structure that will have its clusters counted
             lattice_sites : list(list(_icet.LatticeSite))
                group of sites, represented by `cluster` that will be counted
             cluster : _icet.Cluster
                cluster used as identification on what sites the clusters belong to
             order_intact : bool
                if true the order of the sites will remain the same otherwise the
                vector of species being counted will be sorted
             )pbdoc",
             py::arg("structure"),
             py::arg("lattice_sites"),
             py::arg("cluster"),
             py::arg("order_intact"))
        .def("count_orbit_list", &ClusterCounts::countOrbitList,
             R"pbdoc(
             Counts sites in orbit list.

             Parameters
             ----------
             structure : _icet.Structure
                atomic configuration
             orbit_list : _icet.OrbitList
                orbit list
             order_intact : bool
                if true do not reorder clusters
                before comparison (i.e., ABC != ACB)
             permute_sites : bool
                if true the sites will be permuted according to the permutations associated with the orbit
             )pbdoc",
             py::arg("structure"),
             py::arg("orbit_list"),
             py::arg("order_intact"),
             py::arg("permute_sites"))
        .def("__len__", &ClusterCounts::size)
        .def("reset", &ClusterCounts::reset)
        .def("get_cluster_counts", [](const ClusterCounts &clusterCounts) {
            py::dict clusterCountDict;
            for (const auto &mapPair : clusterCounts.getClusterCounts())
            {
                py::dict d;
                for (const auto &vecInt_intPair : mapPair.second)
                {
                    py::list element_symbols;
                    for (auto el : vecInt_intPair.first)
                    {
                        auto getElementSymbols = PeriodicTable::intStr[el];
                        element_symbols.append(getElementSymbols);
                    }
                    d[py::tuple(element_symbols)] = vecInt_intPair.second;
                }
                clusterCountDict[py::cast(mapPair.first)] = d;
            }
            return clusterCountDict;
        })
    ;

    // @todo convert getters to properties
    // @todo document Orbit in pybindings
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
        .def("get_equivalent_sites",
             &Orbit::getEquivalentSites)
        .def("get_allowed_sites_permutations",
             &Orbit::getAllowedSitesPermutations)
        .def("get_representative_sites",
             &Orbit::getRepresentativeSites)
        .def("get_equivalent_sites_permutations",
             &Orbit::getPermutationsOfEquivalentSites)

        .def("get_representative_cluster", &Orbit::getRepresentativeCluster,
             R"pbdoc(
             The representative cluster represents the geometrical version of what
             this orbit is.
             )pbdoc")
        /*
        ------------ END removal -----------------------
        */
        .def_property_readonly(
             "representative_cluster", &Orbit::getRepresentativeCluster,
             R"pbdoc(
             The representative cluster is the cluster to which all other symmetry equivalent clusters can be related.
             )pbdoc")
        .def_property(
             "permutations_to_representative",
             &Orbit::getPermutationsOfEquivalentSites, &Orbit::setEquivalentSitesPermutations,
             R"pbdoc(
             Get the list of permutations. Where `permutations_to_representative[i]` takes
             `self.equivalent_sites[i]` to the same order as `self.representative_sites`.

             This can be used if one for example wants to count elements and is
             interested in difference between ABB, BAB, BBA and so on. If one
             counts the lattice sites that are permuted according to these
             permutations one will get the correct counts.
             )pbdoc")
        .def_property_readonly(
             "order",
             [](const Orbit &orbit) { return orbit.getRepresentativeCluster().order(); },
             R"pbdoc(
             Returns the order of the orbit. The order is the same as the numberof bodies in the
             representative cluster or the number of lattice sites per element in equivalent_sites.
             )pbdoc")
        .def_property_readonly(
             "radius",
             [](const Orbit &orbit) { return orbit.getRepresentativeCluster().radius(); },
             R"pbdoc(
             Returns the radius of the
             representative cluster.
             )pbdoc")
        .def_property_readonly(
             "permuted_sites",
             &Orbit::getPermutedEquivalentSites,
             "equivalent sites but permuted to representative site.")
        .def_property_readonly(
             "representative_sites", &Orbit::getRepresentativeSites,
             R"pbdoc(
             The representative sites is a list of lattice sites that are
             uniquely picked out for this orbit which can be used to represent
             and distinguish between orbits.
             )pbdoc")
        .def_property(
             "equivalent_sites",
             &Orbit::getEquivalentSites,
             &Orbit::setEquivalentSites,
             "list of equivalent lattice sites")
        .def("get_permuted_sites_by_index",
             &Orbit::getPermutedSitesByIndex,
             R"pbdoc(
             Returns the equivalent sites at position `index` using
             the permutation of the representative cluster.

             Parameters
             ----------
             index : int
                index of site to return
             )pbdoc",
             py::arg("index"))
        .def("get_mc_vectors", &Orbit::getMultiComponentVectors,
             R"pbdoc(
             Return the mc vectors for this orbit given the allowed components.
             The mc vectors are returned as a list of tuples

             Parameters
             ----------
             allowed_components : list(int)
                The allowed components for the lattice sites,
                allowed_components[i] correspond to the number
                of allowed compoments at lattice site
                orbit.representative_sites[i].)pbdoc")
        .def("sort", &Orbit::sort,
             R"pbdoc(
             Sorts the list of equivalent sites.
             )pbdoc")
        .def("get_all_possible_mc_vector_permutations",
             &Orbit::getAllPossibleMultiComponentVectorPermutations,
             R"pbdoc(
             Similar to get all permutations but needs to be filtered through the number of allowed elements.

             Parameters
             ----------
             allowed_components : list(int)
                 The allowed components for the lattice sites,
                 `allowed_components[i]` correspond to the lattice site
                 `self.representative_sites[i]`.

             returns all_mc_vectors : list(list(int)
             )pbdoc")
        .def_property("allowed_permutations",
             [](const Orbit &orbit)
             {
                 std::set<std::vector<int>> allowedPermutations = orbit.getAllowedSitesPermutations();
                 std::vector<std::vector<int>> retPermutations(allowedPermutations.begin(), allowedPermutations.end());
                 return retPermutations;
             },
             [](Orbit &orbit, const std::vector<std::vector<int>> &newPermutations)
             {
                 std::set<std::vector<int>> allowedPermutations;
                 for (const auto &perm : newPermutations)
                 {
                     allowedPermutations.insert(perm);
                 }
                 orbit.setAllowedSitesPermutations(allowedPermutations);
             },
             R"pbdoc(
             Gets the list of equivalent permutations for this orbit. If this
             orbit is a triplet and the permutation [0,2,1] exists this means
             that The lattice sites [s1, s2, s3] are equivalent to [s1, s3,
             s2] This will have the effect that for a ternary CE the cluster
             functions (0,1,0) will not be considered since it is equivalent
             to (0,0,1).
             )pbdoc")
        .def_property(
             "permutations_to_representative",
             &Orbit::getPermutationsOfEquivalentSites,
             &Orbit::setEquivalentSitesPermutations,
             R"pbdoc(
             list of permutations;
             permutations_to_representative[i] takes self.equivalent_sites[i] to
             the same order as self.representative_sites.

             This can be used if you for example want to count elements and are
             interested in difference between ABB, BAB, BBA and so on. If you count
             the lattice sites that are permuted according to these permutations
             then you will get the correct counts.
             )pbdoc")
        .def("__len__", &Orbit::size)
        .def("__str__",
             [](const Orbit &orbit)
             {
                 std::ostringstream msg;
                 msg << "radius: " << orbit.radius();
                 msg << "  equivalent_sites:";
                 for (const auto sites : orbit._equivalentSites)
                 {
                     msg << "  ";
                     for (const auto site : sites)
                     {
                         msg << " " << site;
                     }
                 }
                 return msg.str();
             })
        .def(py::self < py::self)
        .def(py::self + Eigen::Vector3d());

    py::class_<OrbitList>(m, "_OrbitList")
        .def(py::init<>())
        .def(py::init<const Structure &,
                      const std::vector<std::vector<LatticeSite>> &,
                      const std::vector<NeighborList> &,
                      const double>(),
             R"pbdoc(
             Constructs an OrbitList object from a matrix of equivalent sites.

             Parameters
             ----------
             structure : _icet.Structure
                 (supercell) structure for which to generate orbit list
             matrix_of_equivalent_sites : list(list(_icet.LatticeSite))
                 matrix of symmetry equivalent sites
             neighbor_lists : list(icet.NeighborList)
                 neighbor lists for each (cluster) order
             position_tolerance
                 tolerance applied when comparing positions in Cartesian coordinates
             )pbdoc",
             py::arg("structure"),
             py::arg("matrix_of_equivalent_sites"),
             py::arg("neighbor_lists"),
             py::arg("position_tolerance"))
        .def_property_readonly(
             "orbits",
             &OrbitList::getOrbits,
             "list(_icet.Orbit) : list of orbits")
        .def("get_orbit_list", &OrbitList::getOrbits,
             "Returns the list of orbits")
        .def("check_equivalent_clusters",
             &OrbitList::checkEquivalentClusters,
             R"pbdoc(
             Debug function for checking that all equivalent sites in every
             orbit yield the same radius.

             Parameters
             ----------
             position_tolerance : float
                 tolerance applied when evaluating positions in Cartesian coordinates
             )pbdoc",
             py::arg("position_tolerance"))
        .def("add_orbit",
             &OrbitList::addOrbit,
             "Adds an orbit.")
        .def("get_number_of_nbody_clusters",
             &OrbitList::getNumberOfNBodyClusters,
             "Returns the number of orbits.")
        .def("get_orbit",
             &OrbitList::getOrbit,
             "Returns a copy of the orbit at position i in the orbit list.")
        .def("_remove_inactive_orbits",
             &OrbitList::removeInactiveOrbits)
        .def("clear",
             &OrbitList::clear,
             "Clears the list of orbits.")
        .def("sort", &OrbitList::sort,
             R"pbdoc(
             Sorts the orbits by order and radius.

             Parameters
             ----------
             position_tolerance : float
                 tolerance applied when comparing positions in Cartesian coordinates
             )pbdoc",
             py::arg("position_tolerance"))
        .def("remove_orbit",
             &OrbitList::removeOrbit,
             R"pbdoc(
             Removes the orbit with the input index.

             Parameters
             ---------
             index : int
                 index of the orbit to be removed
             )pbdoc")
        .def("_is_row_taken",
             &OrbitList::isRowsTaken,
             R"pbdoc(
             Returns true if rows exist in taken_rows.

             Parameters
             ----------
             taken_rows: set(tuple(int))
                 unique collection of row index
             rows: list(int)
                 row indices
             )pbdoc",
             py::arg("taken_rows"),
             py::arg("rows"))
        .def("_get_sites_translated_to_unitcell",
             &OrbitList::getSitesTranslatedToUnitcell,
             R"pbdoc(
             Returns a set of sites where at least one site is translated inside the unit cell.

             Parameters
             ----------
             lattice_neighbors: list(_icet.LatticeSite)
                set of lattice sites that might be representative for a cluster
             sort_it: bool
                If true sort translasted sites.
             )pbdoc",
             py::arg("lattice_neighbors"),
             py::arg("sort_it"))
        .def("_get_all_columns_from_sites",
             &OrbitList::getAllColumnsFromSites,
             R"pbdoc(
             Finds the sites in column1, extract and return all columns along with their unit cell
             translated indistinguishable sites.

             Parameters
             ----------
             sites : list(_icet.LatticeSite)
                 sites that correspond to the columns that will be returned
             column1 : list(_icet.LatticeSite)
                 sites in the first column of matrix of equivalent sites
             matrix_of_equivalent_sites : list(list(_icet.LatticeSite))
                 matrix of equivalent sites
             )pbdoc",
             py::arg("sites"),
             py::arg("column1"),
             py::arg("permutation_matrix"))
        .def("get_primitive_structure",
             &OrbitList::getPrimitiveStructure,
             "Returns the primitive atomic structure used to construct the OrbitList instance.")
        .def("__len__",
             &OrbitList::size,
             "Returns the total number of orbits counted in the OrbitList instance.")
        .def_property_readonly("matrix_of_equivalent_positions",
             &OrbitList::getMatrixOfEquivalentSites,
             "list(list(_icet.LatticeSite)) : matrix_of_equivalent_positions")
        ;

    py::class_<LocalOrbitListGenerator>(m, "LocalOrbitListGenerator")
        .def(py::init<const OrbitList &,
                      const Structure &,
                      const double>(),
             R"pbdoc(
             Constructs a LocalOrbitListGenerator object from an orbit list
             and a structure.

             Parameters
             ----------
             orbit_list : _icet.OrbitList
                 an orbit list set up from a primitive structure
             supercell : _icet.Structure
                 supercell build up from the same primitive structure used to set the input orbit list
             fractional_position_tolerance : float
                 tolerance for positions in fractional coordinates
             )pbdoc",
             py::arg("orbit_list"),
             py::arg("supercell"),
             py::arg("fractional_position_tolerance"))
        .def("generate_local_orbit_list",
             (OrbitList(LocalOrbitListGenerator::*)(const size_t)) & LocalOrbitListGenerator::getLocalOrbitList,
             R"pbdoc(
             Generates and returns the local orbit list from an input index corresponding a specific offset of
             the primitive structure.

             Parameters
             ----------
             index : int
                 index of the unique offsets list
             )pbdoc",
             py::arg("index"))
        .def("generate_local_orbit_list",
             (OrbitList(LocalOrbitListGenerator::*)(const Vector3d &)) & LocalOrbitListGenerator::getLocalOrbitList,
             R"pbdoc(
             Generates and returns the local orbit list from a specific offset of the primitive structure.

             Parameters
             ----------
             unique_offset : NumPy array
                 offset of the primitive structure
             )pbdoc",
             py::arg("unique_offset"))
        .def("generate_full_orbit_list",
             &LocalOrbitListGenerator::getFullOrbitList,
             R"pbdoc(
             Generates and returns a local orbit list, which orbits included the equivalent sites
             of all local orbit list in the supercell.
             )pbdoc")
        .def("clear",
             &LocalOrbitListGenerator::clear,
             "Clears the list of offsets and primitive-to-supercell map of the LocalOrbitListGenerator object.")
        .def("get_number_of_unique_offsets",
             &LocalOrbitListGenerator::getNumberOfUniqueOffsets,
             "Returns the number of unique offsets")
        .def("_get_primitive_to_supercell_map",
             &LocalOrbitListGenerator::getMapFromPrimitiveToSupercell,
             "Returns the primitive to supercell mapping")
        .def("_get_unique_primcell_offsets",
             &LocalOrbitListGenerator::getUniquePrimitiveCellOffsets,
             "Returns a list with offsets of primitive structure that span to position of atoms in the supercell.");

    /// @todo Check which of the following members must actually be exposed.
    /// @todo Turn getters into properties if possible. (Some might require massaging in cluster_space.py.)
    py::class_<ClusterSpace>(m, "ClusterSpace", py::dynamic_attr())
        .def(py::init<std::vector<std::vector<std::string>> &,
                      const OrbitList,
                      const double,
                      const double>())
        .def("get_cluster_vector",
             [](const ClusterSpace &clusterSpace,
                const Structure &structure,
                const double fractionalPositionTolerance)
                {
                    auto cv = clusterSpace.getClusterVector(structure, fractionalPositionTolerance);
                    return py::array(cv.size(), cv.data());
                },
             R"pbdoc(
             Returns the cluster vector corresponding to the input structure.
             The first element in the cluster vector will always be one (1) corresponding to
             the zerolet. The remaining elements of the cluster vector represent averages
             over orbits (symmetry equivalent clusters) of increasing order and size.

             Parameters
             ----------
             structure : _icet.Structure
                 atomic configuration
             fractional_position_tolerance : float
                 tolerance applied when comparing positions in fractional coordinates
             )pbdoc",
             py::arg("structure"),
             py::arg("fractional_position_tolerance"))
        .def("_get_orbit_list", &ClusterSpace::getOrbitList)
        .def("get_orbit", &ClusterSpace::getOrbit)
        .def_property_readonly("species_maps", &ClusterSpace::getSpeciesMaps)
        .def("get_multi_component_vectors_by_orbit", &ClusterSpace::getMultiComponentVectorsByOrbit)
        .def("get_chemical_symbols",
             &ClusterSpace::getChemicalSymbols,
             "Returns list of species associated with cluster space as chemical symbols.")
        .def("get_cutoffs", &ClusterSpace::getCutoffs)
        .def("_get_primitive_structure", &ClusterSpace::getPrimitiveStructure)
        .def("get_multi_component_vector_permutations", &ClusterSpace::getMultiComponentVectorPermutations)
        .def("get_number_of_allowed_species_by_site", &ClusterSpace::getNumberOfAllowedSpeciesBySite)
        .def("_compute_multi_component_vectors",
             &ClusterSpace::computeMultiComponentVectors,
             "Compute the multi-component vectors (internal).")
        .def("_prune_orbit_list_cpp", &ClusterSpace::pruneOrbitList)
        .def("evaluate_cluster_function",
             &ClusterSpace::evaluateClusterFunction,
             "Evaluates value of a cluster function.")
        .def("__len__", &ClusterSpace::getClusterSpaceSize);

    py::class_<ClusterExpansionCalculator>(m, "_ClusterExpansionCalculator")
        .def(py::init<const ClusterSpace &, const Structure &, const double>())
        .def("get_local_cluster_vector",
             [](ClusterExpansionCalculator &calc,
                const std::vector<int> &occupations,
                const int index,
                const std::vector<size_t> indices)
             {
                auto cv = calc.getLocalClusterVector(occupations, index, indices);
                return py::array(cv.size(), cv.data());
              },
              R"pbdoc(
              Returns a cluster vector that only considers clusters that contain the input index.

              Parameters
              ----------
              occupations : list(int)
                  the occupation vector for the supercell
              index : int
                  local index of the supercell
              ignored_indices : list(int)
                  list of indices that have already had their local energy calculated;
                  this is required to prevent double counting
              )pbdoc",
              py::arg("occupations"),
              py::arg("index"),
              py::arg("ignored_indices"));
}
