#include "Structure.hpp"
#include "NeighborList.hpp"
#include "ManyBodyNeighborList.hpp"
#include "Cluster.hpp"
#include "PermutationMatrix.hpp"
#include "LatticeSite.hpp"
#include "ClusterCounts.hpp"
#include "LocalOrbitListGenerator.hpp"
#include "ClusterSpace.hpp"
#include "ClusterExpansionCalculator.hpp"
#include <pybind11/pybind11.h>
#include "Symmetry.hpp"
#include "PeriodicTable.hpp"
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

        _OrbitList
        ----------
        .. autoclass:: _OrbitList
           :members:
           :undoc-members:

        PermutationMatrix
        -----------------
        .. autoclass:: PermutationMatrix
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

             Parameters
             ----------
             positions : list of vectors
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
         )pbdoc")
        .def(py::init<>())
        .def(
            py::init<const Eigen::Matrix<double, Dynamic, 3, Eigen::RowMajor> &,
                    const std::vector<std::string> &,
                    const Eigen::Matrix3d &,
                    const std::vector<bool> &,
                    double>(),
            "Initializes an icet Structure instance.",
            py::arg("positions"),
            py::arg("chemical_symbols"),
            py::arg("cell"),
            py::arg("pbc"),
            py::arg("tolerance") = 1e-5)
        // @todo Remove getter/setter, keep property.
        .def(
            "get_pbc", &Structure::getPBC,
            "Returns the periodic boundary conditions.")
        .def(
            "set_pbc", &Structure::setPBC,
            "Sets the periodic boundary conditions.")
        .def_property(
            "pbc",
            &Structure::getPBC,
            &Structure::setPBC,
            "list(int) : periodic boundary conditions")
        // @todo Remove getter/setter, keep property.
        .def(
            "get_cell", &Structure::getCell,
            "Returns the cell metric.")
        .def(
            "set_cell", &Structure::setCell,
            "Sets the cell metric.")
        .def_property(
            "cell",
            &Structure::getCell,
            &Structure::setCell,
            "list(list(float)) : cell metric")
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
             Sets the positions in Cartesian coordinates.

             Parameters
             ----------
             positions : list of NumPy arrays
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
             Sets the unique sites.

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
        .def("set_number_of_allowed_species", (void (Structure::*)(const std::vector<int> &)) &Structure::setNumberOfAllowedSpecies,
             py::arg("numbersOfAllowedSpecies"),
             R"pbdoc(
             Sets the number of allowed species on each site.

             This method allows one to specify for each site in the structure
             the number of species allowed on that site.

             Parameters
             ----------
             numbersOfAllowedSpecies : list of int
         )pbdoc")
        .def("set_number_of_allowed_species",
            (void (Structure::*)(const int)) &Structure::setNumberOfAllowedSpecies,
             py::arg("numbersOfAllowedSpecies"),
             R"pbdoc(
             Sets the number of allowed species on each site.

             This method allows one to specify for each site in the structure
             the number of species allowed on that site.

             Parameters
             ----------
             numbersOfAllowedSpecies : int
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
             Initializes a neighbor list instance.

             Parameters
             ----------
             cutoff : float
                 cutoff to be used for constructing the neighbor list
         )pbdoc")
        .def("build",
             &NeighborList::build,
             py::arg("structure"),
             R"pbdoc(
             Builds a neighbor list for the given atomic configuration.

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
             Checks if two sites are neighbors.

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

    // @todo document ManyBodyNeighborList in pybindings
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
             Initializes a cluster instance.

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
        .def_property_readonly("radius",
                               &Cluster::radius,
                               "float : the radius of the cluster")
        .def_property_readonly("order",
                               &Cluster::order,
                               "int : order of the cluster (= number of sites)")
        .def("__hash__", [](const Cluster &cluster) { return std::hash<Cluster>{}(cluster); })
        .def("__len__", &Cluster::order)
        .def("__str__", [](const Cluster &cluster) {
            std::string niceStr = "Radius: " + std::to_string(cluster.radius());
            niceStr += " Vertex-distances:";
            for (const auto d : cluster.distances())
            {
                niceStr += " " + std::to_string(d);
            }
            return niceStr;
        })
        .def(py::self < py::self)
        .def(py::self == py::self);
    ;

    // @todo document PermutationMatrix in pybindings
    py::class_<::PermutationMatrix>(m, "PermutationMatrix")
        .def(py::init<const std::vector<Vector3d> &,
                      const std::vector<Matrix3d> &>())
        .def("build", &::PermutationMatrix::build)
        .def("get_permuted_positions", &::PermutationMatrix::getPermutedPositions)
        .def("get_indexed_positions", &::PermutationMatrix::getIndexedPermutedPositions)

        ;

    py::class_<LatticeSite>(m, "LatticeSite")
        .def(py::init<const int, const Vector3d &>())
        .def("print", &LatticeSite::print)
        .def_property("index",
                      &LatticeSite::index,
                      &LatticeSite::setIndex,
                      "int : site index")
        .def_property("unitcell_offset",
                      &LatticeSite::unitcellOffset,
                      &LatticeSite::setUnitcellOffset,
                      "list of three ints : unit cell offset (in units of the cell vectors)")
        .def(py::self < py::self)
        .def(py::self == py::self)
        .def(py::self + Eigen::Vector3d())
        .def("__hash__", [](const LatticeSite &latticeNeighbor) { return std::hash<LatticeSite>{}(latticeNeighbor); });

    // @todo document ClusterCounts in pybindings
    py::class_<ClusterCounts>(m, "ClusterCounts")
        .def(py::init<>())
        .def("count_lattice_neighbors", &ClusterCounts::countLatticeSites)
        .def("count", (void (ClusterCounts::*)(const Structure &, const std::vector<LatticeSite> &)) & ClusterCounts::count)
        .def("count", (void (ClusterCounts::*)(const Structure &, const std::vector<std::vector<LatticeSite>> &, const Cluster &, bool)) & ClusterCounts::count,
             R"pbdoc(
            Counts the vectors in latticeSites assuming these sets of sites are
            represented by the cluster `cluster`.

            Parameters
            ----------
            structure : icet Structure
               structure that will have its clusters counted
            latticeSites : list of list of LatticeSite objects
               group of sites, represented by `cluster` that will be counted
            cluster : icet Cluster
               cluster used as identification on what sites the clusters belong to
            orderIntact : bool
               if true the order of the sites will remain the same otherwise the
               vector of species being counted will be sorted

         )pbdoc")
        .def("count_orbit_list", &ClusterCounts::countOrbitList,
             R"pbdoc(
             Counts sites in the orbit list.

             Parameters
             ----------
             structure : icet Structure
             orbit_list : icet OrbitList
             order_intact : bool
                if true do not reorder clusters
                before comparison (i.e., ABC != ACB)
             permuteSites : bool
                if true the sites will be permuted
                according to the corresponding permutations in the orbit
         )pbdoc")
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
        });

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
        .def("get_equivalent_sites", &Orbit::getEquivalentSites)
        .def("get_allowed_sites_permutations", &Orbit::getAllowedSitesPermutations)
        .def("get_representative_sites", &Orbit::getRepresentativeSites)
        .def("get_equivalent_sites_permutations", &Orbit::getPermutationsOfEquivalentSites)

        .def("get_representative_cluster", &Orbit::getRepresentativeCluster,
             R"pbdoc(
        The representative cluster represents the geometrical version of what
        this orbit is.
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
        .def_property("permutations_to_representative", &Orbit::getPermutationsOfEquivalentSites, &Orbit::setEquivalentSitesPermutations,
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
        .def_property_readonly("radius", [](const Orbit &orbit) { return orbit.getRepresentativeCluster().radius(); },
                               R"pbdoc(        Returns the radius of the
        representative cluster.
        )pbdoc")
        .def_property_readonly("permuted_sites", &Orbit::getPermutedEquivalentSites,
                               R"pbdoc(Get the equivalent sites but permuted
        to representative site.)pbdoc")
        .def_property_readonly("representative_sites", &Orbit::getRepresentativeSites,
                               R"pbdoc(
        The representative sites
        is a list of lattice sites
        that are uniquely picked out
        for this orbit which can be
        used to represent and distinguish
        between orbits.
        )pbdoc")
        .def_property("equivalent_sites", &Orbit::getEquivalentSites, &Orbit::setEquivalentSites,
                      R"pbdoc(
        List of equivalent Lattice Sites
        )pbdoc")
        .def("get_sites_with_permutation", &Orbit::getSitesWithPermutation,
             R"pbdoc(Return the permuted to representative
        sites of equivalent_sites[index].)pbdoc")
        .def("get_mc_vectors", &Orbit::getMultiComponentVectors,
             R"pbdoc(
        Return the mc vectors for this orbit given the allowed components.
        The mc vectors are returned as a list of tuples

        Parameters
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
        .def("get_all_possible_mc_vector_permutations", &Orbit::getAllPossibleMultiComponentVectorPermutations,
             R"pbdoc(
        Similar to get all permutations but
        needs to be filtered through the
        number of allowed elements.

        Parameters
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
        .def_property(
            "permutations_to_representative",
            &Orbit::getPermutationsOfEquivalentSites, &Orbit::setEquivalentSitesPermutations,
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

        .def(py::self < py::self)
        .def(py::self == py::self)
        .def(py::self + Eigen::Vector3d());

    py::class_<OrbitList>(m, "_OrbitList")
        .def(py::init<>())
        .def(
            py::init<const Structure &,
                     const std::vector<std::vector<LatticeSite>> &,
                     const std::vector<NeighborList> &>(),
            R"pbdoc(
            Constructs an OrbitList object from a permutation matrix with
            LatticeSite type entries and a Structure instance.

            Parameters
            ----------
            permutation_matrix : list(list(_icet.LatticeSite))
                permutation matrix with lattice sites
            structure : _icet.Structure
                primitive atomic structure
            )pbdoc",
            py::arg("structure"),
            py::arg("permutation_matrix"),
            py::arg("neighbor_lists"))
        .def_property_readonly(
            "orbits",
            &OrbitList::getOrbits,
            "list(_icet.Orbit) : list of orbits")
        .def("get_orbit_list", &OrbitList::getOrbits,
             "Returns the list of orbits")
        .def("add_orbit", &OrbitList::addOrbit,
             "Add an Orbit object to the OrbitList")
        .def("get_number_of_nbody_clusters", &OrbitList::getNumberOfNBodyClusters,
             "Returns the number of orbits in the OrbitList")
        .def("get_orbit", &OrbitList::getOrbit,
             "Returns a copy of the orbit at the position i in the OrbitList")
        .def("_remove_inactive_orbits",
	     &OrbitList::removeInactiveOrbits)
        .def("clear", &OrbitList::clear,
             "Clears the list of orbits.")
        .def("sort", &OrbitList::sort,
             "Sorts the orbits by orbit comparison")
        .def(
	    "remove_orbit",
	    &OrbitList::removeOrbit,
	    R"pbdoc(
            Removes the orbit with the input index.

            Parameters
            ---------
            index : int
                index of the orbit to be removed
            )pbdoc")
//        .def("_find_orbit", (int (OrbitList::*)(const Cluster &) const) & OrbitList::findOrbit,
//             R"pbdoc(
//                 Returns the index of the orbit with the given representative cluster.
//
//             Parameters
//             ---------
//             cluster: icet Cluster object
//                representative cluster of an orbit
//             )pbdoc",
//             py::arg("cluster"))
        .def("_is_row_taken", &OrbitList::isRowsTaken,
            R"pbdoc(
            Returns true if rows exist in taken_rows.

            Parameters
            ----------
            taken_rows: set of tuple of ints
                Unique collection of row index
            rows: list of ints
                row indices
             )pbdoc",
             py::arg("taken_rows"),
             py::arg("rows"))
        .def("_get_sites_translated_to_unitcell", &OrbitList::getSitesTranslatedToUnitcell,
             R"pbdoc(
                Returns a set of sites where at least one site is translated inside the unit cell.

             Parameters
             ----------
             lattice_neighbors: list of icet LatticeSite objects
                set of lattice sites that might be representative for a cluster
             sort_it: bool
                If true sort translasted sites.
             )pbdoc",
             py::arg("lattice_neighbors"),
             py::arg("sort_it"))
        .def("_get_all_columns_from_sites", &OrbitList::getAllColumnsFromSites,
             R"pbdoc(
                Finds the sites in column1, extract and return all columns along with their unit cell
                translated indistinguishable sites.

             Parameters
             ----------
             sites : list of icet LatticeSite objects
                sites that define the columns that will be returned
             column1 : list of icet LatticeSite objects
                first column of permutation matrix
             permutation_matrix : list of list of LatticeSite objects
                permutation matrix
             )pbdoc",
             py::arg("sites"),
             py::arg("column1"),
             py::arg("permutation_matrix"))
        .def("get_primitive_structure", &OrbitList::getPrimitiveStructure,
             "Returns the primitive atomic structure used to construct the OrbitList instance")
        .def("__len__", &OrbitList::size,
             "Returns the total number of orbits counted in the OrbitList instance")
        .def_property_readonly("permutation_matrix", &OrbitList::getPermutationMatrix,
             "list of list of icet LatticeSite objects: permutation_matrix")
        // .def("print", &OrbitList::print, py::arg("verbosity") = 0)
        // .def("get_supercell_orbit_list", &OrbitList::getSupercellOrbitList)
        ;

    py::class_<LocalOrbitListGenerator>(m, "LocalOrbitListGenerator")
        .def(py::init<const OrbitList &, const Structure &>(),
             R"pbdoc(
             Constructs a LocalOrbitListGenerator object from an orbit list
             and a structure.

             Parameters
             ----------
             orbit_list : _icet.OrbitList
                an orbit list set up from a primitive structure
             supercell : _icet.Structure
                supercell build up from the same primitive structure used to set the input orbit list
             )pbdoc",
             py::arg("orbit_list"),
             py::arg("supercell"))
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
        .def("generate_local_orbit_list", (OrbitList(LocalOrbitListGenerator::*)(const Vector3d &)) & LocalOrbitListGenerator::getLocalOrbitList,
             R"pbdoc(
             Generates and returns the local orbit list from a specific offset of the primitive structure.

             Parameters
             ----------
             unique_offset : NumPy array
                offset of the primitive structure
             )pbdoc",
             py::arg("unique_offset"))
        .def("generate_full_orbit_list", &LocalOrbitListGenerator::getFullOrbitList,
             R"pbdoc(
             Generates and returns a local orbit list, which orbits included the equivalent sites
             of all local orbit list in the supercell.
             )pbdoc")
        .def("clear", &LocalOrbitListGenerator::clear,
             "Clears the list of offsets and primitive-to-supercell map of the LocalOrbitListGenerator object")
        .def("get_number_of_unique_offsets",
	     &LocalOrbitListGenerator::getNumberOfUniqueOffsets,
             "Returns the number of unique offsets")
        .def("_get_primitive_to_supercell_map", &LocalOrbitListGenerator::getMapFromPrimitiveToSupercell,
             "Returns the primitive to supercell mapping")
        .def("_get_unique_primcell_offsets", &LocalOrbitListGenerator::getUniquePrimitiveCellOffsets,
             "Returns a list with offsets of primitive structure that span to position of atoms in the supercell");

    /// @todo Check which of the following members must actually be exposed.
    /// @todo Turn getters into properties if possible. (Some might require massaging in cluster_space.py.)
    py::class_<ClusterSpace>(m, "ClusterSpace", py::dynamic_attr())
        .def(py::init<std::vector<std::vector<std::string>> &, const OrbitList>())
        .def("get_cluster_vector", [](const ClusterSpace &ClusterSpace, const Structure &structure) {
            auto cv = ClusterSpace.getClusterVector(structure);
            return py::array(cv.size(), cv.data());
        })
        .def("_get_orbit_list", &ClusterSpace::getOrbitList)
        .def("get_orbit", &ClusterSpace::getOrbit)
        .def_property_readonly("species_maps", &ClusterSpace::getSpeciesMaps)
        .def("get_cluster_space_info", &ClusterSpace::getClusterSpaceInfo)
        .def("get_cluster_space_size", &ClusterSpace::getClusterSpaceSize)
        .def("get_chemical_symbols", &ClusterSpace::getChemicalSymbols, "Returns list of species associated with cluster space as chemical symbols.")
        .def("get_cutoffs", &ClusterSpace::getCutoffs)
        .def("_get_primitive_structure", &ClusterSpace::getPrimitiveStructure)
        .def("get_multi_component_vector_permutations", &ClusterSpace::getMultiComponentVectorPermutations)
        .def("get_number_of_allowed_species_by_site", &ClusterSpace::getNumberOfAllowedSpeciesBySite)
        .def(
	    "_precompute_multi_component_vectors",
	    &ClusterSpace::precomputeMultiComponentVectors,
	    "Precompute the multi-component vectors (internal).")
        .def("_prune_orbit_list_cpp", &ClusterSpace::pruneOrbitList)
        .def("__len__", &ClusterSpace::getClusterSpaceSize);


    py::class_<ClusterExpansionCalculator>(m, "_ClusterExpansionCalculator")
        .def(py::init<const ClusterSpace &, const Structure &>(),
             R"pbdoc(
             Initializes a cluster expansion calculator.

             Parameters
             ----------
             cluster_space : _icet.ClusterSpace
                defines the cluster space
             structure : icet Structure object
                the supercell the calculator will operate on
        )pbdoc",
             py::arg("cluster_space"),
             py::arg("structure"))
        .def("get_local_cluster_vector", [](ClusterExpansionCalculator &ceCalc, const std::vector<int> &occupations, const int index, const std::vector<size_t> indices) {
            auto cv = ceCalc.getLocalClusterVector(occupations, index, indices);
            return py::array(cv.size(), cv.data());
        },
             R"pbdoc(
             Returns a cluster vector that only considers clusters that contain the input index.

             Parameters
             ----------
             occupations : list of int
                 the occupation vector for the supercell
             index : int
                 local index of the supercell
             ignoredIndices : list of int
                list of indices that have already had their local energy calculated;
                this is required to prevent double counting
        )pbdoc");
}
