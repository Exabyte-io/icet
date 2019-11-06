try:
    from mip import Model, minimize, xsum
    from mip.constants import BINARY
except ImportError:
    raise ImportError('Python-MIP (https://python-mip.readthedocs.io/en/latest/) is required in order'
                      ' to use the GroundStateFinder.')

import numpy as np
from itertools import combinations

from ase import Atoms
from icet import ClusterExpansion
from icet.core.orbit_list import OrbitList
from icet.core.orbit import Orbit
from icet.core.lattice_site import LatticeSite
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure
from typing import List


class GroundStateFinder():

    """This class provides functionality for determining the ground states
    using a binary cluster expansion. This is efficiently achieved through
    through the use of mixed integer programming, based on the method presented
    by `Larsen et al
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.256101>`_.
    This class is, therefore, heavily reliant on the `Python-MIP package
    <https://python-mip.readthedocs.io/en/latest/>`_.


    Parameters
    ----------
    cluster_expansion : ClusterExpansion
        cluster expansion that shall be used to find the ground state
    species_to_count : str
        chemical symbol representing the species that will be counted when
        distinguishing between different configurations. Otherwise the first
        species in the list of chemical symbols on the, single, active
        sublattice is used.

    Example
    -------
    The following snippet illustrate how to determine the ground state for a
    Au-Ag alloy. Here, the parameters of the cluster
    expansion are set to emulate a simple Ising model in order to obtain an
    example that can be run without modification. In practice, one should of
    course use a proper cluster expansion::

        from ase.build import bulk
        from icet import ClusterExpansion, ClusterSpace
        from icet.extras.ground_state_finder import GroundStateFinder

        # prepare cluster expansion
        # the setup emulates a second nearest-neighbor (NN) Ising model
        # (zerolet and singlet ECIs are zero; only first and second neighbor
        # pairs are included)
        prim = bulk('Au')
        chemical_symbols = ['Ag', 'Au']
        cs = ClusterSpace(prim, cutoffs=[4.3], chemical_symbols=chemical_symbols)
        ce = ClusterExpansion(cs, [0, 0, 0.1, -0.02])

        # prepare initial configuration
        structure = prim.repeat(3)

        # set up the ground state finder and calculate the ground state energy
        gsf = GroundStateFinder(ce, species_to_count='Ag')
        ground_state = gsf.get_ground_state(structure, 5)
        print('Ground state energy:', ce.predict(ground_state))
    """

    def __init__(self,
                 cluster_expansion: ClusterExpansion,
                 species_to_count: str = None) -> None:

        # Check that there is only one active sublattice
        self._cluster_expansion = cluster_expansion
        cluster_space = self._cluster_expansion.get_cluster_space_copy()
        primitive_structure = cluster_space.primitive_structure
        sublattices = cluster_space.get_sublattices(primitive_structure)
        if len(sublattices.active_sublattices) > 1:
            raise NotImplementedError('Only binaries are implemented as of yet.')

        # Check that there are no more than two allowed species
        species = list(sublattices.active_sublattices[0].chemical_symbols)
        if len(species) > 2:
            raise NotImplementedError('Only binaries are implemented as of yet.')

        # Reorder the species if a specific species shall be counted
        if species_to_count is not None:
            if species_to_count not in species:
                raise ValueError('The specified species {} is not found on the active sublattice'
                                 ' ({})'.format(species_to_count, species))
            if species[-1] == species_to_count:
                species = list(reversed(species))
        self._species = species

        # Define cluster functions for elements
        self._id_map = {i: sym for i, sym in enumerate(reversed(self._species))}

        # Generate orbit list
        primitive_structure.set_chemical_symbols(
            [els[0] for els in cluster_space.chemical_symbols])
        cutoffs = cluster_space.cutoffs
        self._orbit_list = OrbitList(primitive_structure, cutoffs)

        # Transform the ECIs
        self._transform_ECIs(primitive_structure)

    def _create_cluster_maps(self, structure: Atoms) -> None:
        """
        Create maps that include information regarding which sites and orbits
        are associated with each cluster as well as the number of clusters per
        orbit

        Parameters
        ----------
        structure
            atomic configuration
        """
        # Generate full orbit list
        lolg = LocalOrbitListGenerator(self._orbit_list,
                                       Structure.from_atoms(structure))
        full_orbit_list = lolg.generate_full_orbit_list()

        # Create maps of atoms and orbits for all clusters
        cluster_to_sites_map = []
        cluster_to_orbit_map = []
        orbit_counter = 0
        for i in range(len(full_orbit_list)):
            allowed_orbit = False
            equivalent_clusters = full_orbit_list.get_orbit(
                i).get_equivalent_sites()
            allowed_cluster = True
            for cluster in equivalent_clusters:
                cluster_sites = []
                for site in cluster:
                    if structure[site.index].symbol not in self._species:
                        allowed_cluster = False
                        break
                    cluster_sites.append(site.index)
                if allowed_cluster:
                    allowed_orbit = True
                    cluster_to_sites_map.append(cluster_sites)
                    cluster_to_orbit_map.append(orbit_counter)
            if allowed_orbit:
                orbit_counter += 1

        # calculate the number of clusters per orbit
        nclusters_per_orbit = [cluster_to_orbit_map.count(
            i) for i in range(cluster_to_orbit_map[-1] + 1)]
        nclusters_per_orbit = [1] + nclusters_per_orbit

        self._cluster_to_sites_map = cluster_to_sites_map
        self._cluster_to_orbit_map = cluster_to_orbit_map
        self._nclusters_per_orbit = nclusters_per_orbit

    def _get_active_orbit_indices(self,  structure: Atoms) -> List[int]:
        """
        Generate a list with the indices of all active orbits

        Parameters
        ----------
        structure
            atomic configuration
        """
        # Generate full orbit list
        lolg = LocalOrbitListGenerator(self._orbit_list,
                                       Structure.from_atoms(structure))
        full_orbit_list = lolg.generate_full_orbit_list()

        # Determine the active orbits
        active_orbit_indices = []
        for i in range(len(full_orbit_list)):
            equivalent_clusters = full_orbit_list.get_orbit(
                i).get_equivalent_sites()
            if all(structure[site.index].symbol in self._species
                   for cluster in equivalent_clusters for site in cluster):
                active_orbit_indices.append(i)

        return active_orbit_indices

    def _get_transformation_matrix(self, structure: Atoms) -> np.ndarray:
        """
        Determine the matrix that transforms the cluster functions in the form
        of spin variables, (:math:`\\sigma_i\\in\\{-1,1\\}`), to their binary
        equivalents, (:math:`x_i\\in\\{0,1\\}`). The form is obtained by
        performing the substitution (:math:`\\sigma_i=1-2x_i`) in the
        expression for cluster expansion of the total energy
        Parameters
        ----------
        structure
            atomic configuration
        """
        # Generate full orbit list
        lolg = LocalOrbitListGenerator(self._orbit_list,
                                       Structure.from_atoms(structure))
        full_orbit_list = lolg.generate_full_orbit_list()

        # Determine the number of active orbits
        active_orbit_indices = self._get_active_orbit_indices(structure)

        # Go through all clusters associated with each active orbit and
        # determine its contribution to each orbit
        transformation = np.zeros((len(active_orbit_indices) + 1,
                                   len(active_orbit_indices) + 1))
        transformation[0, 0] = 1.0
        for i, orb_index in enumerate(active_orbit_indices, 1):
            orbit = full_orbit_list.get_orbit(orb_index)
            rep_sites = orbit.get_representative_sites()
            # add contributions to the lower order orbits to which the
            # subclusters belong
            for sub_order in range(orbit.order + 1):
                n_terms_target = len(list(combinations(rep_sites, sub_order)))
                n_terms_actual = 0
                if sub_order == 0:
                    transformation[0, i] += 1.0
                    n_terms_actual += 1
                if sub_order == orbit.order:
                    transformation[i, i] += (-2.0) ** (sub_order)
                    n_terms_actual += 1
                else:
                    comb_sub_sites = combinations(rep_sites, sub_order)
                    for sub_sites in comb_sub_sites:
                        for j, sub_index in enumerate(active_orbit_indices, 1):
                            sub_orbit = full_orbit_list.get_orbit(sub_index)
                            if sub_orbit.order != sub_order:
                                continue
                            if is_sites_in_orbit(sub_orbit, sub_sites):
                                transformation[j, i] += (-2.0) ** (sub_order)
                                n_terms_actual += 1
                # check that the number of contributions matches the number
                # of subclusters
                if n_terms_actual != n_terms_target:
                    raise Exception("Fewer matches ({}) than expected ({}) for"
                                    " sub-cluster: {}".format(n_terms_actual,
                                                              n_terms_target,
                                                              sub_sites))

        return transformation

    def _transform_ECIs(self, structure: Atoms) -> None:
        """
        Transforms the list of ECIs, obtained using cluster functions in the
        form of of spin variables, (:math:`\\sigma_i\\in\\{-1,1\\}`), to their
        equivalents for the case of binary variables,
        (:math:`x_i\\in\\{0,1\\}`).
        ----------
        structure
            atomic configuration
        """
        ecis = self._cluster_expansion.parameters
        A = self._get_transformation_matrix(structure)

        self._transformed_parameters = np.dot(A, ecis)

    def _get_total_energy(self, cluster_instance_activities: List[int]
                          ) -> List[float]:
        """
        Calculate the total energy using the expression based on binary
        variables

        .. math::

            H({\\boldsymbol x}, {\\boldsymbol E})=E_0+
            \\sum\\limits_j\\sum\\limits_{{\\boldsymbol c}
            \\in{\\boldsymbol C}_j}E_jy_{{\\boldsymbol c}},

        where (:math:`y_{{\\boldsymbol c}}=
        \\prod\\limits_{i\\in{\\boldsymbol c}}x_i`)
        ----------
        cluster_instance_activities
            list of cluster instance activities, (:math:`y_{{\\boldsymbol c}}`)
        """

        E = [0.0 for _ in self._transformed_parameters]
        for i in range(len(cluster_instance_activities)):
            orbit = self._cluster_to_orbit_map[i]
            E[orbit + 1] += cluster_instance_activities[i]
        E[0] = 1

        E = [E[orbit] * self._transformed_parameters[orbit] / self._nclusters_per_orbit[orbit]
             for orbit in range(len(self._transformed_parameters))]

        return E

    def get_ground_state(self, structure: Atoms,
                         species_count: float,
                         verbose: int = 1) -> Atoms:
        """
        Find the ground state for a given structure and species count, which
        refers to the `count_species`, if provided when initializing the
        instance of this class, or the first species in the list of chemical
        symbols for the active sublattice.
        ----------
        structure
            atomic configuration
        species_count
            species count in the desired configuration
        verbose
            0 to disable solver messages printed on the screen, 1 to enable
        """
        self._create_cluster_maps(structure)
        # Initiate MIP model
        prob = Model("CE")

        # Set verbosity
        prob.verbose = verbose

        # Spin variables (remapped) for all atoms in the structure
        xs = []
        site_to_active_index_map = {}
        for i, sym in enumerate(structure.get_chemical_symbols()):
            if sym in self._species:
                site_to_active_index_map[i] = len(xs)
                xs.append(prob.add_var(name="atom_{}".format(i), var_type=BINARY))

        ys = []
        for i in range(len(self._cluster_to_orbit_map)):
            ys.append(prob.add_var(name="cluster_{}".format(i), var_type=BINARY))

        # The objective function is added to 'prob' first
        prob.objective = minimize(xsum(self._get_total_energy(ys)))

        # The five constraints are entered
        prob.add_constr(xsum(xs) == species_count, "Species count")

        count = 0
        for i, cluster in enumerate(self._cluster_to_sites_map):
            for atom in cluster:
                prob.add_constr(ys[i] <= xs[site_to_active_index_map[atom]],
                                "Decoration -> cluster {}".format(count))
                count += 1

        for i, cluster in enumerate(self._cluster_to_sites_map):
            prob.add_constr(ys[i] >= 1 - len(cluster) +
                            xsum(xs[site_to_active_index_map[atom]] for atom in cluster),
                            "Decoration -> cluster {}".format(count))
            count += 1

        # The problem is solved using python-MIPs choice of solver, which is Girubi, if available,
        # and COIN-OR Branch-and-Cut, otherwise
        status = prob.optimize()

        # The status of the solution is printed to the screen
        if str(status) != 'OptimizationStatus.OPTIMAL':
            raise RuntimeError('No optimal solution found.')

        # Each of the variables is printed with it's resolved optimum value
        gs = structure.copy()

        for v in prob.vars:
            if 'atom' in v.name:
                # print(v.name)
                index = int(v.name.split('_')[-1])
                gs[index].symbol = self._id_map[int(v.x)]

        assert abs(prob.objective_value + prob.objective_const
                   - self._cluster_expansion.predict(gs)) < 1e-6

        return gs


def is_sites_in_orbit(orbit: Orbit, sites: List[LatticeSite]) -> bool:
    """
    Check if the list of lattice sites is found among the equivalent sites for
    the orbit
    ----------
    orbit
        orbit
    sites
        list of lattice sites
    """
    if orbit.order == len(sites):
        equivalent_sites = orbit.get_equivalent_sites()
        if set(sites) in [set(es) for es in equivalent_sites]:
            return True
        sites_indices = [s.index for s in sites]
        for orbit_sites in equivalent_sites:
            orbit_sites_indices = [s.index for s in orbit_sites]
            if set(sites_indices) != set(orbit_sites_indices):
                continue
            relative_offsets = []
            matched_sites = []
            for i in range(len(sites)):
                for j in range(len(orbit_sites)):
                    if sites[i].index == orbit_sites[j].index and j not in matched_sites:
                        relative_offsets.append(
                            sites[i].unitcell_offset - orbit_sites[j].unitcell_offset)
                        matched_sites.append(j)
                        break
            if all(np.array_equal(ro, relative_offsets[0]) for ro in relative_offsets):
                return True
    return False
