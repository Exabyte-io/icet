from math import inf
from typing import List, Dict

from ase import Atoms
from ase.data import chemical_symbols as periodic_table
from .. import ClusterExpansion
from ..core.orbit_list import OrbitList
from ..core.local_orbit_list_generator import LocalOrbitListGenerator
from ..core.structure import Structure
from .variable_transformation import transform_ECIs
from ..input_output.logging_tools import logger

try:
    import mip
    from mip.constants import BINARY
except ImportError:
    raise ImportError('Python-MIP '
                      '(https://python-mip.readthedocs.io/en/latest/) is '
                      'required in order to use the ground state finder.')


class GroundStateFinder:
    """
    This class provides functionality for determining the ground states
    using a binary cluster expansion. This is efficiently achieved through the
    use of mixed integer programming (MIP) as shown by Larsen *et al.* in
    `Phys. Rev. Lett. 120, 256101 (2018)
    <https://doi.org/10.1103/PhysRevLett.120.256101>`_.

    This class relies on the `Python-MIP package
    <https://python-mip.readthedocs.io>`_. Python-MIP can be used together
    with `Gurobi <https://www.gurobi.com/>`_, which is not open source
    but issues academic licenses free of charge. Pleaase note that
    Gurobi needs to be installed separately. The `GroundStateFinder` works
    also without Gurobi, but if performance is critical, Gurobi is highly
    recommended.

    Warning
    -------
    In order to be able to use Gurobi with python-mip one must ensure that
    `GUROBI_HOME` should point to the installation directory
    (``<installdir>``)::

        export GUROBI_HOME=<installdir>

    Note
    ----
    The current implementation only works for binary systems.


    Parameters
    ----------
    cluster_expansion : ClusterExpansion
        cluster expansion for which to find ground states
    structure : Atoms
        atomic configuration
    solver_name : str, optional
        'gurobi', alternatively 'grb', or 'cbc', searches for available
        solvers if not informed
    verbose : bool, optional
        whether to display solver messages on the screen
        (default: True)


    Example
    -------
    The following snippet illustrates how to determine the ground state for a
    Au-Ag alloy. Here, the parameters of the cluster
    expansion are set to emulate a simple Ising model in order to obtain an
    example that can be run without modification. In practice, one should of
    course use a proper cluster expansion::

        from ase.build import bulk
        from icet import ClusterExpansion, ClusterSpace
        from icet.tools.ground_state_finder import GroundStateFinder

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
        gsf = GroundStateFinder(ce, structure)
        ground_state = gsf.get_ground_state({'Ag': 5})
        print('Ground state energy:', ce.predict(ground_state))
    """
    def __init__(self,
                 cluster_expansion: ClusterExpansion,
                 structure: Atoms,
                 solver_name: str = None,
                 verbose: bool = True) -> None:
        # Check that there is only one active sublattice
        self._cluster_expansion = cluster_expansion
        self.structure = structure
        cluster_space = self._cluster_expansion.get_cluster_space_copy()
        primitive_structure = cluster_space.primitive_structure
        sublattices = cluster_space.get_sublattices(primitive_structure)
        if len(sublattices.active_sublattices) > 1:
            raise NotImplementedError('Currently, only one active sublattice'
                                      ' is allowed.')

        # Check that there are no more than two species on the active
        # sublattice
        species = list(sublattices.active_sublattices[0].chemical_symbols)
        if len(species) > 2:
            raise NotImplementedError('Only binaries are implemented as of '
                                      'yet.')
        self._species = species

        # Define cluster functions for elements
        species_map = cluster_space.species_maps[0]
        self._id_map = {periodic_table[n]: 1 - species_map[n]
                        for n in species_map.keys()}
        self._reverse_id_map = {}
        for key, value in self._id_map.items():
            self._reverse_id_map[value] = key

        # Generate orbit list
        primitive_structure.set_chemical_symbols(
            [els[0] for els in cluster_space.chemical_symbols])
        cutoffs = cluster_space.cutoffs
        self._orbit_list = OrbitList(primitive_structure, cutoffs)

        # Generate full orbit list
        lolg = LocalOrbitListGenerator(self._orbit_list,
                                       Structure.from_atoms(primitive_structure))
        full_orbit_list = lolg.generate_full_orbit_list()

        # Determine the number of active orbits
        active_orbit_indices = self._get_active_orbit_indices(primitive_structure)

        # Transform the ECIs
        binary_ecis = transform_ECIs(primitive_structure,
                                     full_orbit_list,
                                     active_orbit_indices,
                                     self._cluster_expansion.parameters)
        self._transformed_parameters = binary_ecis

        # Build model
        if solver_name is None:
            solver_name = ''
        self._model = self._build_model(structure, solver_name, verbose)

        # Properties that are defined when searching for a ground state
        self._optimization_status = None

    def _build_model(self, structure: Atoms, solver_name: str,
                     verbose: bool, xcount: int = -1) -> mip.Model:
        """
        Build a Python-MIP model based on the provided structure

        Parameters
        ----------
        structure
            atomic configuration
        solver_name
            'gurobi', alternatively 'grb', or 'cbc', searches for
            available solvers if not informed
        verbose
            whether to display solver messages on the screen
        xcount
            constraint for the species count
        """

        # Create cluster maps
        self._create_cluster_maps(structure)

        # Initiate MIP model
        model = mip.Model('CE', solver_name=solver_name)
        model.solver.set_mip_gap(0)   # avoid stopping prematurely
        model.solver.set_emphasis(2)  # focus on finding optimal solution
        model.preprocess = 2          # maximum preprocessing

        # Set verbosity
        model.verbose = int(verbose)

        # Spin variables (remapped) for all atoms in the structure
        xs = []
        site_to_active_index_map = {}
        for i, sym in enumerate(structure.get_chemical_symbols()):
            if sym in self._species:
                site_to_active_index_map[i] = len(xs)
                xs.append(model.add_var(name='atom_{}'.format(i),
                                        var_type=BINARY))
        self.xs = xs

        ys = []
        for i in range(len(self._cluster_to_orbit_map)):
            ys.append(model.add_var(name='cluster_{}'.format(i),
                                    var_type=BINARY))

        # The objective function is added to 'model' first
        model.objective = mip.minimize(mip.xsum(self._get_total_energy(ys)))

        # The five constraints are entered
        # TODO: don't create cluster constraints for singlets
        constraint_count = 0
        for i, cluster in enumerate(self._cluster_to_sites_map):
            orbit = self._cluster_to_orbit_map[i]
            ECI = self._transformed_parameters[orbit + 1]
            assert ECI != 0

            if len(cluster) < 2 or ECI < 0:  # no "downwards" pressure
                for atom in cluster:
                    model.add_constr(ys[i] <= xs[site_to_active_index_map[atom]],
                                     'Decoration -> cluster {}'.format(constraint_count))
                    constraint_count += 1

            if len(cluster) < 2 or ECI > 0:  # no "upwards" pressure
                model.add_constr(ys[i] >= 1 - len(cluster) +
                                 mip.xsum(xs[site_to_active_index_map[atom]]
                                 for atom in cluster),
                                 'Decoration -> cluster {}'.format(constraint_count))
                constraint_count += 1

        # Set species constraint
        model.add_constr(mip.xsum(xs) == xcount, 'Species count')

        # Update the model so that variables and constraints can be queried
        if model.solver_name.upper() in ['GRB', 'GUROBI']:
            model.solver.update()
        return model

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

        # Create maps of site indices and orbits for all clusters
        cluster_to_sites_map = []
        cluster_to_orbit_map = []
        orbit_counter = 0
        for i in range(len(full_orbit_list)):

            allowed_orbit = False
            allowed_cluster = True

            equivalent_clusters = full_orbit_list.get_orbit(
                i).get_equivalent_sites()

            # Determine the sites and the orbit associated with each cluster
            for cluster in equivalent_clusters:
                cluster_sites = []

                # Go through all sites in the cluster
                for site in cluster:

                    # Ensure that all sites in the cluster are occupied by allowed elements
                    if structure[site.index].symbol not in self._species:
                        allowed_cluster = False
                        break

                    # Add the site to the list of sites for this cluster
                    cluster_sites.append(site.index)

                if allowed_cluster:

                    # Do not include clusters for which the ECI is 0
                    ECI = self._transformed_parameters[orbit_counter + 1]
                    if ECI == 0:
                        continue

                    allowed_orbit = True

                    # Add the the list of sites and the orbit to the respective cluster maps
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

    def _get_total_energy(self, cluster_instance_activities: List[int]
                          ) -> List[float]:
        """
        Calculates the total energy using the expression based on binary
        variables

        .. math::

            H({\\boldsymbol x}, {\\boldsymbol E})=E_0+
            \\sum\\limits_j\\sum\\limits_{{\\boldsymbol c}
            \\in{\\boldsymbol C}_j}E_jy_{{\\boldsymbol c}},

        where (:math:`y_{{\\boldsymbol c}}=
        \\prod\\limits_{i\\in{\\boldsymbol c}}x_i`).

        Parameters
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

    def get_ground_state(self,
                         species_count: Dict[str, float],
                         max_seconds: float = inf,
                         threads: int = 0) -> Atoms:
        """
        Finds the ground state for a given structure and species count, which
        refers to the `count_species`, if provided when initializing the
        instance of this class, or the first species in the list of chemical
        symbols for the active sublattice.

        Parameters
        ----------
        species_count
            dictionary with count for one of the species on the active
            sublattice
        max_seconds
            maximum runtime in seconds (default: inf)
        threads
            number of threads to be used when solving the problem, given that a
            positive integer has been provided. If set to 0 the solver default
            configuration is used while -1 corresponds to all available
            processing cores.
        """
        # Check that the species_count is consistent with the cluster space
        if len(species_count) != 1:
            raise ValueError('Provide counts for one of the species on the '
                             'active sublattice ({}), '
                             'not {}!'.format(self._species,
                                              list(species_count.keys())))
        species_to_count = list(species_count.keys())[0]
        if species_to_count not in self._species:
            raise ValueError('The species {} is not present on the active '
                             'sublattice'
                             ' ({})'.format(species_to_count, self._species))
        if self._id_map[species_to_count] == 1:
            xcount = species_count[species_to_count]
        else:
            active_count = len([sym for sym in self.structure.get_chemical_symbols()
                                if sym in self._species])
            xcount = active_count - species_count[species_to_count]

        # The model is solved using python-MIPs choice of solver, which is
        # Gurobi, if available, and COIN-OR Branch-and-Cut, otherwise.
        model = self._model

        # Set the number of threads
        model.threads = threads

        # Update the species count
        # temporary hack until python-mip supports setting RHS directly:
        if model.solver_name.upper() in ['GUROBI', 'GRB']:
            # remove the old constraint and add a new one
            idx = model.solver.constr_get_index('Species count')
            model.solver.remove_constrs([idx])
            model.add_constr(mip.xsum(self.xs) == xcount, 'Species count')
        else:
            # rebuild the whole model
            self._model = model = self._build_model(self.structure,
                                                    model.solver_name,
                                                    bool(model.verbose),
                                                    xcount=xcount)

        # Optimize the model
        self._optimization_status = model.optimize(max_seconds=max_seconds)

        # The status of the solution is printed to the screen
        if str(self._optimization_status) != 'OptimizationStatus.OPTIMAL':
            logger.warning('No optimal solution found.')

        # Each of the variables is printed with it's resolved optimum value
        gs = self.structure.copy()

        for v in model.vars:
            if 'atom' in v.name:
                index = int(v.name.split('_')[-1])
                gs[index].symbol = self._reverse_id_map[int(v.x)]

        # Assert that the solution agrees with the prediction
        prediction = self._cluster_expansion.predict(gs)
        if model.solver_name.upper() in ['GUROBI', 'GRB']:
            assert abs(model.objective_value - prediction) < 1e-6
        elif model.solver_name.upper() == 'CBC':
            assert abs(model.objective_const +
                       model.objective_value - prediction) < 1e-6
        return gs

    @property
    def optimization_status(self) -> mip.OptimizationStatus:
        """Optimization status"""
        return self._optimization_status

    @property
    def model(self) -> mip.Model:
        """Python-MIP model"""
        return self._model.copy()
