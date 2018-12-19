from pulp import *
from icet import OrbitList
from icet import ClusterExpansion
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure
from icet import ClusterExpansion
import numpy as np
from numpy.linalg.linalg import LinAlgError


class GroundStateFinder():

    def __init__(self, cluster_expansion, elements, disallowed_elements=[]):
        if len(elements) > 2:
            raise NotImplementedError('Only binaries implemented as yet.')
        self.cluster_expansion = cluster_expansion
        self.elements = elements

        # Define cluster functions for elements
        self.id_map = {self.elements[0]: 0, self.elements[1]: 1}
        self.reverse_id_map = {}
        for key, value in self.id_map.items():
            self.reverse_id_map[value] = key
        self.disallowed_elements = disallowed_elements

        cluster_space = self.cluster_expansion.cluster_space
        primitive_structure = cluster_space.primitive_structure
        primitive_structure.set_chemical_symbols(
            [els[0] for els in cluster_space.chemical_symbols])
        cutoffs = cluster_space.cutoffs
        self.orbit_list = OrbitList(primitive_structure, cutoffs)

        self.transformed_parameters = None
        repeat = 1
        while repeat < 10 and self.transformed_parameters is None:
            try:
                self._transform_ECIs(primitive_structure.repeat(repeat))
            except LinAlgError:
                repeat += 1
            except RuntimeError:
                repeat += 1
        if self.transformed_parameters is None:
            raise Exception('Failed to find transformation of ECIs.')

    def _define_cluster_maps(self, structure):
        lolg = LocalOrbitListGenerator(self.orbit_list,
                                       Structure.from_atoms(structure))
        full_orbit_list = lolg.generate_full_orbit_list()

        cluster_to_atoms_map = []
        cluster_to_orbit_map = []
        orbit_counter = 0
        for i in range(len(full_orbit_list)):
            allowed_orbit = False
            equivalent_clusters = full_orbit_list.get_orbit(
                i).get_equivalent_sites()
            allowed_cluster = True
            for cluster in equivalent_clusters:
                cluster_atoms = []
                for site in cluster:
                    if structure[site.index].symbol in self.disallowed_elements:
                        allowed_cluster = False
                        break
                    cluster_atoms.append(site.index)
                if allowed_cluster:
                    allowed_orbit = True
                    cluster_to_atoms_map.append(cluster_atoms)
                    cluster_to_orbit_map.append(orbit_counter)
            if allowed_orbit:
                orbit_counter += 1
        nclusters_per_orbit = [cluster_to_orbit_map.count(
            i) for i in range(cluster_to_orbit_map[-1] + 1)]
        nclusters_per_orbit = [1] + nclusters_per_orbit
        self.cluster_to_atoms_map = cluster_to_atoms_map
        self.cluster_to_orbit_map = cluster_to_orbit_map
        self.nclusters_per_orbit = nclusters_per_orbit

    def _get_total_energy(self, ys):
        #Es = []
        # for i in range(len(parameters) - 1):
        #    E = parameters[i + 1] #/ cluster_to_orbit_map.count(i)
        #    Es.append(E)

        E = [0.0 for _ in self.transformed_parameters]
        for i in range(len(ys)):
            orbit = self.cluster_to_orbit_map[i]
            E[orbit + 1] += ys[i]
        E[0] = 1

        E = [E[orbit] * self.transformed_parameters[orbit] / self.nclusters_per_orbit[orbit]
             for orbit in range(len(self.transformed_parameters))]
        return sum(E)

    def _transform_ECIs(self, structure):
        self._define_cluster_maps(structure)
        cv_old = []
        cv_new = []
        for i in range(len(self.cluster_expansion.cluster_space)):
            syms = structure.get_chemical_symbols()
            syms = self._populate_randomly(structure.get_chemical_symbols())

            atoms_dec = structure.copy()
            atoms_dec.set_chemical_symbols(syms)

            decoration = [self.id_map.get(sym, -1) for sym in syms]

            cv = self._reconstruct_cluster_vector(decoration,
                                                  {0: 1, 1: -1})
            cv_icet = self.cluster_expansion.cluster_space.get_cluster_vector(
                atoms_dec)

            assert np.allclose(cv, cv_icet)
                
            cv_old.append(cv)

            cv = self._reconstruct_cluster_vector(decoration,
                                                  {0: 0, 1: 1})
            cv_new.append(cv)

        A = np.linalg.solve(cv_new, cv_old)
        A_n = np.dot(np.linalg.inv(cv_new), cv_old)
        if not np.allclose(A, A_n):
            raise RuntimeError()

        ecis = self.cluster_expansion.parameters
        self.transformed_parameters = np.dot(A, ecis)

    def _populate_randomly(self, chemical_symbols):
        syms = []
        for i in range(len(chemical_symbols)):
            if chemical_symbols[i] in self.elements:
                syms.append(np.random.choice(self.elements))
            else:
                syms.append(chemical_symbols[i])
        return syms

    def _reconstruct_cluster_vector(self, decoration,
                                    cluster_functions,
                                    divide_by_number=True):
        cv = [0 for i in range(self.cluster_to_orbit_map[-1] + 1)]
        cv = [1.0] + cv

        for i in range(len(self.cluster_to_atoms_map)):
            # print(decoration[cluster_to_atoms_map[i]])

            atoms = self.cluster_to_atoms_map[i]
            orbit = self.cluster_to_orbit_map[i] + 1

            prod = 1
            for atom in atoms:
                if decoration[atom] == -1:
                    continue
                prod *= cluster_functions[decoration[atom]]
            cv[orbit] += prod

        if divide_by_number:
            cv = [prod / m for prod, m in zip(cv, self.nclusters_per_orbit)]
        else:
            cv = [prod / 1 for prod, m in zip(cv, self.nclusters_per_orbit)]
        return cv

    def get_ground_state(self, supercell, count):
        self._define_cluster_maps(supercell)
        prob = LpProblem("CE", LpMinimize)

        # Spin variables (remapped) for all atoms in the structure
        xs = []
        site_to_active_index_map = {}
        for i, sym in enumerate(supercell.get_chemical_symbols()):
            if sym not in self.disallowed_elements:
                site_to_active_index_map[i] = len(xs)
                xs.append(LpVariable("atom_{}".format(i), 0, 1, LpInteger))

        ys = []
        for i in range(len(self.cluster_to_orbit_map)):
            ys.append(LpVariable("cluster_{}".format(i), 0, 1, LpInteger))

        # The objective function is added to 'prob' first
        prob += self._get_total_energy(ys), "Total energy"

        # The five constraints are entered
        prob += sum(xs) == count, "Concentration"

        count = 0
        for i, cluster in enumerate(self.cluster_to_atoms_map):
            for atom in cluster:
                prob += ys[i] <= xs[site_to_active_index_map[atom]
                                    ], "Decoration -> cluster {}".format(count)
                count += 1

        for i, cluster in enumerate(self.cluster_to_atoms_map):
            prob += ys[i] >= 1 - len(cluster) + sum(xs[site_to_active_index_map[atom]]
                                                    for atom in cluster), "Decoration -> cluster {}".format(count)
            count += 1

        # The problem is solved using PuLP's choice of Solver
        prob.solve()

        # The status of the solution is printed to the screen
        if LpStatus[prob.status] != 'Optimal':
            raise RuntimeError('No optimal solution found.')

        # Each of the variables is printed with it's resolved optimum value
        gs = supercell.copy()

        for v in prob.variables():
            if 'atom' in v.name:
                # print(v.name)
                index = int(v.name.split('_')[-1])
                gs[index].symbol = self.reverse_id_map[int(v.varValue)]

        assert abs(value(prob.objective) - ce.predict(gs)) < 1e-6

        return gs

if __name__ == "__main__":
    ce = ClusterExpansion.read('current_cluster_expansion.ce')

    # If it crashes saying
    #     assert np.allclose(cv, cv_icet)
    #   AssertionError
    # try to change order of the elements ['Si', 'Al'] -> ['Al', 'Si']
    gsf = GroundStateFinder(ce, ['Si', 'Al'], disallowed_elements=['Ba'])

    # Give a primitive structure with the right symbol
    # May be worth double checking that He is really on the Ba sites
    primitive_structure = ce.cluster_space.primitive_structure
    syms = ['Al' if atom.symbol == 'H' else 'Ba' for atom in primitive_structure]
    primitive_structure.set_chemical_symbols(syms)

    # 10 is the number of 'Al' atoms
    #(the second element in the list given to __init__)
    gs = gsf.get_ground_state(primitive_structure, 10)
    
    from ase.visualize import view
    view(gs)
