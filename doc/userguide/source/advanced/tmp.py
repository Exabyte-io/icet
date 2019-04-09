from icet.tools import enumerate_structures
from icet import ClusterSpace
from ase.build import bulk
from icet.tools.structure_generation import _get_sqs_cluster_vector

atoms = bulk('Au', a=4.0)
cs = ClusterSpace(atoms, [8.0, 4.0], ['Au', 'Pd'])
cr = {'Au': (0.5, 0.5)}
orbit_data = cs.orbit_data
target_cv = _get_sqs_cluster_vector(cs, {'Au': 0.5, 'Pd': 0.5})

optimality_weight = 1.0
best_score = 1e9
for structure in enumerate_structures(atoms, range(9), ['Au', 'Pd'],
                                      concentration_restrictions=cr):
    cv = cs.get_cluster_vector(structure)

    diff = abs(cv - target_cv)
    score = sum(diff)
    longest_optimal_radius = 0
    for orbit_index, d in enumerate(diff):
        orbit = orbit_data[orbit_index]
        if orbit['order'] != 2:
            continue
        if d < 1e-6:
            longest_optimal_radius = orbit['radius']
        else:
            break
        score -= optimality_weight * longest_optimal_radius
    
    if score < best_score:
        best_score = score
        best_structure = structure

print(cs.get_cluster_vector(best_structure))