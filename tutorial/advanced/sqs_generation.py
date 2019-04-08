"""
This example demonstrates how to generate special quasirandom structure.
"""

# Import modules
from ase import Atom
from ase.build import bulk
from icet import ClusterSpace
from icet.tools.structure_generation import (generate_sqs,
                                             generate_target_structure)

# Generate SQS for binary fcc, 50 % concentration
from icet.io.logging import set_log_config

set_log_config(level='INFO')
atoms = bulk('Au')
cutoffs = [8.0, 4.0]
cs = ClusterSpace(atoms, cutoffs, ['Au', 'Pd'])
target_concentrations = {'Au': 0.5, 'Pd': 0.5}
sqs = generate_sqs(cluster_space=cs,
                   max_size=8,
                   include_smaller_cells=False,
                   target_concentrations=target_concentrations)
print('Cluster vector of generated structure:', cs.get_cluster_vector(sqs))

# Generate SQS for a system with sublattices
atoms = bulk('Au', a=4.0)
atoms.append(Atom('H', position=(2.0, 2.0, 2.0)))
cutoffs = [7.0]
cs = ClusterSpace(atoms, cutoffs, [['Au', 'Pd', 'Cu'], ['H', 'V']])
target_concentrations = {'Au': 6 / 16, 'Pd': 1 / 16, 'Cu': 1 / 16,
                         'H': 2 / 16, 'V': 6 / 16}
sqs = generate_sqs(cluster_space=cs,
                   max_size=16,
                   target_concentrations=target_concentrations,
                   n_steps=50000)
print('Cluster vector of generated structure:', cs.get_cluster_vector(sqs))

# Generate structure with a specified cluster vector
atoms = bulk('Au')
cutoffs = [5.0]
cs = ClusterSpace(atoms, cutoffs, ['Au', 'Pd'])
target_cluster_vector = [1.0, 0.0] + [0.5] * (len(cs) - 2)
target_concentrations = {'Au': 0.5, 'Pd': 0.5}
sqs = generate_target_structure(cluster_space=cs,
                                max_size=8,
                                target_cluster_vector=target_cluster_vector,
                                target_concentrations=target_concentrations)
print('Cluster vector of generated structure:', cs.get_cluster_vector(sqs))
