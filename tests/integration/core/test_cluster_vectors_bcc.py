"""
This scripts checks the computation of cluster vectors for three body centerd
cubic based structures.
"""

import numpy as np
from ase.build import bulk, make_supercell
from icet import ClusterSpace

cutoffs = [8.0, 7.0]
species = ['W', 'Ti']
prototype = bulk('W')
cs = ClusterSpace(prototype, cutoffs, species)

# structure #1
cv = cs.get_cluster_vector(prototype)
cv_target = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
assert np.all(np.abs(cv_target - cv) < 1e-6)

# structure #2
conf = make_supercell(prototype, [[2, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 2]])
conf[0].symbol = 'Ti'
conf[1].symbol = 'Ti'
cv = cs.get_cluster_vector(conf)
cv_target = np.array([1.0, 0.0, 0.0, -0.3333333333333333,
                      -0.3333333333333333, 0.0, 1.0, 1.0,
                      0.0, -0.3333333333333333,
                      -0.3333333333333333, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
assert np.all(np.abs(cv_target - cv) < 1e-6)

# structure #3
conf = make_supercell(prototype, [[1,  0, 1],
                                  [0,  1, 1],
                                  [0, -1, 3]])
conf[0].symbol = 'Ti'
conf[1].symbol = 'Ti'
conf[2].symbol = 'Ti'
cv = cs.get_cluster_vector(conf)
cv_target = np.array([1.0, -0.5, 0.0, 0.6666666666666666,
                      0.3333333333333333, 0.0, 0.0, 1.0,
                      0.0, 0.6666666666666666, 0.3333333333333333,
                      -0.16666666666666666, 0.5, 0.16666666666666666,
                      -0.5, -0.16666666666666666, -0.5,
                      0.16666666666666666, -0.5, -0.5,
                      0.16666666666666666, 0.5, -0.16666666666666666,
                      -0.5, 0.16666666666666666, 0.16666666666666666,
                      -0.5, 0.5, -0.16666666666666666, 0.5,
                      0.16666666666666666, -0.16666666666666666,
                      -0.5, 0.5, 0.16666666666666666, -0.5])
assert np.all(np.abs(cv_target - cv) < 1e-6)
