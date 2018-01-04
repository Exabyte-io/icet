"""
This script checks the computation of cluster vectors for three HCP-based
structures.
"""

import numpy as np
from ase.build import bulk, make_supercell
from icetdev import Structure, ClusterSpace, get_singlet_info

cutoffs = [8.0, 7.0]
subelements = ['Re', 'Ti']

print('')
prototype = bulk('Re')
cs = ClusterSpace(prototype, cutoffs, subelements)

# testing info functionality
try:
    print(cs)
except:  # NOQA
    assert False, '__repr__ function fails for ClusterSpace'
try:
    print(get_singlet_info(prototype))
except:  # NOQA
    assert False, 'get_singlet_info function fails for ClusterSpace'

# structure #1
print(' structure #1')
conf = Structure.from_atoms(prototype)
cv = cs.get_cluster_vector(conf)
cv_target = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0])
assert np.all(np.abs(cv_target - cv) < 1e-6)

# structure #2
print(' structure #2')
conf = make_supercell(prototype, [[2, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 2]])
conf[0].symbol = 'Ti'
conf[1].symbol = 'Ti'
conf = Structure.from_atoms(conf)
cv = cs.get_cluster_vector(conf)
cv_target = np.array([1.0, 0.6666666666666666, 0.3888888888888889,
                      0.5555555555555556, 0.1111111111111111, 0.0,
                      0.3888888888888889, 0.2222222222222222, 0.0,
                      0.3333333333333333, 0.3888888888888889,
                      0.3333333333333333, 0.05555555555555555,
                      0.3888888888888889, 0.6666666666666666,
                      0.3888888888888889, 0.1111111111111111,
                      0.2222222222222222, 0.3888888888888889, 0.25,
                      0.6666666666666666, 0.5, 0.25, 0.0, 0.2222222222222222,
                      0.25, 0.3055555555555556, 0.2222222222222222,
                      -0.08333333333333333, 0.2638888888888889,
                      -0.1388888888888889, -0.4444444444444444, 0.25, 0.25,
                      0.2361111111111111, 0.25, -0.16666666666666666,
                      0.08333333333333333, 0.19444444444444445, 0.25,
                      0.2777777777777778, -0.08333333333333333,
                      -0.2222222222222222, -0.1111111111111111, -
                      0.06944444444444445, 0.1111111111111111,
                      0.19444444444444445, 0.25, -0.1527777777777778,
                      0.2222222222222222, 0.1388888888888889,
                      -0.05555555555555555, 0.2638888888888889,
                      -0.3333333333333333, 0.4722222222222222,
                      0.19444444444444445, 0.25, 0.1388888888888889,
                      0.20833333333333334, 0.4722222222222222,
                      0.18055555555555555, -0.4722222222222222,
                      -0.1388888888888889, 0.2222222222222222,
                      0.19444444444444445, -0.1388888888888889,
                      -0.16666666666666666, 0.19444444444444445,
                      -0.1388888888888889, -0.16666666666666666,
                      -0.1111111111111111, 0.16666666666666666,
                      0.2222222222222222, 0.19444444444444445, 0.5,
                      0.3333333333333333, -0.1111111111111111,
                      0.08333333333333333, 0.1111111111111111,
                      0.25, 0.16666666666666666, 0.1111111111111111,
                      0.2222222222222222, 0.1388888888888889,
                      0.1111111111111111, -0.08333333333333333,
                      0.16666666666666666, -0.3888888888888889,
                      0.2222222222222222, 0.19444444444444445,
                      0.16666666666666666, 0.25])
assert np.all(np.abs(cv_target - cv) < 1e-6)

# structure #3
print(' structure #3')
conf = make_supercell(prototype, [[1,  0, 1],
                                  [0,  1, 1],
                                  [0, -1, 3]])
conf[0].symbol = 'Ti'
conf[1].symbol = 'Ti'
conf[2].symbol = 'Ti'
conf = Structure.from_atoms(conf)
cv = cs.get_cluster_vector(conf)
cv_target = np.array([1.0, 0.14285714285714285, -0.5238095238095238,
                      0.5714285714285714, -0.5238095238095238,
                      0.5714285714285714, -0.5476190476190477,
                      0.7142857142857143, 0.7142857142857143,
                      0.7142857142857143, -0.5714285714285714,
                      0.7142857142857143, -0.5714285714285714,
                      -0.47619047619047616, 0.5714285714285714,
                      -0.5238095238095238, 0.7142857142857143,
                      -0.5714285714285714, -0.5952380952380952,
                      -0.21428571428571427, 0.7142857142857143,
                      0.7142857142857143, -0.25, 0.5238095238095238,
                      -0.23809523809523808, -0.2619047619047619,
                      0.5476190476190477, -0.07142857142857142,
                      -0.09523809523809523, -0.25, -0.08333333333333333,
                      -0.2857142857142857, -0.047619047619047616,
                      -0.2619047619047619, -0.05952380952380952,
                      -0.2619047619047619, 0.5476190476190477,
                      -0.09523809523809523, -0.07142857142857142,
                      0.38095238095238093, 0.40476190476190477,
                      -0.07142857142857142, 0.40476190476190477,
                      0.40476190476190477, -0.07142857142857142,
                      -0.2619047619047619, -0.08333333333333333,
                      -0.27380952380952384, -0.08333333333333333,
                      -0.08333333333333333, -0.2619047619047619,
                      -0.11904761904761904, -0.25, 0.2857142857142857,
                      -0.09523809523809523, 0.40476190476190477,
                      0.38095238095238093, -0.09523809523809523,
                      -0.09523809523809523, -0.07142857142857142,
                      -0.08333333333333333, 0.38095238095238093,
                      -0.10714285714285714, -0.21428571428571427,
                      -0.11904761904761904, 0.38095238095238093,
                      0.38095238095238093, -0.10714285714285714,
                      -0.10714285714285714, 0.19047619047619047,
                      -0.09523809523809523, -0.13095238095238096,
                      -0.09523809523809523, -0.11904761904761904,
                      0.2857142857142857, 0.2857142857142857,
                      -0.09523809523809523, -0.23809523809523808,
                      -0.05952380952380952, -0.2857142857142857,
                      0.38095238095238093, 0.38095238095238093,
                      -0.07142857142857142, -0.08333333333333333,
                      -0.09523809523809523, -0.10714285714285714,
                      0.39285714285714285, -0.2857142857142857,
                      -0.08333333333333333, -0.11904761904761904,
                      -0.09523809523809523, -0.11904761904761904])
assert np.all(np.abs(cv_target - cv) < 1e-6)
