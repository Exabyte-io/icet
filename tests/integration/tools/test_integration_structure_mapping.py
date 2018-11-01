from icet.tools import map_structure_to_reference
from icet import ClusterSpace
from ase import Atom
from ase.build import (bulk,
                       make_supercell)
import numpy as np

# Construct reference structure and corresponding cluster space
a = 5.0
reference = bulk('Y', a=a, crystalstructure='fcc')
reference.append(Atom('O', (1 * a / 4., 1 * a / 4., 1 * a / 4.)))
reference.append(Atom('O', (3 * a / 4., 3 * a / 4., 3 * a / 4.)))
cs = ClusterSpace(reference, [5.0, 3], ['Y', 'Al', 'O', 'V'])

# Construct test structure
P = [[4, 1, -3], [1, 3, 1], [-1, 1, 3]]
atoms = make_supercell(reference, P)

# Change some elements
for index in [0, 3, 5, 6, 7, 10, 12, 18, 20, 25, 26, 37, 39, 40, 46]:
    if atoms[index].symbol == 'Y':
        atoms[index].symbol = 'Al'
    elif atoms[index] == 'O':
        del atoms[index]

# Rattle the structure
rattle = [[-0.2056,  0.177, -0.586],
          [-0.1087, -0.0637,  0.0402],
          [0.2378,  0.0254,  0.1339],
          [-0.0932,  0.1039,  0.1613],
          [-0.3034,  0.03,  0.0373],
          [0.1191, -0.4569, -0.2607],
          [0.4468, -0.1684,  0.156],
          [-0.2064, -0.2069, -0.1834],
          [-0.1518, -0.1406,  0.0796],
          [0.0735,  0.3443,  0.2036],
          [-0.1934,  0.0082,  0.1599],
          [-0.2035, -0.1698, -0.4892],
          [0.2953,  0.1704, -0.114],
          [0.1658,  0.2163, -0.2673],
          [-0.2358,  0.0391, -0.0278],
          [0.0549, -0.2883, -0.0088],
          [-0.0484,  0.2817,  0.1408],
          [0.0576, -0.0962, -0.062],
          [-0.0288,  0.2464,  0.1156],
          [-0.2742, -0.0108, -0.0102],
          [-0.0195,  0.0503,  0.0098],
          [-0.0663, -0.1356,  0.1544],
          [-0.1901, -0.4753,  0.2122],
          [0.1885, -0.1248,  0.0486],
          [-0.2931, -0.2401, -0.1282],
          [-0.163,  0.0505,  0.013],
          [0.1473, -0.0321,  0.27],
          [-0.0415,  0.068,  0.3321],
          [0.2479, -0.4068, -0.1289],
          [0.4214,  0.1869, -0.0758],
          [-0.1135,  0.1379, -0.2678],
          [0.1374, -0.2622,  0.0814],
          [-0.0238,  0.1081,  0.211],
          [0.4011,  0.0236,  0.2343],
          [-0.2432,  0.1574, -0.1606],
          [-0.116, -0.0166,  0.0354],
          [-0.2305,  0.138, -0.0022],
          [-0.1694,  0.0919,  0.2108],
          [-0.3321, -0.2151, -0.4694],
          [0.1538,  0.265,  0.2094],
          [0.0191, -0.0818, -0.0003],
          [0.032, -0.094,  0.4365],
          [-0.062,  0.0829, -0.186],
          [0.24,  0.1143, -0.0972],
          [0.1433, -0.0711,  0.0604],
          [-0.1105,  0.1382,  0.3087],
          [-0.0975,  0.0003,  0.1385],
          [0.2458, -0.0425,  0.3163]]

atoms.positions = atoms.positions + rattle

# Do the mapping
mapped_structure, r_max, r_av = \
    map_structure_to_reference(atoms, reference,
                               tolerance_mapping=0.9,
                               vacancy_type='O',
                               inert_species=['Y', 'Al'])

# Calculate cluster vector and compare to expected value
cv = cs.get_cluster_vector(mapped_structure)

cv_target = np.array([1.,  0.,  0.125,  1., -1.,  0.,
                      -1., -0.,  0., -0., -0.125,  0.,
                      -0.125, -1.,  0., -1.,  1.,  0.,
                      1.,  0.,  0.,  1.,  0., -0.,
                      0., -0.125,  0.125,  1.,  1.,  0.,
                      1.,  0.,  0.,  1., -0.,  0.,
                      -0., -0.125,  0., -0.125, -1.,  0.,
                      -1.,  1.,  0.,  1.,  0.,  0.,
                      1.,  1.,  0.,  1.,  0.,  0.,
                      1.,  0.,  0.,  0.,  0.083333,  0.125,
                      1.,  1.,  0.,  1.,  0.,  0.,
                      1.,  0.,  0.,  0.,  0.,  0.,
                      0.,  0.125,  0.,  0.125,  0.,  0.,
                      0.125,  1.,  0.,  1.,  0.,  0.,
                      1.])


assert np.allclose(cv, cv_target), 'Cluster vector does not match target'
