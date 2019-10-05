import numpy as np
from ase import Atom
from ase.build import (bulk,
                       make_supercell)
from icet import ClusterSpace
from icet.tools import map_structure_to_reference

# Construct reference structure and corresponding cluster space
a = 5.0
reference = bulk('Y', a=a, crystalstructure='fcc')
reference.append(Atom('O', (1 * a / 4., 1 * a / 4., 1 * a / 4.)))
reference.append(Atom('O', (3 * a / 4., 3 * a / 4., 3 * a / 4.)))
cs = ClusterSpace(reference, [5.0, 3.0], [['Y', 'Al'], ['O', 'X'], ['O', 'X']])

# Construct test structure
P = [[4, 1, -3], [1, 3, 1], [-1, 1, 3]]
structure = make_supercell(reference, P)

# Change some elements
to_delete = []
for atom in structure:
    if atom.position[2] < 10.0:
        if atom.symbol == 'Y':
            atom.symbol = 'Al'
        elif atom.symbol == 'O':
            atom.symbol = 'X'
del structure[to_delete]

# Calculate cluster vector of ideal mapped_structure
cv_ideal = cs.get_cluster_vector(structure)

# Add some strain
A = [[1.04, 0.03, 0], [0, 1, 0], [0, 0, 1]]
structure.set_cell(np.dot(structure.cell, A), scale_atoms=True)

# Remove vacancies
del structure[[atom.index for atom in structure if atom.symbol == 'X']]

# Rattle the structure
rattle = [[-0.2056, 0.177, -0.586],
          [-0.1087, -0.0637, 0.0402],
          [0.2378, 0.0254, 0.1339],
          [-0.0932, 0.1039, 0.1613],
          [-0.3034, 0.03, 0.0373],
          [0.1191, -0.4569, -0.2607],
          [0.4468, -0.1684, 0.156],
          [-0.2064, -0.2069, -0.1834],
          [-0.1518, -0.1406, 0.0796],
          [0.0735, 0.3443, 0.2036],
          [-0.1934, 0.0082, 0.1599],
          [-0.2035, -0.1698, -0.4892],
          [0.2953, 0.1704, -0.114],
          [0.1658, 0.2163, -0.2673],
          [-0.2358, 0.0391, -0.0278],
          [0.0549, -0.2883, -0.0088],
          [-0.0484, 0.2817, 0.1408],
          [0.0576, -0.0962, -0.062],
          [-0.0288, 0.2464, 0.1156],
          [-0.2742, -0.0108, -0.0102],
          [-0.0195, 0.0503, 0.0098],
          [-0.0663, -0.1356, 0.1544],
          [-0.1901, -0.4753, 0.2122],
          [0.1885, -0.1248, 0.0486],
          [-0.2931, -0.2401, -0.1282],
          [-0.163, 0.0505, 0.013],
          [0.1473, -0.0321, 0.27],
          [-0.0415, 0.068, 0.3321],
          [0.2479, -0.4068, -0.1289],
          [0.4214, 0.1869, -0.0758],
          [-0.1135, 0.1379, -0.2678],
          [0.1374, -0.2622, 0.0814],
          [-0.0238, 0.1081, 0.211],
          [0.4011, 0.0236, 0.2343],
          [-0.2432, 0.1574, -0.1606],
          [-0.116, -0.0166, 0.0354]]

structure.positions = structure.positions + rattle

# Do the mapping
mapped_structure, _ = map_structure_to_reference(structure, reference,
                                                 inert_species=['Y', 'Al'])

# Calculate cluster vector and compare to expected value
cv_mapped = cs.get_cluster_vector(mapped_structure)

assert np.allclose(cv_mapped, cv_ideal), 'Cluster vector does not match target'
