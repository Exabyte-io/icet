"""
This script checks that all atom objects in the database can have
its cluster vector computed
"""

import random
import numpy as np
from ase.db import connect
from icet import ClusterSpace


def generate_mixed_structure(atoms_prim, chemical_symbols):
    """
    Generate a supercell structure based on the input structure and populate it
    randomly with the species specified.
    """
    repeat = [1] * 3
    for i, pbc in enumerate(atoms_prim.pbc):
        if pbc:
            repeat[i] = 5

    atoms = atoms_prim.copy().repeat(repeat)
    for at in atoms:
        element = random.choice(chemical_symbols)
        at.symbol = element

    return atoms


def generate_cluster_vector_set(n, atoms_prim, chemical_symbols,
                                cluster_space):
    """
    Generate a set of cluster vectors from cluster space.
    """
    cluster_vectors = []
    for i in range(n):
        atoms = generate_mixed_structure(atoms_prim, chemical_symbols)
        cv = cluster_space.get_cluster_vector(atoms)
        cluster_vectors.append(cv)

    return cluster_vectors


def assert_decorrelation(matrix, tolerance=0.99):
    """
    Confirm that the correlation between any two columns of the input matrix
    does not exceed the tolerance specified.

    Parameters
    ----------
    matrix : list of lists/NumPy array
        input matrix
    tolerance : float
        the correlation of any two columns must be lower than this value
    """
    A = np.array(matrix)
    for i in range(len(matrix[0])):
        if i == 0:  # skip zerolets (always one)
            continue
        col_i = matrix[:, i]
        for j in range(len(matrix[0])):
            if j <= i:
                continue
            col_j = A[:, j]
            corr = np.dot(col_i, col_j)
            corr /= np.linalg.norm(col_i) * np.linalg.norm(col_j)
            msg = 'columns {} and {} correlated <i|j>= {}'.format(i, j, corr)
            assert corr < tolerance, msg


db = connect('structures_for_testing.db')
chemical_symbols = ['H', 'He', 'Pb']
for row in db.select():
    atoms_row = row.toatoms()
    if not all(atoms_row.pbc):
        continue
    atoms_tag = row.tag
    cutoffs = [1.4] * 3
    if len(atoms_row) == 0:
        continue
    atoms_row.wrap()
    cluster_space = ClusterSpace(atoms_row, cutoffs, chemical_symbols)
    cvs = generate_cluster_vector_set(5, atoms_row, chemical_symbols, cluster_space)
