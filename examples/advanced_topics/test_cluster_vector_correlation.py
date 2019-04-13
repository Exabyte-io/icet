"""
This example demonstrates how to checks the column correlation for a set of
clustervectors and asserts that none of the columns are highly correlated
"""

# Import modules
import random
import numpy as np

from ase.db import connect
from icet import ClusterSpace


# Function for generating random structures
def generate_random_structure(atoms_prim, chemical_species, repeat=8):
    """
    Generate a random structure with atoms_prim as a base
    and fill it randomly with symbols from chemical_species
    """

    atoms = atoms_prim.copy().repeat(repeat)

    for at in atoms:
        at.symbol = random.choice(chemical_species)

    return atoms


# Function for generating cluster vectors
def generate_cv_set(n, atoms_prim, chemical_species, clusterspace, repeat=8):
    """
    Generate a set of cluster vectors from a cluster space
    """
    clustervectors = []
    for i in range(n):
        conf = generate_random_structure(atoms_prim, chemical_species, repeat)
        cv = clusterspace.get_cluster_vector(conf)
        clustervectors.append(cv)

    return clustervectors


# Function for calculating column correlations
def get_column_correlation(i, j, cv_matrix):
    """
    Returns the correlation between column i and j

    cv_matrix: numpy matrix
    """
    col_i = cv_matrix[:, i]
    col_j = cv_matrix[:, j]

    corr = np.dot(col_i, col_j) / \
        (np.linalg.norm(col_i) * np.linalg.norm(col_j))

    return corr


# Function for asserting that columns are not correlated
def assert_no_correlation(cvs, tol=0.99):
    """
    Check that no column in cvs are above tolerance
    """
    cvs_matrix = np.array(cvs)
    for i in range(len(cvs[0])):
        # Do not loop over zerolet since this is always ones
        if i == 0:
            continue
        for j in range(len(cvs[0])):
            if j <= i:
                continue
            corr = get_column_correlation(i, j, cvs_matrix)
            assert corr < tol, 'columns {} and {} were correlated with'\
                ' {}'.format(i, j, corr)


# Create a list of the chemical_species that shall be considered and set the
# cutoff distance for singlets to 2.0 Å.
chemical_species = ['Pd', 'H', 'V']
cutoffs = [2.0]
repeat = 8

# Test the correlation between columns for a set of structures in a
# previously generated database.
db = connect('PdHVac-fcc.db')
for row in db.select('id<=10'):
    atoms_row = row.toatoms()
    atoms_id = row.id
    atoms_form = row.formula
    print('Testing structure: {} (id={}) with cutoffs {}'.format(atoms_form,
                                                                 atoms_id,
                                                                 cutoffs))
    atoms_row.wrap()  # Wrap all atoms into the unit cell
    cluster_space = ClusterSpace(atoms_row, cutoffs, chemical_species)

    cvs = generate_cv_set(20, atoms_row, chemical_species,
                          cluster_space, repeat)
    assert_no_correlation(cvs)
    print('Number of atoms: {}    Length of cluster vector: {}'.format(
        len(atoms_row.repeat(repeat)), len(cvs[0])))
