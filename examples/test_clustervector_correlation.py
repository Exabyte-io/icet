"""
This example demonstrates how to checks the column correlation for a set of
clustervectors and asserts that none of the columns are highly correlated
"""

# Start import
import random

import numpy as np

from ase.db import connect

from icetdev.cluster_space import ClusterSpace
from icetdev.structure import Structure
# End import


# Start generate_random_structure
def generateRandomStructure(atoms_prim, subelements):
    """
    Generate a random structure with atoms_prim as a base
    and fill it randomly with elements in subelements
    """

    atoms = atoms_prim.copy().repeat(8)

    for at in atoms:
        element = random.choice(subelements)
        at.symbol = element

    return atoms
# End generate_random_structure


# Start generate_cv_set
def generateCVSet(n, atoms_prim, subelements, clusterspace):
    """
    generate a set of clustervectors from clusterspace
    """
    clustervectors = []
    for i in range(n):
        conf = generateRandomStructure(atoms_prim, subelements)
        conf = Structure().from_atoms(conf)
        cv = clusterspace.get_cluster_vector(conf)
        clustervectors.append(cv)

    return clustervectors
# End generate_cv_set


# Start get_column_correlation
def getColumnCorrelation(i, j, cv_matrix):
    """
    Returns the correlation between column i and j

    cv_matrix: numpy matrix
    """
    col_i = cv_matrix[:, i]
    col_j = cv_matrix[:, j]

    corr = np.dot(col_i, col_j) / \
        (np.linalg.norm(col_i) * np.linalg.norm(col_j))

    return corr
# End get_column_correlation


# Start assert_no_correlation
def assertNoCorrelation(cvs, tol=0.99):
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
            corr = getColumnCorrelation(i, j, cvs_matrix)
            assert corr < tol, "columns {} and {} were correletated with"\
                " {}".format(i, j, corr)
# End assert_no_correlation


# Test the correlation between columns for a set of structures in a
# pregenerated database.
# Start test
highest_order = 1
db = connect("PdHVac-fcc.db")
subelements = ["Pd", "H", "V"]
for row in db.select("id<=10"):
    atoms_row = row.toatoms()
    atoms_id = row.id
    atoms_form = row.formula
    cutoffs = [1.1] * highest_order
    if atoms_row.get_pbc().all():
        print("Testing structure: {}(id={}) with cutoffs {}".format(atoms_form,
                                                                    atoms_id,
                                                                    cutoffs))
        atoms_row.wrap()
        clusterspace = ClusterSpace(atoms_row, cutoffs, subelements)

        cvs = generateCVSet(20, atoms_row, subelements, clusterspace)
        assertNoCorrelation(cvs)
        print("size of atoms {}. len of cv {}".format(
            len(atoms_row.repeat([5, 3, 2])), len(cvs[0])))
# End test
