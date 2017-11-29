'''
Handling of Smith Normal Form matrices
'''

import numpy as np


class SmithNormalForm(object):
    '''
    Smith Normal Form matrix.
    '''

    def __init__(self, H):
        self.compute_snf(H)
        self.S = tuple([self.S_matrix[i, i] for i in range(3)])
        self.ncells = self.S[0] * self.S[1] * self.S[2]
        self.group_order = None
        self.hnfs = []

        # Help list for permuting labelings
        blocks = [self.ncells // self.S[0]]
        for i in range(1, 3):
            blocks.append(blocks[-1] // self.S[i])
        self.blocks = blocks

    def compute_snf(self, H, tol=1e-3):
        '''
        Compute Smith Normal Form for 3x3 matrix. Note that H = L*S*R.

        Parameters
        ----------
        H : ndarray
            3x3 matrix
        '''
        A = H.copy()
        L = np.eye(3, dtype=int)
        R = np.eye(3, dtype=int)
        while True:
            # Clear upper row and leftmost column in such
            # a way that greatest common denominator ends
            # up in A[0, 0], in a standard Smith Normal Form way
            # (Euclidean algorithm for finding greatest common divisor)
            while sorted(A[0])[1] != 0 or sorted(A[:, 0])[1] != 0:
                A, R = _gcd_reduce_row(A, R, 0)
                A, L = _gcd_reduce_column(A, L, 0)

            # Do the same thing for lower 2x2 matrix
            while sorted(A[1, 1:])[0] != 0 or sorted(A[1:, 1])[0] != 0:
                A, R = _gcd_reduce_row(A, R, 1)
                A, L = _gcd_reduce_column(A, L, 1)

            # If last diagonal entry is negative,
            # make it positive
            if A[2, 2] < 0:
                A[2, 2] = -A[2, 2]
                L[2] = -L[2]

            # Check that the diagonal entry i,i divides
            # diagonal entry i+1, i+1. Otherwise,
            # add row i+1 to i and start over.
            if A[2, 2] % A[1, 1] != 0:
                A[1] = A[1] + A[2]
                L[1] = L[1] + L[2]
            elif A[1, 1] % A[0, 0] != 0:
                A[0] = A[0] + A[1]
                L[0] = L[0] + L[1]
            else:
                break
        assert (abs(np.dot(np.dot(L, H), R) - A) < tol).all()
        self.S_matrix = A
        self.L = L

    def add_hnf(self, hnf):
        '''Add HNF to SNF.

        Paramaters
        ----------
        hnf : HermiteNormalForm object
        '''
        self.hnfs.append(hnf)


    def set_group_order(self):
        '''
        Set group representation of an SNF matrix (the G matrix in HarFor08).
        '''
        group_order = []
        for i in range(self.S[0]):
            for j in range(self.S[1]):
                for k in range(self.S[2]):
                    group_order.append([i, j, k])
        self.group_order = group_order


def _switch_rows(A, i, j):
    '''
    Switch rows in matrix.

    Parameters
    ---------
    A : ndarray
        Matrix in which rows will be swapped.
    i : int
        Index of row 1 to be swapped.
    j : int
        Index of row 2 to be swapped.

    Returns
    -------
    ndarray
        Matrix with swapped rows.
    '''
    row = A[j].copy()
    A[j] = A[i]
    A[i] = row
    return A


def _switch_columns(A, i, j):
    '''
    Switch columns in matrix.

    Parameters
    ---------
    A : ndarray
        Matrix in which columns will be swapped.
    i : int
        Index of column 1 to be swapped.
    j : int
        Index of column 2 to be swapped.

    Returns
    -------
    ndarray
        Matrix with swapped columns.
    '''
    col = A[:, j].copy()
    A[:, j] = A[:, i]
    A[:, i] = col
    return A


def _gcd_reduce_row(A, R, i):
    '''
    Use column operations to make A[i, i] the greatest common
    denominator of the elements in row i and the other elements
    zero.

    Parameters
    ----------
    A : ndarray
        Matrix whose row is to be cleared.
    R : ndarray
        Matrix that should be subject to the same operations.
    i : int
        Index of row to be treated.

    Returns
    -------
    ndarray
        Treated matrix A.
    ndarray
        Matrix that has been subject to the same operations.
    '''
    for j in range(i, 3):
        if A[i, j] < 0:
            A[:, j] = -1 * A[:, j]
            R[:, j] = -1 * R[:, j]
    while np.sort(A[i, i:])[1 - i] > 0:
        max_index = np.argmax(A[i, i:]) + i
        min_index = np.argmin(A[i, i:]) + i
        if max_index == min_index:
            max_index += 1
        if A[i, min_index] == 0 and i == 0:
            if np.sort(A[i])[1] > 0:
                min_index += 1
                min_index = min_index % 3
                if min_index == max_index:
                    min_index += 1
                    min_index = min_index % 3
            if A[i, min_index] == A[i, max_index]:
                tmp = min_index
                min_index = max_index
                max_index = tmp
        A[:, max_index] = A[:, max_index] - A[:, min_index]
        R[:, max_index] = R[:, max_index] - R[:, min_index]
    max_index = np.argmax(A[i])
    A = _switch_columns(A, i, max_index)
    R = _switch_columns(R, i, max_index)
    return A, R


def _gcd_reduce_column(A, L, j):
    '''
    Use row operations to make A[i, i] the greatest common
    denominator of the elements in column i and the other elements
    zero.

    Parameters
    ----------
    A : ndarray
        Matrix whose column is to be cleared.
    L : ndarray
        Matrix that should be subject to the same operations.
    i : int
        Index of column to be treated.

    Returns
    -------
    ndarray
        Treated matrix A.
    ndarray
        Matrix that has been subject to the same operations.
    '''
    for i in range(j, 3):
        if A[i, j] < 0:
            A[i] = -1 * A[i]
            L[i] = -1 * L[i]
    while np.sort(A[j:, j])[1 - j] > 0:
        max_index = np.argmax(A[j:, j]) + j
        min_index = np.argmin(A[j:, j]) + j
        if max_index == min_index:
            max_index += 1
        if A[min_index, j] == 0 and j == 0:
            if np.sort(A[:, j])[1] > 0:
                min_index += 1
                min_index = min_index % 3
                if min_index == max_index:
                    min_index += 1
                    min_index = min_index % 3
            if A[min_index, j] == A[max_index, j]:
                tmp = min_index
                min_index = max_index
                max_index = tmp
        A[max_index] = A[max_index] - A[min_index]
        L[max_index] = L[max_index] - L[min_index]
    max_index = np.argmax(A[:, j])
    A = _switch_rows(A, j, max_index)
    L = _switch_rows(L, j, max_index)
    return A, L


def get_unique_snfs(hnfs):
    '''
    For a list of Hermite Normal Forms, obtain the set of unique Smith Normal
    Forms.

    Paramters
    ---------
    hnfs : list of HermiteNormalForm objects

    Returns
    -------
    list of SmithNormalForm objects
        The unique Smith Normal Form matrices.
    '''
    snfs = []
    for hnf in hnfs:
        # Check whether the snf is new or already encountered
        snf = hnf.snf
        snf_is_new = True
        for snf_comp in snfs:
            if snf_comp.S == snf.S:
                snf_is_new = False
                snf_comp.add_hnf(hnf)
                break
        if snf_is_new:
            snf.set_group_order()
            snf.add_hnf(hnf)
            snfs.append(snf)
    return snfs
