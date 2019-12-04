"""
Collection of tools for managing and analyzing linear models.

Todo
-----
Consider what functionality we actually want here

"""

import numpy as np
from collections import namedtuple


ScatterData = namedtuple('ScatterData', ['target', 'predicted'])


def compute_correlation_matrix(A: np.ndarray) -> np.ndarray:
    """
    Returns the correlation matrix for the rows in the fit matrix.

    Notes
    -----
    Naive implementation.

    Parameters
    ----------
    A
        fit matrix
    """
    N = A.shape[0]
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            norm = np.linalg.norm(A[i, :]) * np.linalg.norm(A[j, :])
            c_ij = np.dot(A[i, :], A[j, :]) / norm
            C[i, j] = c_ij
            C[j, i] = c_ij
    return C


def estimate_loocv(A: np.ndarray, y_targ: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the approximative LOO-CV-RMSE.

    Parameters
    ----------
    A
        Matrix in OLS problem y=Ax, should be inversible
    y_targ
        Target values for y
    y_pred
        OLS obtained prediction for y

    Returns
    -------
    float
        LOO-CV-RMSE
    """
    if len(A[1, :]) > len(A[:, 1]):
        raise ValueError('Matrix is underdetermined')

    H = A.dot(np.linalg.inv(A.T.dot(A))).dot(A.T)
    e = (y_targ - y_pred) / (1 - np.diag(H))

    return np.linalg.norm(e) / np.sqrt(len(e))
