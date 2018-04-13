"""
Collection of tools for managing and analysisng linear models

Todo
-----
Rename this file
Consider what functionality we actually want here

"""

import numpy as np
from collections import namedtuple


ScatterData = namedtuple('ScatterData', ['target', 'predicted'])


def compute_rmse(A, x, y_target):
    """
    Computes root mean squared error for a linear model.

    Computes error between `y_predicted` and `y_target`, where
    `f_predicted = A.x`.

    A : NumPy array
        fit matrix
    x : list / NumPy vector
        parameters
    y_target : list / NumPy vector
        target vector

    Returns
    -------
    float
        root mean squared error
    """
    assert A.shape[0] == y_target.shape[0]
    if A.shape[0] == 0:
        return np.nan

    delta_y = np.dot(A, x) - y_target
    RMSE = np.sqrt(np.mean(delta_y**2))
    return RMSE


def compute_mae(A, x, y_target):
    """
    Computes mean absolute error for a linear model.

    Computes error between `y_predicted` and `y_target`, where
    `y_predicted = A.x`.

    Parameters
    ----------
    A : NumPy array
        fit matrix
    x : list / NumPy vector
        parameters
    y_target : list / NumPy vector
        target vector

    Returns
    -------
    float
        mean absolute error
    """
    assert A.shape[0] == y_target.shape[0]
    if A.shape[0] == 0:
        return np.nan

    delta_y = np.dot(A, x) - y_target
    MAE = np.mean(np.abs(delta_y))
    return MAE


def compute_correlation_matrix(A):
    """
    Computes correlation matrix for rows in input matrix.

    Naive implementation.

    Parameters
    ----------
    A : NumPy array
        fit matrix

    Returns
    -------
    array
        correlation matrix
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
