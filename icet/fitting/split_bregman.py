"""
This module implements the split-Bregman algorithm described in
T. Goldstein and S. Osher, SIAM J. Imaging Sci. 2, 323 (2009);
doi:10.1137/080725891
"""

import numpy as np
from scipy.optimize import minimize


def fit_split_bregman(A, y, mu=1e-3, lmbda=100, n_iters=1000, tol=1e-6,
                      verbose=0):
    """
    Split-Bregman algorithm described in T. Goldstein and S. Osher,
    SIAM J. Imaging Sci. 2, 323 (2009); doi:10.1137/080725891.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    mu : float
        Sparseness parameter
    lmbda : float
        Split Bregman parameter
    n_iters : int
        maximal number of split bregman iterations.
    tol : float
        tolerance for when stopping split bregman iterations.

    Returns
    ----------
    results : dict
        parameters
    """

    n_cols = A.shape[1]
    d = np.zeros(n_cols)
    b = np.zeros(n_cols)
    x = np.zeros(n_cols)

    old_norm = 0.0

    # Precompute for speed.
    AtA = np.dot(A.conj().transpose(), A)
    ftA = np.dot(y.conj().transpose(), A)
    ii = 0
    for i in range(n_iters):
        if verbose:
            print('Iteration ', i)
        args = (A, y, mu, lmbda, d, b, AtA, ftA)
        res = minimize(_objective_function, x, args, method='BFGS', options={
                       'disp': False}, jac=_objective_function_derivative)
        x = res.x

        d = _shrink(mu*x + b, 1.0/lmbda)
        b = b + mu*x - d

        new_norm = np.linalg.norm(x)
        ii = ii + 1

        if verbose:
            print('|new_norm-old_norm| = ', abs(new_norm-old_norm))
        if abs(new_norm-old_norm) < tol:
            break

        old_norm = new_norm
    else:
        print('Warning: Split Bregman ran for max iters')

    fit_results = {'parameters': x}
    return fit_results


def _objective_function(x, A, y, mu, lmbda, d, b, AtA, ftA):
    """ Objective function to be minimized.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    mu : float
        the parameter that adjusts sparseness.
    lmbda : float
        Split Bregman parameter
    d : np.ndarray
        same notation as Nelson, Hart paper
    b : np.ndarray
        same notation as Nelson, Hart paper
    AtA : np.ndarray
        sensing matrix transpose times sensing matrix.
    ftA : np.ndarray
        np.dot(y.conj().transpose(), A)
    """

    error_vector = np.dot(A, x) - y

    obj_function = 0.5*np.vdot(error_vector, error_vector)

    if obj_function.imag > 0.0:
        raise RuntimeError(
            'Objective function contains non-zero imaginary part.)')

    sparseness_correction = d - b - mu*x
    obj_function += 0.5*lmbda * \
        np.vdot(sparseness_correction, sparseness_correction)

    if obj_function.imag > 0.0:
        raise RuntimeError(
            'Objective function contains non-zero imaginary part.)')

    return obj_function


def _objective_function_derivative(x, A, y, mu, lmbda, d, b, AtA, ftA):
    """ Derivative of the objective function.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    mu : float
        the parameter that adjusts sparseness.
    lmbda : float
        Split Bregman parameter
    d : np.ndarray
        same notation as Nelson, Hart paper
    b : np.ndarray
        same notation as Nelson, Hart paper
    AtA : np.ndarray
        sensing matrix transpose times sensing matrix.
    ftA : np.ndarray
        np.dot(y.conj().transpose(), A)

    """
    ret = np.squeeze(np.dot(x[np.newaxis, :], AtA) -
                     ftA - lmbda*mu*(d - mu * x - b))
    return ret


def _shrink(y, alpha):
    """
    Shrink operator as defined in Eq. (11) (p. 5)
    in Nelson, Hart (Compressive sensing as a new
    paradigm for model building).

    Parameters
    -----------
    y : np.ndarray
    alpha : float
    """
    return np.sign(y) * np.maximum(np.abs(y) - alpha, 0.0)
