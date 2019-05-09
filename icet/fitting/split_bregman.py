"""
This module implements the split-Bregman algorithm described in
T. Goldstein and S. Osher, SIAM J. Imaging Sci. 2, 323 (2009);
doi:10.1137/080725891
"""

import numpy as np
from scipy.optimize import minimize
from typing import Any, Dict


def fit_split_bregman(A: np.ndarray,
                      y: np.ndarray,
                      mu: float = 1e-3,
                      lmbda: float = 100,
                      n_iters: int = 1000,
                      tol: float = 1e-6) -> Dict[str, Any]:
    """
    Determines the solution :math:`\\boldsymbol{x}` to the linear
    problem :math:`\\boldsymbol{A}\\boldsymbol{x}=\\boldsymbol{y}` using
    the split-Bregman algorithm described in T. Goldstein and S. Osher,
    SIAM J. Imaging Sci. 2, 323 (2009); doi:10.1137/080725891.
    The thus obtained parameters are returned in the form of a
    dictionary with a key named `parameters`

    Parameters
    ----------
    A
        fit matrix
    y
        target array
    mu
        sparseness parameter
    lmbda
        weight of additional L2-norm in split-Bregman
    n_iters
        maximal number of split-Bregman iterations
    tol
        convergence criterion iterative minimization
    """

    def _shrink(y: np.ndarray, alpha: float) -> np.ndarray:
        """
        Shrinkage operator as defined in Eq. (11) of the paper by Nelson
        et al., Phys. Rev. B 87, 035125 (2013); doi:10.1103/PhysRevB.87.035125.
        """
        return np.sign(y) * np.maximum(np.abs(y) - alpha, 0.0)

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
        args = (A, y, mu, lmbda, d, b, AtA, ftA)
        res = minimize(_objective_function, x, args, method='BFGS',
                       options={'disp': False},
                       jac=_objective_function_derivative)
        x = res.x

        d = _shrink(mu*x + b, 1.0/lmbda)
        b = b + mu*x - d

        new_norm = np.linalg.norm(x)
        ii = ii + 1

        if abs(new_norm-old_norm) < tol:
            break

        old_norm = new_norm

    fit_results = {'parameters': x}
    return fit_results


def _objective_function(x: np.ndarray, A: np.ndarray, y: np.ndarray,
                        mu: float, lmbda: float, d: np.ndarray, b: np.ndarray,
                        AtA: np.ndarray, ftA: np.ndarray) -> np.ndarray:
    """
    Returns the objective function to be minimized.

    Parameters
    -----------
    X
        fit matrix
    y
        target array
    mu
        the parameter that adjusts sparseness.
    lmbda
        Split Bregman parameter
    d
        same notation as Nelson et al. paper
    b
        same notation as Nelson et al. paper
    AtA
        sensing matrix transpose times sensing matrix.
    ftA
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


def _objective_function_derivative(x: np.ndarray,
                                   A: np.ndarray,
                                   y: np.ndarray,
                                   mu: float,
                                   lmbda: float,
                                   d: np.ndarray,
                                   b: np.ndarray,
                                   AtA: np.ndarray,
                                   ftA: np.ndarray) -> np.ndarray:
    """
    Returns the derivative of the objective function.

    Parameters
    -----------
    X
        fit matrix
    y
        target array
    mu
        the parameter that adjusts sparseness.
    lmbda
        Split Bregman parameter
    d
        same notation as Nelson, Hart paper
    b
        same notation as Nelson, Hart paper
    AtA
        sensing matrix transpose times sensing matrix.
    ftA
        np.dot(y.conj().transpose(), A)
    """
    ret = np.squeeze(np.dot(x[np.newaxis, :], AtA) -
                     ftA - lmbda*mu*(d - mu * x - b))
    return ret
