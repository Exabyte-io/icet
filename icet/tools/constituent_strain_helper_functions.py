from numba import njit
import numpy as np


@njit
def redlich_kister(x: float, *coeffs: float) -> float:
    """
    Evaluate Redlich-Kister polynomial
    with coefficients :attr:`coeff` at point(s) :attr:`x`.

    Parameters
    ----------
    x
        Point in interval [0, 1] where polynomial
        should be evaluated.
    coeffs
        Redlich-Kister coefficients,
        ``coeffs[0] (1 - 2x)^0 x (1 - x) + coeffs[1] * (1 - 2x)^1 x (1 - x) + ...``
    """
    y = 0.0
    for p in range(len(coeffs)):
        y += coeffs[p] * (1 - 2 * x)**p
    y *= x * (1 - x)
    return y


def redlich_kister_vector(x: np.ndarray, *coeffs: float) -> float:
    """
    Evaluate Redlich-Kister polynomial
    with coefficients :attr:`coeff` at points(s) :attr:`x`.

    Parameters
    ----------
    x
        Array of points in interval [0, 1] where polynomial
        should be evaluated.
    coeffs
        Redlich-Kister coefficients,
        ``coeffs[0] (1 - 2x)^0 x (1 - x) + coeffs[1] * (1 - 2x)^1 x (1 - x) + ...``
    """
    return np.array([redlich_kister(xi, *coeffs) for xi in x])


@njit
def _get_structure_factor(occupations: np.ndarray,
                          positions: np.ndarray,
                          kpt: np.ndarray,
                          spin_up: int) -> float:
    """
    Calculate structure factor for a k-point in a structure
    with atomic positions and occupations.

    Parameters
    ----------
    occupations
        Occupation vector as atomic numbers.
    positions
        Atomic positions.
    kpt
        k-point at which to evaluate the structure factor.
    spin_up
        Atomic number of species to treat as :math:`+1`
        (the other species will get :math:`-1`).
    """
    S = 0.0
    for i in range(len(positions)):
        position = positions[i]
        occupation = occupations[i]
        if occupation == spin_up:
            spin = 1
        else:
            spin = -1
        S += spin * np.exp(1j * 2 * np.pi * np.dot(kpt, position))
    return S / len(occupations)


@njit
def _get_partial_structure_factor(kpt: np.ndarray, position: np.ndarray, natoms: int) -> float:
    """
    Get a term in structure factor sum specific to a
    k-point and a position in real space.

    Parameters
    ----------
    kpt
        k-point to evaluate.
    position
        Position in real space.
    natoms
        Number of atoms in the system.
    """
    dS = np.exp(1j * 2 * np.pi * np.dot(kpt, position))
    return dS / natoms
