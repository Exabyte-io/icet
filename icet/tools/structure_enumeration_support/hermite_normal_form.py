"""
Handling of Hermite Normal Form matrices
"""

import numpy as np
from .smith_normal_form import SmithNormalForm


class HermiteNormalForm(object):
    """
    Hermite Normal Form matrix.
    """

    def __init__(self, H, rotations, translations, basis_shifts):
        self.H = H
        self.snf = SmithNormalForm(H)
        self.transformations = []
        self.compute_transformations(rotations, translations, basis_shifts)

    def compute_transformations(self, rotations, translations, basis_shifts,
                                tol=1e-3):
        """
        Save transformations (based on rotations) that turns the supercell
        into an equivalent supercell. Precompute these transformations,
        consisting of permutation as well as translation and basis shift, for
        later use.

        Parameters
        ----------
        rotations : list of ndarrays
        translations : list of ndarrays
        basis_shifts : ndarray
        """

        for R, T, basis_shift in zip(rotations, translations,
                                     basis_shifts):
            check = np.dot(np.dot(np.linalg.inv(self.H), R), self.H)
            check = check - np.round(check)
            if (abs(check) < tol).all():
                LRL = np.dot(self.snf.L, np.dot(R, np.linalg.inv(self.snf.L)))

                # Should be an integer matrix
                assert (abs(LRL - np.round(LRL)) < tol).all()
                LRL = np.round(LRL).astype(np.int64)
                LT = np.dot(T, self.snf.L.T)
                self.transformations.append([LRL, LT, basis_shift])


def yield_hermite_normal_forms(det, pbc):
    """
    Yield all Hermite Normal Form matrices with determinant det.

    Parameters
    ----------
    det : int
        Target determinant of HNFs
    pbc : list of bools
        Periodic boundary conditions of the primitive structure

    Yields
    ------
    ndarray
        3x3 HNF matrix
    """

    # 1D
    if sum(pbc) == 1:
        hnf = np.eye(3, dtype=int)
        for i, bc in enumerate(pbc):
            if bc:
                hnf[i, i] = det
                break
        yield hnf

    # 2D
    elif sum(pbc) == 2:
        for a in range(1, det + 1):
            if det % a == 0:
                c = det // a
                for b in range(0, c):
                    if not pbc[0]:
                        hnf = [[1, 0, 0],
                               [0, a, 0],
                               [0, b, c]]
                    elif not pbc[1]:
                        hnf = [[a, 0, 0],
                               [0, 1, 0],
                               [b, 0, c]]
                    else:
                        hnf = [[a, 0, 0],
                               [b, c, 0],
                               [0, 0, 1]]
                    yield np.array(hnf)

    # 3D
    else:
        for a in range(1, det + 1):
            if det % a == 0:
                for c in range(1, det // a + 1):
                    if det // a % c == 0:
                        f = det // (a * c)
                        for b in range(0, c):
                            for d in range(0, f):
                                for e in range(0, f):
                                    hnf = [[a, 0, 0],
                                           [b, c, 0],
                                           [d, e, f]]
                                    yield np.array(hnf)


def yield_reduced_hnfs(ncells, symmetries, pbc, tol=1e-3):
    """
    For a fixed determinant N (i.e., a number of atoms N), yield all
    Hermite Normal Forms (HNF) that are inequivalent under symmetry
    operations of the parent lattice.'

    Parameters
    ----------
    N : int
        Determinant (or, equivalently, the number of atoms) of the HNF.
    symmetries : dict of lists
        Symmetry operations of the parent lattice.
    pbc : list of bools
        Periodic boundary conditions of the primitive structure

    Yields
    ------
    list of ndarrays
        Symmetrically inequivalent HNFs with determinant N.
    """
    rotations = symmetries['rotations']
    translations = symmetries['translations']
    basis_shifts = symmetries['basis_shifts']
    hnfs = []

    for hnf in yield_hermite_normal_forms(ncells, pbc):

        # Throw away HNF:s that yield equivalent supercells
        hnf_inv = np.linalg.inv(hnf)
        duplicate = False
        for R in rotations:
            HR = np.dot(hnf_inv, R)
            for hnf_previous in hnfs:
                check = np.dot(HR, hnf_previous.H)
                check = check - np.round(check)
                if (abs(check) < tol).all():
                    duplicate = True
                    break
            if duplicate:
                break
        if duplicate:
            continue

        # If it's not a duplicate, save the hnf
        # and the supercell so that it can be compared to
        hnf = HermiteNormalForm(hnf, rotations, translations, basis_shifts)
        hnfs.append(hnf)
        yield hnf
