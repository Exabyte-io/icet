from collections import OrderedDict
from typing import List

import numpy as np

from ase import Atoms
from icet import ClusterSpace
from icet.core.sublattices import Sublattices
from mchammer.calculators.base_calculator import BaseCalculator


class TargetVectorCalculator(BaseCalculator):
    r"""
    A :class:`TargetVectorCalculator` enables evaluation of the similarity
    between a structure and a target cluster vector. Such a comparison
    can be carried out in many ways, and this implementation follows the
    measure proposed by van de Walle *et al.* in Calphad **42**, 13
    (2013) [WalTiwJon13]_. Specifically, the objective function
    :math:`Q` is calculated as

    .. math::
        Q = - \omega L + \sum_{\alpha}
        \left||\Gamma_{\alpha} - \Gamma^{\text{target}}_{\alpha}\right||.

    Here, :math:`\Gamma_{\alpha}` are components in the cluster vector
    and :math:`\Gamma^\text{target}_{\alpha}` the corresponding
    target values. The factor :math:`\omega` is the radius of the
    largest pair cluster such that all clusters with the same or smaller
    radii have :math:`\Gamma_{\alpha} -
    \Gamma^\text{target}_{\alpha} = 0`.

    Parameters
    ----------
    structure
        Structure for which to set up calculator.
    cluster_space
        Cluster space from which to build calculator.
    target_vector
        Vector to which any vector will be compared.
    weights
        Weighting of each component in cluster vector
        comparison. By default set to 1.0 for all components.
    optimality_weight
        Factor :math:`L`, a high value of which effectively
        favors a complete series of optimal cluster correlations
        for the smallest pairs (see above).
    optimality_tol
        Tolerance for determining whether a perfect match
        has been achieved (used in conjunction with :math:`L`).
    name
        Human-readable identifier for this calculator.
    """

    def __init__(self, structure: Atoms, cluster_space: ClusterSpace,
                 target_vector: List[float],
                 weights: List[float] = None,
                 optimality_weight: float = 1.0,
                 optimality_tol: float = 1e-5,
                 name: str = 'Target vector calculator') -> None:
        super().__init__(name=name)

        if len(target_vector) != len(cluster_space):
            raise ValueError('Cluster space and target vector '
                             'must have the same length')
        self.cluster_space = cluster_space
        self.target_vector = target_vector

        if weights is None:
            weights = np.array([1.0] * len(cluster_space))
        else:
            if len(weights) != len(cluster_space):
                raise ValueError('Cluster space and weights '
                                 'must have the same length')
        self.weights = np.array(weights)

        if optimality_weight is not None:
            self.optimality_weight = optimality_weight
            self.optimality_tol = optimality_tol
            self.as_list = self.cluster_space.as_list
        else:
            self.optimality_weight = None
            self.optimality_tol = None
            self.as_list = None

        self._cluster_space = cluster_space
        self._structure = structure
        self._sublattices = self._cluster_space.get_sublattices(structure)

    def calculate_total(self, occupations: List[int]) -> float:
        """
        Calculates and returns the similarity value :math:`Q`
        of the current configuration.

        Parameters
        ----------
        occupations
            The entire occupation vector (i.e., list of atomic species).
        """
        self._structure.set_atomic_numbers(occupations)
        cv = self.cluster_space.get_cluster_vector(self._structure)
        return compare_cluster_vectors(cv, self.target_vector,
                                       self.as_list,
                                       weights=self.weights,
                                       optimality_weight=self.optimality_weight,
                                       tol=self.optimality_tol)

    def calculate_change(self):
        raise NotImplementedError

    @property
    def sublattices(self) -> Sublattices:
        """ Sublattices of the calculators structure. """
        return self._sublattices


def compare_cluster_vectors(cv_1: np.ndarray, cv_2: np.ndarray,
                            as_list: List[OrderedDict],
                            weights: List[float] = None,
                            optimality_weight: float = 1.0,
                            tol: float = 1e-5) -> float:
    """
    Calculate a quantity that measures similarity between two cluster
    vecors.

    Parameters
    ----------
    cv_1
        Cluster vector 1.
    cv_2
        Cluster vector 2.
    as_list
        Orbit data as obtained by :attr:`ClusterSpace.as_list`.
    weights
        Weight assigned to each cluster vector element.
    optimality_weight
        Wuantity :math:`L` in [WalTiwJon13]_
        (see :class:`mchammer.calculators.TargetVectorCalculator`).
    tol
        Numerical tolerance for determining whether two elements are equal.
    """
    if weights is None:
        weights = np.ones(len(cv_1))
    diff = abs(cv_1 - cv_2)
    score = np.dot(diff, weights)
    if optimality_weight:
        longest_optimal_radius = 0
        for orbit_index, d in enumerate(diff):
            orbit = as_list[orbit_index]
            if orbit['order'] != 2:
                continue
            if d < tol:
                longest_optimal_radius = orbit['radius']
            else:
                break
        score -= optimality_weight * longest_optimal_radius
    return score
