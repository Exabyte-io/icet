from collections import OrderedDict
from typing import List

import numpy as np

from ase import Atoms
from icet import ClusterSpace
from icet.core.sublattices import Sublattices
from mchammer.calculators.base_calculator import BaseCalculator


class TargetVectorCalculator(BaseCalculator):
    """
    A ``TargetVectorCalculator`` enables evaluation of the similarity
    between a structure and a target cluster vector. Such a comparison
    can be carried out in many ways, and this implementation follows the
    measure proposed by van de Walle *et al.* in Calphad **42**, 13
    (2013) [WalTiwJon13]_. Specifically, the objective function
    :math:`Q` is calculated as

    .. math::
        Q = - \\omega L + \\sum_{\\alpha}
         \\left| \\Gamma_{\\alpha} - \\Gamma^{\\text{target}}_{\\alpha}
         \\right|.

    Here, :math:`\\Gamma_{\\alpha}` are components in the cluster vector
    and :math:`\\Gamma^\\text{target}_{\\alpha}` the corresponding
    target values. The factor :math:`\\omega` is the radius of the
    largest pair cluster such that all clusters with the same or smaller
    radii have :math:`\\Gamma_{\\alpha} -
    \\Gamma^\\text{target}_{\\alpha} = 0`.

    Parameters
    ----------
    atoms
        structure for which to set up calculator
    cluster_space
        cluster space from which to build calculator
    target_vector
        vector to which any vector will be compared
    weights
        weighting of each component in cluster vector
        comparison, by default 1.0 for all components
    optimality_weight
        factor :math:`L`, a high value of which effectively
        favors a complete series of optimal cluster correlations
        for the smallest pairs (see above)
    optimality_tol
        tolerance for determining whether a perfect match
        has been achieved (used in conjunction with :math:`L`)
    name
        human-readable identifier for this calculator
    """

    def __init__(self, atoms: Atoms, cluster_space: ClusterSpace,
                 target_vector: List[float],
                 weights: List[float] = None,
                 optimality_weight: float = 1.0,
                 optimality_tol: float = 1e-5,
                 name: str = 'Target vector calculator') -> None:
        super().__init__(atoms=atoms, name=name)

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
            self.orbit_data = self.cluster_space.orbit_data
        else:
            self.optimality_weight = None
            self.optimality_tol = None
            self.orbit_data = None

        self._cluster_space = cluster_space

    def calculate_total(self, occupations: List[int]) -> float:
        """
        Calculates and returns the similarity value :math:`Q`
        of the current configuration.

        Parameters
        ----------
        occupations
            the entire occupation vector (i.e. list of atomic species)
        """
        self.atoms.set_atomic_numbers(occupations)
        cv = self.cluster_space.get_cluster_vector(self.atoms)
        return compare_cluster_vectors(cv, self.target_vector,
                                       self.orbit_data,
                                       weights=self.weights,
                                       optimality_weight=self.optimality_weight,
                                       tol=self.optimality_tol)

    def calculate_local_contribution(self, occupations: List[int]) -> float:
        """
        Not yet implemented, forwards calculation to
        calculate_total.
        """
        return self.calulate_total(occupations)

    @property
    def sublattices(self) -> Sublattices:
        """Sublattices of the calculators structure."""
        sl = self.cluster_space.get_sublattices(self.atoms)
        return sl


def compare_cluster_vectors(cv_1: np.ndarray, cv_2: np.ndarray,
                            orbit_data: OrderedDict,
                            weights: List[float] = None,
                            optimality_weight: float = 1.0,
                            tol: float = 1e-5) -> float:
    """
    Calculate a quantity that measures similarity between two cluster
    vecors.

    Parameters
    ----------
    cv_1
        cluster vector 1
    cv_2
        cluster vector 2
    orbit_data
        orbit data as obtained by ``ClusterSpace.orbit_data``
    weights
        Weight assigned to each cluster vector element
    optimality_weight
        quantity :math:`L` in [WalTiwJon13]_
        (see :class:`mchammer.calculators.TargetVectorCalculator`)
    tol
        numerical tolerance for determining whether two elements are
        exactly equal
    """
    if weights is None:
        weights = np.ones(len(cv_1))
    diff = abs(cv_1 - cv_2)
    score = np.dot(diff, weights)
    if optimality_weight:
        longest_optimal_radius = 0
        for orbit_index, d in enumerate(diff):
            orbit = orbit_data[orbit_index]
            if orbit['order'] != 2:
                continue
            if d < tol:
                longest_optimal_radius = orbit['radius']
            else:
                break
        score -= optimality_weight * longest_optimal_radius
    return score
