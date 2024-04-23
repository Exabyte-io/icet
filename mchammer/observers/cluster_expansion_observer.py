from ase import Atoms
from icet import ClusterExpansion
from mchammer.observers.base_observer import BaseObserver


class ClusterExpansionObserver(BaseObserver):
    """
    This class represents a cluster expansion (CE) observer.

    A CE observer allows to compute a property described by a CE along the
    trajectory sampled by a Monte Carlo (MC) simulation. In general this CE
    differs from the CE that is used to generate the trajectory. For example in
    a canonical MC simulation the latter would usually represent an energy
    (total or mixing energy) whereas the former CE(s) could map lattice
    constant or band gap.

    Parameters
    ----------
    cluster_expansion
        Cluster expansion to be used for observation.
    interval
        Observation interval. Defaults to ``None`` meaning that if the
        observer is used in a Monte Carlo simulations, then the :class:`Ensemble` object
        will determine the interval.
    """

    def __init__(self, cluster_expansion: ClusterExpansion,
                 interval: int = None) -> None:
        super().__init__(interval=interval, return_type=float, tag='ClusterExpansionObserver')
        self._cluster_expansion = cluster_expansion

    def get_observable(self, structure: Atoms) -> float:
        """
        Returns the value of the property from a cluster expansion model
        for a given atomic configuration.

        Parameters
        ----------
        structure
            Input atomic structure.
        """
        return self._cluster_expansion.predict(structure)
