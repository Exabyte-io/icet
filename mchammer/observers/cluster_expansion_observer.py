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
    cluster_expansion : :class:`icet.ClusterExpansion` cluster expansion model
        to be used for observation
    tag : str
        human readable observer name (default: `ClusterExpansionObserver`)
    interval : int
        observation interval during the Monte Carlo simulation

    Attributes
    ----------
    tag : str
        name of observer
    interval : int
        observation interval
    """

    def __init__(self, cluster_expansion: ClusterExpansion,
                 interval: int = None,
                 tag: str = 'ClusterExpansionObserver') -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)
        self._cluster_expansion = cluster_expansion

        if interval is None:
            raise ValueError("The value of interval must be specified")

    def get_observable(self, atoms: Atoms) -> float:
        """
        Returns the value of the property from a cluster expansion model
        for a given atomic configuration.

        Parameters
        ----------
        atoms
            input atomic structure.
        """
        return self._cluster_expansion.predict(atoms)
