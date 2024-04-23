from mchammer.observers.base_observer import BaseObserver
from icet.tools import ConstituentStrain
from ase import Atoms


class ConstituentStrainObserver(BaseObserver):
    """This class represents a constituent strain observer.  It allows
    observation of constituent strain energy separate from the energy
    calculated by the cluster expansion.

    Parameters
    ----------
    constituent_strain
        :class:`ConstituentStrain` object.
    interval
        Observation interval. Defaults to ``None`` meaning that if the
        observer is used in a Monte Carlo simulations, then the :class:`Ensemble` object
        will determine the interval.
    """

    def __init__(self,
                 constituent_strain: ConstituentStrain,
                 interval: int = None) -> None:
        super().__init__(interval=interval, return_type=dict, tag='ConstituentStrainObserver')
        self.constituent_strain = constituent_strain

    def get_observable(self, structure: Atoms) -> dict:
        """Returns the constituent strain energy for a given atomic configuration.

        Parameters
        ----------
        structure
            Input atomic structure.
        """
        cs = self.constituent_strain.get_constituent_strain(structure.get_atomic_numbers())
        cs = {'constituent_strain_energy': cs * len(structure)}
        return cs
