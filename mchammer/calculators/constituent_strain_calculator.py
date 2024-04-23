from icet.tools import ConstituentStrain
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from typing import List, Union
import numpy as np


class ConstituentStrainCalculator(ClusterExpansionCalculator):
    """
    Calculator for handling cluster expansions with strain.

    Parameters
    ----------
    constituent_strain
        :class:`ConstituentStrain` object defining the strain energy
        properties of the system. The supercell used to
        create this object should correspond to the one
        used when running Monte Carlo simulations with this
        calculator
    cluster_expansion
        Cluster expansion based on which to set up :class:`ClusterExpansionCalculator`.
    name
        Human-readable identifier for this calculator.
    scaling
        Scaling factor applied to the property value predicted by the
        cluster expansion.
    """

    def __init__(self, constituent_strain: ConstituentStrain,
                 cluster_expansion: ClusterExpansion,
                 name: str = 'Constituent Strain Calculator',
                 scaling: Union[float, int] = None):
        self.constituent_strain = constituent_strain
        super().__init__(structure=constituent_strain.supercell,
                         cluster_expansion=cluster_expansion,
                         name=name,
                         scaling=scaling)

    def calculate_total(self, *, occupations: np.ndarray) -> float:
        """
        Calculates and returns the total property value of the current
        configuration.

        Parameters
        ----------
        occupations
            The entire occupation vector (i.e., an array of atomic numbers as integers).
        """
        e = super().calculate_total(occupations=occupations)
        e += len(occupations) * \
            self.constituent_strain.get_constituent_strain(occupations)
        return e

    def calculate_change(self, *, sites: List[int],
                         current_occupations: List[int],
                         new_site_occupations: List[int]) -> float:
        """
        Calculates and returns the sum of the contributions to the property
        due to the sites specified in :attr:`sites`.

        Parameters
        ----------
        sites
            Indices of sites at which occupations will be changed.
        current_occupations
            Entire occupation vector (atomic numbers) before change.
        new_site_occupations
            Atomic numbers after change at the sites defined by :attr:`sites`.
        """
        if len(new_site_occupations) > 1:
            raise NotImplementedError('Only single flips are currently allowed in '
                                      'conjunction with the constituent strain calculator.')
        e = super().calculate_change(sites=sites,
                                     current_occupations=current_occupations,
                                     new_site_occupations=new_site_occupations)
        de_cs = self.constituent_strain.get_constituent_strain_change(current_occupations,
                                                                      sites[0])
        e += len(current_occupations) * de_cs
        return e

    def accept_change(self):
        """Informs the :class:`ConstituentStrain` object that the most recent
        change was accepted, such that the new structure factor can be
        stored.
        """
        self.constituent_strain.accept_change()
