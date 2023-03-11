from typing import List

import numpy as np

from calorine.calculators import CPUNEP
from ase import Atoms
from ase.data import chemical_symbols
from icet.core.sublattices import Sublattices
from mchammer.calculators.base_calculator import BaseCalculator


class NEPCalculator(BaseCalculator):
    """A :class:`NEPCalculator` object enables the sampling of a
    configuration using neuroevolution potential (NEP) models.  It is
    commonly employed when setting up a Monte Carlo simulation, see
    :ref:`ensembles`.

    Parameters
    ----------
    model_file : str
        path to file with NEP model parameters (typically called ``nep.txt``)
    structure : ase.Atoms
        structure for which to set up the calculator
    primitive_structure
        the primitive structure the allowed species reference to
    allowed_species : List[List[str]]
        list of the allowed species on each site of the primitive structure
    name
        human-readable identifier for this calculator

    """

    def __init__(self,
                 model_file: str,
                 structure: Atoms,
                 primitive_structure: Atoms,
                 allowed_species: List[List[str]],
                 name: str = 'NEP Calculator',
                 ) -> None:
        super().__init__(name=name)

        # keep reference (!) to structure; this is necessary in order
        # to keep track of changes in positions
        self._structure = structure

        # set up and attach calculator
        calc = CPUNEP(model_file)
        self._structure.calc = calc

        # set up sublattices
        self._sublattices = Sublattices(
            allowed_species, primitive_structure, structure, fractional_position_tolerance=1e-5)

    def calculate_total(self, *, occupations: List[int]) -> float:
        """
        Calculates and returns the total property value of the current
        configuration.

        Parameters
        ----------
        occupations
            the entire occupation vector (i.e., list of atomic species)
        """
        assert len(occupations) == len(self._structure)
        self._structure.symbols = [chemical_symbols[k] for k in occupations]
        return self._structure.get_potential_energy()

    def calculate_change(self, *,
                         sites: List[int],
                         current_occupations: List[int],
                         new_site_occupations: List[int]) -> float:
        """
        Calculates and returns the sum of the contributions to the property
        due to the sites specified in `local_indices`

        Parameters
        ----------
        sites
            index of sites at which occupations will be changed
        current_occupations
            entire occupation vector (atomic numbers) before change
        new_site_occupations
            atomic numbers after change at the sites defined by `sites`
        """
        occupations = np.array(current_occupations)

        e_before = self.calculate_total(occupations=occupations)
        occupations[sites] = np.array(new_site_occupations)
        e_after = self.calculate_total(occupations=occupations)
        self._structure.symbols = [chemical_symbols[k] for k in occupations]
        return e_after - e_before

    @property
    def sublattices(self) -> Sublattices:
        """Sublattices of the calculators structure."""
        return self._sublattices
