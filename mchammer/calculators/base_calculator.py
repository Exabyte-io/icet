from abc import ABC, abstractmethod
from ase import Atoms
from typing import List


class BaseCalculator(ABC):
    """
    Base class for calculators.

    Attributes
    ----------
    name : str
        human-readable calculator name
    """

    def __init__(self, structure, name='BaseCalculator'):
        self._structure = structure.copy()
        self._structure.wrap()
        self.name = name

    @property
    def structure(self) -> Atoms:
        """ atomic structure associated with calculator """
        return self._structure

    @abstractmethod
    def calculate_total(self):
        pass

    @abstractmethod
    def calculate_local_contribution(self):
        pass

    def update_occupations(self, indices: List[int], species: List[int]):
        """Updates the occupation (species) of the associated atomic
        structure.

        Parameters
        ----------
        indices
            sites to update
        species
            new occupations (species) by atomic number
        """
        if not isinstance(indices, list) and not isinstance(species, list):
            raise TypeError('sites and species must be of type list')
        if len(indices) != len(species):
            raise ValueError('sites and species must have the same length')
        self._structure.numbers[indices] = species

    @property
    def sublattices(self):
        raise NotImplementedError()
