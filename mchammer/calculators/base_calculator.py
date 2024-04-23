from abc import ABC, abstractmethod


class BaseCalculator(ABC):
    """
    Base class for calculators.

    Attributes
    ----------
    name : str
        Human-readable calculator name.
    """

    def __init__(self, name='BaseCalculator'):
        self.name = name

    @abstractmethod
    def calculate_total(self):
        pass

    @abstractmethod
    def calculate_change(self):
        pass

    def accept_change(self):
        """
        Some calculators depend on the state of the occupdations,
        in which they need to be informed if the occupations change.
        """
        pass

    @property
    def sublattices(self):
        raise NotImplementedError()
