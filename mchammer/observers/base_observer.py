"""
Definition of the base observer class.
"""

from abc import ABC, abstractmethod
from typing import Any
from ase import Atoms


class BaseObserver(ABC):
    """
    Base observer class.

    Parameters
    ----------
    interval
        Observation interval. Defaults to ``None`` meaning that if the
        observer is used in a Monte Carlo simulation, then the :class:`Ensemble` object
        will determine the interval.
    tag
        Human-readable tag used for identifying the observer.
    """

    def __init__(self,
                 return_type: type,
                 interval: int = None,
                 tag: str = 'BaseObserver') -> None:
        self._tag = tag
        self._interval = interval
        self._return_type = return_type

    @property
    def tag(self) -> str:
        """ Human-readable tag used for identifying the observer. """
        return self._tag

    @tag.setter
    def tag(self, tag: str) -> None:
        self._tag = tag

    @property
    def interval(self) -> int:
        """ Observation interval. """
        return self._interval

    @interval.setter
    def interval(self, interval: int) -> None:
        self._interval = interval

    @property
    def return_type(self) -> type:
        """ Data type of the observed data. """
        return self._return_type

    @abstractmethod
    def get_observable(self, structure: Atoms) -> Any:
        """
        Method used for extracting data.

        Returns
        -------
        self.return_type()

        When implementing this method use the following names for the
        following types of data:

        ASE Atoms object : `structure`.
        list of chemical species : `species`.
        icet cluster expansion : `cluster_expansion`.
        mchammer calculator : `calculator`.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """ String representation of object. """
        width = 60
        name = self.__class__.__name__
        s = [' {} '.format(name).center(width, '=')]

        fmt = '{:15} : {}'
        s += [fmt.format('return_type', self.return_type)]
        s += [fmt.format('interval', self.interval)]
        s += [fmt.format('tag', self.tag)]
        return '\n'.join(s)
