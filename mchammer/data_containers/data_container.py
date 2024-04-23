""" Data container class. """

import numpy as np
from .base_data_container import BaseDataContainer
from ..data_analysis import analyze_data


class DataContainer(BaseDataContainer):
    """
    Data container for storing information concerned with
    Monte Carlo simulations performed with :program:`mchammer`.

    Parameters
    ----------
    structure
        Reference atomic structure associated with the data container.
    ensemble_parameters
        Parameters associated with the underlying ensemble.
    metadata
        Metadata associated with the data container.
    """

    def analyze_data(self, tag: str, start: int = None, max_lag: int = None) -> dict:
        """
        Returns detailed analysis of a scalar observerable.

        Parameters
        ----------
        tag
            Tag of field over which to average.
        start
            Minimum value of trial step to consider. By default the
            smallest value in the mctrial column will be used.
        max_lag
            Maximum lag between two points in data series. By default the
            largest length of the data series will be used.
            Used for computing autocorrelation.

        Raises
        ------
        ValueError
            If observable is requested that is not in data container.
        ValueError
            If observable is not scalar.
        ValueError
            If observations is not evenly spaced.

        Returns
        -------
        dict
            Calculated properties of the data including mean,
            standard_deviation, correlation_length and error_estimate
            (95% confidence).
        """

        # get data for tag
        if tag in ['trajectory', 'occupations']:
            raise ValueError('{} is not scalar'.format(tag))
        steps, data = self.get('mctrial', tag, start=start)

        # check that steps are evenly spaced
        diff = np.diff(steps)
        step_length = diff[0]
        if not np.allclose(step_length, diff):
            raise ValueError('data records must be evenly spaced.')

        summary = analyze_data(data, max_lag=max_lag)
        summary['correlation_length'] *= step_length  # in mc-trials
        return summary

    def get_average(self, tag: str, start: int = None) -> float:
        """
        Returns average of a scalar observable.

        Parameters
        ----------
        tag
            Tag of field over which to average.
        start
            Minimum value of trial step to consider. By default the
            smallest value in the mctrial column will be used.

        Raises
        ------
        ValueError
            If observable is requested that is not in data container.
        ValueError
            If observable is not scalar.
        """
        if tag in ['trajectory', 'occupations']:
            raise ValueError('{} is not scalar'.format(tag))
        data = self.get(tag, start=start)
        return np.mean(data)
