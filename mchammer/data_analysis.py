from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def analyze_data(data: np.ndarray, max_lag: int = None) -> dict:
    """Carries out an extensive analysis of the data series and returns a
    dictionary containing the mean, standard deviation,
    correlation length and a 95% error estimate.

    Parameters
    ----------
    data
        Data series for which to compute autocorrelation function.
    max_lag
        Maximum lag between two data points used for computing autocorrelation.
    """
    summary = dict(mean=data.mean(),
                   std=data.std())
    acf = get_autocorrelation_function(data, max_lag)
    correlation_length = _estimate_correlation_length_from_acf(acf)
    if correlation_length is not None:
        error_estimate = _estimate_error(data, correlation_length, confidence=0.95)
        summary['correlation_length'] = correlation_length
        summary['error_estimate'] = error_estimate
    else:
        summary['correlation_length'] = np.nan
        summary['error_estimate'] = np.nan
    return summary


def get_autocorrelation_function(data: np.ndarray, max_lag: int = None) -> np.ndarray:
    """ Returns autocorrelation function.

    The autocorrelation function is computed using :func:`pandas.Series.autocorr
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.autocorr.html>`.

    Parameters
    ----------
    data
        Data series for which to compute autocorrelation function.
    max_lag
        Maximum lag between two data points.
    """
    if max_lag is None:
        max_lag = len(data) - 1
    if max_lag < 1 or max_lag >= len(data):
        raise ValueError('max_lag should be between 1 and len(data)-1.')
    series = pd.Series(data)
    acf = [series.autocorr(lag) for lag in range(0, max_lag)]
    return np.array(acf)


def get_correlation_length(data: np.ndarray) -> Optional[int]:
    r"""Returns estimate of the correlation length of data.

    The correlation length is taken as the first point where the
    autocorrelation functions is less than :math:`\exp(-2)`. If the
    correlation function never drops below :math:`\exp(-2)` ``np.nan`` is
    returned.

    If the correlation length cannot be computed since the
    autocorrelation function is unconverged the function returns
    ``None``.

    Parameters
    ----------
    data
        Data series for which to the compute autocorrelation function.
    """

    acf = get_autocorrelation_function(data)
    correlation_length = _estimate_correlation_length_from_acf(acf)
    if correlation_length is None:
        return None
    return correlation_length


def get_error_estimate(data: np.ndarray, confidence: float = 0.95) -> Optional[float]:
    r"""Returns estimate of standard error :math:`\mathrm{error}`
    with confidence interval via

    .. math::

       \mathrm{error} = t_\mathrm{factor} * \mathrm{std}(\mathrm{data}) / \sqrt{N_s}

    where :math:`t_\mathrm{factor}` is the factor corresponding to the confidence
    interval and :math:`N_s` is the number of independent measurements
    (with correlation taken into account).

    If the correlation length cannot be computed since the
    autocorrelation function is unconverged the function returns
    ``None``.

    Parameters
    ----------
    data
        Eata series for which to estimate the error.

    """
    correlation_length = get_correlation_length(data)
    if correlation_length is None:
        return None
    error_estimate = _estimate_error(data, correlation_length, confidence)
    return error_estimate


def _estimate_correlation_length_from_acf(acf: np.ndarray) -> Optional[int]:
    """Estimates correlation length from :attr:`acf`. Returns ``None`` if
    the autocorrelation function is uncoverged.
    """
    for i, a in enumerate(acf):
        if a < np.exp(-2):
            return i
    return None  # np.nan


def _estimate_error(data: np.ndarray,
                    correlation_length: int,
                    confidence: float) -> float:
    """ Estimates error using correlation length. """
    t_factor: float = stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    error: float = t_factor * np.std(data) / np.sqrt(len(data) / correlation_length)
    return error
