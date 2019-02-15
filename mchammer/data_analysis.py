import pandas as pd
import numpy as np
import scipy


def analyze_data(data, max_lag=None):
    """ Carries out an extensive analysis of the data series.

    Parameters
    ----------
    data : np.ndarray
        data series to compute autocorrelation function for
    max_lag : int
        maximum lag between two data points

    Returns
    -------
    dict
        calculated properties of the data including, mean, standard deviation,
        correlation length and a 95% error estimate.
    """
    acf = compute_autocorrelation_function(data, max_lag)
    correlation_length = _estimate_correlation_length_from_acf(acf)
    error_estimate = _estimate_error(data, correlation_length, confidence=0.95)
    summary = dict(mean=data.mean(),
                   std=data.std(),
                   correlation_length=correlation_length,
                   error_estimate=error_estimate)
    return summary


def compute_autocorrelation_function(data, max_lag=None):
    """ Returns autocorrelation function.

    The autocorrelation function is computed using Pandas.Series.autocorr

    Parameters
    ----------
    data : np.ndarray
        data series to compute autocorrelation function for
    max_lag : int
        maximum lag between two data points

    Returns
    -------
    numpy.ndarray
        calculated autocorrelation function
    """
    if max_lag is None:
        max_lag = len(data) - 1
    if 1 > max_lag >= len(data):
        raise ValueError('max_lag should be between 1 and len(data)-1.')
    series = pd.Series(data)
    acf = [series.autocorr(lag) for lag in range(0, max_lag)]
    return np.array(acf)


def estimate_correlation_length(data):
    """ Returns estimate of the correlation length of data.

    The correlation length is taken as the first point where the
    autocorrelation functions is less than exp(-2).

    If correlation function never goes below exp(-2) then np.nan is returned

    Parameters
    ----------
    data : np.ndarray
        data series to compute autocorrelation function for

    Returns
    -------
    int
        correlation length
    """

    acf = compute_autocorrelation_function(data)
    correlation_length = _estimate_correlation_length_from_acf(acf)
    return correlation_length


def estimate_error(data, confidence=0.95):
    """ Returns estimate of standard error with confidence interval.

    error = t_factor * std(data) / sqrt(Ns)
    where t_factor is the factor corresponding to the confidence interval
    Ns is the number of independent measurements (with correlation taken
    into account)

    Parameters
    ----------
    data : np.ndarray
        data series to to estimate error for

    Returns
    -------
    float
        error estimate
    """
    correlation_length = estimate_correlation_length(data)
    error_estimate = _estimate_error(data, correlation_length, confidence)
    return error_estimate


def _estimate_correlation_length_from_acf(acf):
    """ Estimate correlation length from acf """
    lengths = np.where(acf < np.exp(-2))[0]  # ACF < exp(-2)
    if len(lengths) == 0:
        return np.nan
    else:
        return lengths[0]


def _estimate_error(data, correlation_length, confidence):
    """ Estimate error using correlation length"""
    t_factor = scipy.stats.t.ppf((1 + confidence) / 2, len(data)-1)
    error = t_factor * np.std(data) / np.sqrt(len(data) / correlation_length)
    return error
