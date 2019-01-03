"""
scikit-learn is an excellent library for training linear models and provides a
large number of useful tools.

This module provides simplified interfaces for vaiours linear model regression
methods. These methods are set up in a way that work out of the box for typical
problems in cluster expansion and force constant potential construction. This
includes slight adjustments scitkit-learn default values.

If you would like more flexibility or extended functionality or ability to
fine-tune parameters that are not included in this interface, it is of course
possible to use scikit-learn directly.
More information about the sklearn linear models can be found at
http://scikit-learn.org/stable/modules/linear_model.html
"""

import numpy as np
from ..io.logging import logger
from collections import OrderedDict
from .split_bregman import fit_split_bregman

from sklearn.linear_model import (Lasso,
                                  LinearRegression,
                                  LassoCV,
                                  ElasticNet,
                                  ElasticNetCV,
                                  BayesianRidge,
                                  ARDRegression)
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import RFE, RFECV
from sklearn.preprocessing import StandardScaler


logger = logger.getChild('fit_methods')


def fit(X, y, fit_method, standardize=True, check_condition=True, **kwargs):
    """ Wrapper function for all available fit methods.

    Parameters
    -----------
    X : np.ndarray or list(list(float))
        fit matrix
    y : np.ndarray
        target array
    fit_method : str
        method to be used for training; possible choice are
        "least-squares", "lasso", "elasticnet", "bayesian-ridge", "ardr"
    standardize : bool
        whether or not to standardize the fit matrix before fitting
    check_condition : bool
        whether or not to carry out a check of the condition number

        N.B.: This can be sligthly more time consuming for larger
        matrices.

    Returns
    ----------
    results : dict
        parameters and other pertinent information
    """

    if fit_method not in available_fit_methods:
        msg = ['Fit method not available']
        msg += ['Please choose one of the following:']
        for key in available_fit_methods:
            msg += [' * ' + key]
        raise ValueError('\n'.join(msg))

    if check_condition:
        cond = np.linalg.cond(X)
        if cond > 1e10:
            logger.warning('Condition number is large, {}'.format(cond))

    if standardize:
        ss = StandardScaler(copy=False, with_mean=False, with_std=True)
        ss.fit_transform(X)  # change in place
        results = fit_methods[fit_method](X, y, **kwargs)
        ss.inverse_transform(X)  # change in place
        ss.transform(results['parameters'].reshape(1, -1)).reshape(-1,)
    else:
        results = fit_methods[fit_method](X, y, **kwargs)
    return results


def _fit_least_squares(X, y):
    """
    Returns the least-squares solution `a` to the linear problem `Xa=y`.

    This function is a wrapper to the `linalg.lstsq` function in NumPy.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array

    Returns
    ----------
    results : dict
        parameters
    """
    results = dict()
    results['parameters'] = np.linalg.lstsq(X, y, rcond=-1)[0]
    return results


def _fit_lasso(X, y, alpha=None, fit_intercept=False, **kwargs):
    """
    Return the solution `a` to the linear problem `Xa=y` obtained by using
    the LASSO method as implemented in scitkit-learn.

    LASSO optimizes the following problem::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    If `alpha` is `None` this function will call the fit_lassoCV which attempts
    to find the optimal alpha via sklearn LassoCV class.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    alpha : float
        alpha value
    fit_intercept : bool
        center data or not, forwarded to sklearn

    Returns
    ----------
    results : dict
        parameters
    """
    if alpha is None:
        return _fit_lassoCV(X, y, fit_intercept=fit_intercept, **kwargs)
    else:
        lasso = Lasso(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        lasso.fit(X, y)
        results = dict()
        results['parameters'] = lasso.coef_
        return results


def _fit_lassoCV(X, y, alphas=None, fit_intercept=False, cv=10, n_jobs=-1,
                 **kwargs):
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by using
    the LassoCV method as implemented in scitkit-learn.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    alphas : list / array
        list of alpha values to be evaluated during regularization path
    fit_intercept : bool
        center data or not, forwarded to sklearn
    cv : int
        how many folds to carry out in cross-validation

    Returns
    -------
    results : dict
        parameters as well as
        `alpha_optimal` (alpha value that yields the lowest validation RMSE),
        `alpha_path` (all tested alpha values),
        `mse_path` (MSE for validation set for each alpha)
    """

    if alphas is None:
        alphas = np.logspace(-8, -0.3, 100)

    lassoCV = LassoCV(alphas=alphas, fit_intercept=fit_intercept, cv=cv,
                      n_jobs=n_jobs, **kwargs)
    lassoCV.fit(X, y)
    results = dict()
    results['parameters'] = lassoCV.coef_
    results['alpha_optimal'] = lassoCV.alpha_
    results['alpha_path'] = lassoCV.alphas_
    results['mse_path'] = lassoCV.mse_path_.mean(axis=1)
    return results


def _fit_elasticnet(X, y, alpha=None, fit_intercept=False, **kwargs):
    """
    Return the solution `a` to the linear problem `Xa=y` obtained by using
    the ElasticNet method as implemented in scitkit-learn.

    If `alpha` is `None` this function will call the fit_lassoCV which attempts
    to find the optimal alpha via sklearn ElasticNetCV class.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    alpha : float
        alpha value
    fit_intercept : bool
        center data or not, forwarded to sklearn

    Returns
    ----------
    results : dict
        parameters
    """
    if alpha is None:
        return _fit_elasticnetCV(X, y, fit_intercept=fit_intercept, **kwargs)
    else:
        elasticnet = ElasticNet(alpha=alpha, fit_intercept=fit_intercept,
                                **kwargs)
        elasticnet.fit(X, y)
        results = dict()
        results['parameters'] = elasticnet.coef_
        return results


def _fit_elasticnetCV(X, y, alphas=None, l1_ratio=None, fit_intercept=False,
                      cv=10, n_jobs=-1, **kwargs):
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by using
    the ElasticNetCV method as implemented in scitkit-learn.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    alphas : list or np.ndarray
        list of alpha values to be evaluated during regularization path
    l1_ratio : float or list(float)
        l1_ratio values to be evaluated during regularization path
    fit_intercept : bool
        center data or not, forwarded to sklearn
    cv : int
        how many folds to carry out in cross-validation

    Returns
    -------
    results : dict
        parameters as well as
        `alpha_optimal` (alpha value that yields the lowest validation RMSE),
        `alpha_path` (all tested alpha values),
        `l1_ratio_optmal` (alpha value that yields the lowest validation RMSE),
        `l1_ratio_path` (all tested `l1_ratio` values)
        `mse_path` (MSE for validation set for each alpha and `l1_ratio`)
    """

    if alphas is None:
        alphas = np.logspace(-8, -0.3, 100)
    if l1_ratio is None:
        l1_ratio = [1.0, 0.995, 0.99, 0.98, 0.97, 0.95, 0.925, 0.9, 0.85,
                    0.8, 0.75, 0.65, 0.5, 0.4, 0.25, 0.1]

    elasticnetCV = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio, cv=cv,
                                fit_intercept=fit_intercept, n_jobs=n_jobs,
                                **kwargs)
    elasticnetCV.fit(X, y)
    results = dict()
    results['parameters'] = elasticnetCV.coef_
    results['alpha_optimal'] = elasticnetCV.alpha_
    results['alpha_path'] = elasticnetCV.alphas_
    results['l1_ratio_path'] = elasticnetCV.l1_ratio
    results['l1_ratio_optimal'] = elasticnetCV.l1_ratio_
    results['mse_path'] = elasticnetCV.mse_path_.mean(axis=2)
    return results


def _fit_bayesian_ridge(X, y, fit_intercept=False, **kwargs):
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by using
    Bayesian ridge regression as implemented in scitkit-learn.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    fit_intercept : bool
        center data or not, forwarded to sklearn

    Returns
    ----------
    results : dict
        parameters
    """
    brr = BayesianRidge(fit_intercept=fit_intercept, **kwargs)
    brr.fit(X, y)
    results = dict()
    results['parameters'] = brr.coef_
    return results


def _fit_ardr(X, y, threshold_lambda=1e6, fit_intercept=False, **kwargs):
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by using
    the automatic relevance determination regression (ARDR) method as
    implemented in scitkit-learn.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    threshold_lambda : float
        threshold lambda parameter forwarded to sklearn
    fit_intercept : bool
        center data or not, forwarded to sklearn

    Returns
    ----------
    results : dict
        parameters
    """
    ardr = ARDRegression(threshold_lambda=threshold_lambda,
                         fit_intercept=fit_intercept, **kwargs)
    ardr.fit(X, y)
    results = dict()
    results['parameters'] = ardr.coef_
    return results


def _fit_rfe_l2(X, y, n_features=None, step=None, **kwargs):
    """Recursive feature elimination (RFE) L2 fitting

    RFE - L2 fitting is a method which runs recusrive feature elimination
    (as implemented in scikit-learn) with least-square fitting. The final model
    is obtained via a least-square fit using the selected features.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    n_features : int
        number of features to select, if None RFECV will be used to determine
        the optimal number of features
    step : int
        number of parameters to eliminate in each iteration

    Returns
    ----------
    results : dict
        parameters and selected features
    """

    n_params = X.shape[1]
    if step is None:
        step = int(np.ceil(n_params / 25))

    if n_features is None:
        return _fit_rfe_l2_CV(X, y, step, **kwargs)
    else:
        # extract features
        lr = LinearRegression(fit_intercept=False)
        rfe = RFE(lr, n_features_to_select=n_features, step=step, **kwargs)
        rfe.fit(X, y)
        features = rfe.support_

        # carry out final fit
        params = np.zeros(n_params)
        params[features] = _fit_least_squares(X[:, features], y)['parameters']

        # finish up
        results = dict(parameters=params, features=features)
        return results


def _fit_rfe_l2_CV(X, y, step=None, rank=1, n_jobs=-1, **kwargs):
    """Recursive feature elimination (RFE) L2 fitting with cross-validation (CV).

    Recursive feature elimination with least-squares fitting using
    cross-validation for optimizing the number of features.

    Parameters
    -----------
    X : np.ndarray
        fit matrix
    y : np.ndarray
        target array
    step : int
        number of parameters to eliminate in each iteration
    rank : int
        rank to use when selecting features

    Returns
    ----------
    results : dict
        parameters and selected features

    """

    n_params = X.shape[1]
    if step is None:
        step = int(np.ceil(n_params / 25))

    # setup
    cv = ShuffleSplit(train_size=0.9, test_size=0.1, n_splits=5)
    lr = LinearRegression(fit_intercept=False)
    rfecv = RFECV(lr, step=step, cv=cv, n_jobs=n_jobs,
                  scoring='neg_mean_squared_error', **kwargs)

    # extract features
    rfecv.fit(X, y)
    ranking = rfecv.ranking_
    features = ranking <= rank

    # carry out final fit
    params = np.zeros(n_params)
    params[features] = _fit_least_squares(X[:, features], y)['parameters']

    # finish up
    results = dict(parameters=params, features=features, ranking=ranking)
    return results


fit_methods = OrderedDict([
    ('least-squares', _fit_least_squares),
    ('lasso', _fit_lasso),
    ('elasticnet', _fit_elasticnet),
    ('bayesian-ridge', _fit_bayesian_ridge),
    ('ardr', _fit_ardr),
    ('rfe-l2', _fit_rfe_l2),
    ('split-bregman', fit_split_bregman)
    ])
available_fit_methods = list(fit_methods.keys())
