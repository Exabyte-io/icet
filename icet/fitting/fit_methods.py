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
from collections import OrderedDict
from sklearn.linear_model import (Lasso,
                                  LassoCV,
                                  Ridge,
                                  RidgeCV,
                                  ElasticNet,
                                  ElasticNetCV,
                                  BayesianRidge,
                                  ARDRegression)
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import RFE, RFECV
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Union
from ..io.logging import logger
from .split_bregman import fit_split_bregman


logger = logger.getChild('fit_methods')


def fit(X: np.ndarray,
        y: np.ndarray,
        fit_method: str,
        standardize: bool = True,
        check_condition: bool = True,
        **kwargs) -> Dict[str, Any]:
    """
    Wrapper function for all available fit methods.  The function
    returns parameters and other pertinent information in the form of
    a dictionary.

    Parameters
    -----------
    X
        fit matrix
    y
        target array
    fit_method
        method to be used for training; possible choice are
        "least-squares", "lasso", "elasticnet", "bayesian-ridge", "ardr",
        "rfe-l2", "split-bregman"
    standardize : bool
        if True the fit matrix is standardized before fitting
    check_condition : bool
        if True the condition number will be checked
        (this can be sligthly more time consuming for larger
        matrices)
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


def _fit_least_squares(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Returns the least-squares solution `a` to the linear problem
    `Xa=y` in the form of a dictionary with a key named `parameters`.

    This function is a wrapper to the `linalg.lstsq` function in NumPy.

    Parameters
    -----------
    X
        fit matrix
    y
        target array
    """
    results = dict()
    results['parameters'] = np.linalg.lstsq(X, y, rcond=-1)[0]
    return results


def _fit_lasso(X: np.ndarray, y: np.ndarray,
               alpha: float = None, fit_intercept: bool = False,
               **kwargs) -> Dict[str, Any]:
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by
    using the LASSO method as implemented in scitkit-learn in the form
    of a dictionary with a key named `parameters`.

    LASSO optimizes the following problem::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    If `alpha` is `None` this function will call `fit_lassoCV` which attempts
    to find the optimal alpha via sklearn's `LassoCV` class.

    Parameters
    ----------
    X
        fit matrix
    y
        target array
    alpha
        alpha value
    fit_intercept
        center data or not, forwarded to sklearn
    """
    if alpha is None:
        return _fit_lassoCV(X, y, fit_intercept=fit_intercept, **kwargs)
    else:
        lasso = Lasso(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        lasso.fit(X, y)
        results = dict()
        results['parameters'] = lasso.coef_
        return results


def _fit_lassoCV(X: np.ndarray,
                 y: np.ndarray,
                 alphas: List[float] = None,
                 fit_intercept: bool = False,
                 cv: int = 10,
                 n_jobs: int = -1,
                 **kwargs) -> Dict[str, Any]:
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by
    using the LassoCV method as implemented in scitkit-learn in the
    form of a dictionary with a key named `parameters`.

    The dictionary will also contain the keys `alpha_optimal` (alpha
    value that yields the lowest validation RMSE), `alpha_path` (all
    tested alpha values), and `mse_path` (MSE for validation set for
    each alpha).

    Parameters
    -----------
    X
        fit matrix
    y
        target array
    alphas
        list of alpha values to be evaluated during regularization path
    fit_intercept
        center data or not, forwarded to sklearn
    cv
        how many folds to carry out in cross-validation
    n_jobs
        number of cores to use during the cross validation.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
        See sklearn's glossary for more details.
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


def _fit_ridge(X, y, alpha=None, fit_intercept=False, **kwargs):
    if alpha is None:
        ridge = RidgeCV(fit_intercept=fit_intercept, **kwargs)
    else:
        ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
    ridge.fit(X, y)
    results = dict()
    results['parameters'] = ridge.coef_
    return results


def _fit_bayesian_ridge(X: np.ndarray, y: np.ndarray,
                        fit_intercept: bool = False,
                        **kwargs) -> Dict[str, Any]:
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by using
    Bayesian ridge regression as implemented in scitkit-learn in the
    form of a dictionary with a key named `parameters`.

    Parameters
    -----------
    X
        fit matrix
    y
        target array
    fit_intercept
        center data or not, forwarded to sklearn
    """
    brr = BayesianRidge(fit_intercept=fit_intercept, **kwargs)
    brr.fit(X, y)
    results = dict()
    results['parameters'] = brr.coef_
    return results


def _fit_elasticnet(X: np.ndarray, y: np.ndarray,
                    alpha: float = None, fit_intercept: bool = False,
                    **kwargs) -> Dict[str, Any]:
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by using
    the ElasticNet method as implemented in scitkit-learn in the
    form of a dictionary with a key named `parameters`.

    If `alpha` is `None` this function will call the fit_lassoCV which attempts
    to find the optimal alpha via sklearn ElasticNetCV class.

    Parameters
    -----------
    X
        fit matrix
    y
        target array
    alpha
        alpha value
    fit_intercept
        center data or not, forwarded to sklearn
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


def _fit_elasticnetCV(X: np.ndarray,
                      y: np.ndarray,
                      alphas: List[float] = None,
                      l1_ratio: Union[float, List[float]] = None,
                      fit_intercept: bool = False,
                      cv: int = 10,
                      n_jobs: int = -1,
                      **kwargs) -> Dict[str, Any]:
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by using
    the ElasticNetCV method as implemented in scitkit-learn in the
    form of a dictionary with a key named `parameters`.

    The dictionary returned by this function will also contain the
    fields `alpha_optimal` (alpha value that yields the lowest
    validation RMSE), `alpha_path` (all tested alpha values),
    `l1_ratio_optmal` (alpha value that yields the lowest validation
    RMSE), `l1_ratio_path` (all tested `l1_ratio` values) `mse_path`
    (MSE for validation set for each alpha and `l1_ratio`)

    Parameters
    -----------
    X
        fit matrix
    y
        target array
    alphas
        list of alpha values to be evaluated during regularization path
    l1_ratio
        l1_ratio values to be evaluated during regularization path
    fit_intercept
        center data or not, forwarded to sklearn
    cv
        how many folds to carry out in cross-validation
    n_jobs
        number of cores to use during the cross validation.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
        See sklearn's glossary for more details.
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


def _fit_ardr(X: np.ndarray,
              y: np.ndarray,
              threshold_lambda: float = 1e6,
              fit_intercept: bool = False,
              **kwargs) -> Dict[str, Any]:
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by
    using the automatic relevance determination regression (ARDR)
    method as implemented in scitkit-learn in the form of a dictionary
    with a key named `parameters`.

    Parameters
    -----------
    X
        fit matrix
    y
        target array
    threshold_lambda
        threshold lambda parameter forwarded to sklearn
    fit_intercept
        center data or not, forwarded to sklearn
    """
    ardr = ARDRegression(threshold_lambda=threshold_lambda,
                         fit_intercept=fit_intercept, **kwargs)
    ardr.fit(X, y)
    results = dict()
    results['parameters'] = ardr.coef_
    return results


class _Estimator:

    def __init__(self, fit_method, **kwargs):
        if fit_method == 'rfe':
            raise ValueError('recursive infinitum')
        self.fit_method = fit_method
        self.kwargs = kwargs
        self.coef_ = None

    def fit(self, X, y):
        fit_func = fit_methods[self.fit_method]
        results = fit_func(X, y, **self.kwargs)
        self.coef_ = results['parameters']

    def get_params(self, deep=True):
        params = {k: v for k, v in self.kwargs.items()}
        params['fit_method'] = self.fit_method
        return params

    def predict(self, A):
        return np.dot(A, self.coef_)


def fit_rfe(X: np.ndarray,
            y: np.ndarray,
            n_features: int = None,
            step: Union[int, float] = 0.04,
            estimator: str = 'least-squares',
            final_estimator: str = None,
            estimator_kwargs: dict = {},
            final_estimator_kwargs: dict = {},
            cv_splits: int = 5,
            n_jobs: int = -1,
            **rfe_kwargs):
    """
    Returns the solution `a` to the linear problem `Xa=y` obtained by
    recursive feature elimination (RFE).

    Parameters
    -----------
    X
        fit matrix
    y
        target array
    n_features
        number of features to select, if None sklearn.feature_selection.RFECV
        will be used to determine the optimal number of features
    step
        if given as integer then corresponds to number of parameters to
        eliminate in each iteration. If given as a float then corresponds to
        the fraction of parameters to remove each iteration.
    estimator
        fit method during RFE algorithm
    final_estimator
        fit_method to be used in final fit,
        if None will default to whichever estimator is being used
    cv_splits
        number of cv-splits to carry out if finding optimal n_features
    n_jobs
        number of cores to use during the cross validation.
        -1 means using all processors.
    """

    # handle kwargs
    if final_estimator is None:
        final_estimator = estimator
        if len(final_estimator_kwargs) == 0:
            final_estimator_kwargs = estimator_kwargs

    estimator_obj = _Estimator(estimator, **estimator_kwargs)
    if n_features is None:
        if 'scoring' not in rfe_kwargs:
            rfe_kwargs['scoring'] = 'neg_mean_squared_error'
        cv = ShuffleSplit(train_size=0.9, test_size=0.1, n_splits=cv_splits)
        rfe = RFECV(estimator_obj, step=step, cv=cv, n_jobs=n_jobs,
                    **rfe_kwargs)
    else:
        rfe = RFE(estimator_obj, n_features_to_select=n_features, step=step,
                  **rfe_kwargs)

    # Carry out RFE
    rfe.fit(X, y)
    features = rfe.support_
    ranking = rfe.ranking_

    # carry out final fit
    n_params = X.shape[1]
    results = fit(X[:, features], y, fit_method=final_estimator,
                  **final_estimator_kwargs)
    params = np.zeros(n_params)
    params[features] = results['parameters']
    results['parameters'] = params

    # finish up
    results['features'] = features
    results['ranking'] = ranking
    return results


fit_methods = OrderedDict([
    ('least-squares', _fit_least_squares),
    ('lasso', _fit_lasso),
    ('ridge', _fit_ridge),
    ('bayesian-ridge', _fit_bayesian_ridge),
    ('elasticnet', _fit_elasticnet),
    ('split-bregman', fit_split_bregman),
    ('ardr', _fit_ardr),
    ('rfe', fit_rfe),
    ])
available_fit_methods = list(fit_methods.keys())
