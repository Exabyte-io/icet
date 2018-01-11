'''
This module provides simplified interfaces for vaiours linear model
regression methods.

More information about the sklearn regression can be found at
http://scikit-learn.org/stable/modules/linear_model.html


Todo
----
* add HuberRegression robust vs outliers
* add ElasticNet
'''

import numpy as np
from ..io.logging import logger
from .tools import compute_rmse
try:
    from sklearn.linear_model import Lasso, BayesianRidge, ARDRegression
    from sklearn.model_selection import KFold
    # arrangement of logger assignments is owed to pep8 requirements
    logger = logger.getChild('fit_methods')
except Exception:
    logger = logger.getChild('fit_methods')
    logger.warning('Failed to import scitkit-learn;'
                   ' several optimizers will fail')


def fit_least_squares(X, y):
    '''
    Return the least-squares solution `a` to the linear problem `Xa=y`.

    This function is a wrapper to the `linalg.lstsq` function in NumPy.

    Parameters
    -----------
    X : matrix / array
        fit matrix
    y : array
        target array

    Returns
    ----------
    results : dict
        dict containing parameters
    '''
    results = {}
    results['parameters'] = np.linalg.lstsq(X, y)[0]
    return results


def fit_lasso(X, y, alpha=None, fit_intercept=False, **kwargs):
    '''
    Return the solution `a` to the linear problem `Xa=y` obtained by using
    the LASSO method as implemented in scitkit-learn.

    LASSO optimizes the following problem::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    If `alpha` is `None` this function will conduct a grid search for an
    optimal alpha value.

    Parameters
    -----------
    X : matrix / array
        fit matrix
    y : array
        target array
    alpha : float
        alpha value
    fit_intercept : bool
        center data or not, forwarded to sklearn

    Returns
    ----------
    results : dict
        dictionary containing parameters
    '''
    if alpha is None:
        return fit_lasso_optimize_alpha(X, y, fit_intercept=fit_intercept,
                                        **kwargs)
    else:
        lasso = Lasso(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        lasso.fit(X, y)
        results = {}
        results['parameters'] = lasso.coef_
        return results


def fit_lasso_optimize_alpha(X, y, alphas=None, fold=10, fit_intercept=False,
                             verbose=False, **kwargs):
    '''
    Return the solution `a` to the linear problem `Xa=y` obtained by using
    the LASSO method as implemented in scitkit-learn.

    The `alpha` parameter is optimized via grid search and the test score is
    computed using k-fold validation.

    Parameters
    -----------
    X : matrix / array
        fit matrix
    y : array
        target array
    alphas : array
        alpha values to evaluate
    fold : int
        Number of times to fold dataset when computing test score
    fit_intercept : bool
        center data or not, forwarded to sklearn
    verbose : boolean
        if True additional information concerning the optimization process will
        be logged

    Returns
    ----------
    results : dict
        dictionary containing parameters,
        alpha-path (all tested alpha values),
        rmse-path (rmse for validation set for each alpha),
        alpha-optimal (the alpha value with lowest rmse validation)
    '''

    if alphas is None:
        alphas = np.logspace(-6, -0.3, 50)

    # Alpha grid search
    lasso = Lasso(fit_intercept=fit_intercept, **kwargs)
    kf = KFold(n_splits=fold, shuffle=False)

    RMSE_path = []
    for i, alpha in enumerate(alphas):
        lasso.alpha = alpha
        cv_fold = []
        for train, test in kf.split(X):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            lasso.fit(X_train, y_train)
            RMSE = compute_rmse(X_test, lasso.coef_, y_test)
            cv_fold.append(RMSE)
        RMSE_path.append(np.mean(cv_fold))

    RMSE_path = np.array(RMSE_path)
    alpha_min = alphas[np.argmin(RMSE_path)]

    if np.argmin(RMSE_path) == 0 or np.argmin(RMSE_path) == len(alphas)-1:
        logger.warning('alpha_min {} at edge of grid search'.format(alpha_min))

    # Make final fit
    lasso.alpha = alpha_min
    lasso.fit(X, y)

    results = {}
    results['parameters'] = lasso.coef_
    results['rmse-path'] = RMSE_path
    results['alpha-path'] = alphas
    results['alpha-optimal'] = alpha_min
    return results


def fit_bayesian_ridge(X, y, fit_intercept=False, **kwargs):
    '''
    Return the solution `a` to the linear problem `Xa=y` obtained by using
    Bayesian ridge regression as implemented in scitkit-learn.

    Parameters
    -----------
    X : matrix / array
        fit matrix
    y : array
        target array
    fit_intercept : bool
        center data or not, forwarded to sklearn

    Returns
    ----------
    results : dict
        dict containing parameters, covariance matrix
    '''
    brr = BayesianRidge(fit_intercept=fit_intercept, **kwargs)
    brr.fit(X, y)
    results = {}
    results['parameters'] = brr.coef_
    results['covariance-matrix'] = brr.sigma_
    return results


def fit_ardr(X, y, threshold_lambda=1e8, fit_intercept=False, **kwargs):
    '''
    Return the solution `a` to the linear problem `Xa=y` obtained by using
    the automatic relevance determination regression (ARDR) method as
    implemented in scitkit-learn.

    Parameters
    -----------
    X : matrix / array
        fit matrix
    y : array
        target array
    threshold_lambda : float
        threshold lambda parameter forwarded to sklearn
    fit_intercept : bool
        center data or not, forwarded to sklearn

    Returns
    ----------
    results : dict
        dictionary containing parameters, covariance matrix
    '''
    ardr = ARDRegression(threshold_lambda=threshold_lambda,
                         fit_intercept=fit_intercept, **kwargs)
    ardr.fit(X, y)
    results = {}
    results['parameters'] = ardr.coef_
    results['covariance-matrix'] = ardr.sigma_
    return results
