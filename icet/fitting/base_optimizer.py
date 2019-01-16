
"""
BaseOptimizer serves as base for all optimizers.
"""

import numpy as np
from abc import ABC, abstractmethod
from .fit_methods import available_fit_methods


class BaseOptimizer(ABC):
    """
    BaseOptimizer class.

    Serves as base class for all Optimizers solving `Ax = y`.

    Parameters
    ----------
    fit_data : tuple of NumPy (N, M) array and NumPy (N) array
        the first element of the tuple represents the fit matrix `A`
        whereas the second element represents the vector of target
        values `y`; here `N` (=rows of `A`, elements of `y`) equals the number
        of target values and `M` (=columns of `A`) equals the number of
        parameters
    fit_method : str
        method to be used for training; possible choice are
        "least-squares", "lasso", "elasticnet", "bayesian-ridge", "ardr",
        "rfe-l2", "split-bregman"
    standardize : bool
        whether or not to standardize the fit matrix before fitting
    check_condition : bool
        whether or not to carry out a check of the condition number

        N.B.: This can be sligthly more time consuming for larger
        matrices.
    seed : int
        seed for pseudo random number generator
    """

    def __init__(self, fit_data, fit_method, standardize=True,
                 check_condition=True, seed=42):
        """
        Attributes
        ----------
        _A : NumPy (N, M) array
            fit matrix
        _y : NumPy (N) array
            target values
        """

        if fit_method not in available_fit_methods:
            raise ValueError('Unknown fit_method: {}'.format(fit_method))

        if fit_data is None:
            raise TypeError('Invalid fit data; Fit data can not be None')
        if fit_data[0].shape[0] != fit_data[1].shape[0]:
            raise ValueError('Invalid fit data; shapes of fit matrix'
                             ' and target vector do not match')
        if len(fit_data[0].shape) != 2:
            raise ValueError('Invalid fit matrix; must have two dimensions')

        self._A, self._y = fit_data
        self._n_rows = self._A.shape[0]
        self._n_cols = self._A.shape[1]
        self._fit_method = fit_method
        self._standarize = standardize
        self._check_condition = check_condition
        self._seed = seed
        self._fit_results = {'parameters': None}

    def compute_rmse(self, A, y):
        """
        Computes the root mean square error using :math:`A`, :math:`y`,
        and the vector of fitted parameters :math:`x` corresponding to
        :math:`||Ax-y||_2`.

        Parameters
        ----------
        A : np.ndarray
            fit matrix (`N,M` array) where `N` (=rows of `A`, elements
            of `y`) equals the number of target values and `M`
            (=columns of `A`) equals the number of parameters
            (=elements of `x`)
        y : np.ndarray
            vector of target values

        Returns
        -------
        float
            root mean squared error

        """
        y_predicted = self.predict(A)
        delta_y = y_predicted - y
        rmse = np.sqrt(np.mean(delta_y**2))
        return rmse

    def predict(self, A):
        """
        Predicts data given an input matrix :math:`A`, i.e., :math:`Ax`,
        where :math:`x` is the vector of the fitted parameters.

        Parameters
        ----------
        A : np.ndarray
            fit matrix where `N` (=rows of `A`, elements of `y`) equals the
            number of target values and `M` (=columns of `A`) equals the number
            of parameters

        Returns
        -------
        np.ndarray or float
            vector of predicted values; float if single row provided as input

        """
        return np.dot(A, self.parameters)

    def get_contributions(self, A):
        """
        Computes the average contribution to the predicted values from each
        element of the parameter vector.

        Parameters
        ----------
        A : np.ndarray
            fit matrix where `N` (=rows of `A`, elements of `y`) equals the
            number of target values and `M` (=columns of `A`) equals the number
            of parameters

        Returns
        -------
        np.ndarray
            average contribution for each row of `A` from each parameter
        """
        return np.mean(np.abs(np.multiply(A, self.parameters)), axis=0)

    @abstractmethod
    def train(self):
        pass

    @property
    def summary(self):
        """ dict : comprehensive information about the optimizer """
        info = dict()
        info['seed'] = self.seed
        info['fit_method'] = self.fit_method
        info['standardize'] = self.standardize
        info['n_target_values'] = self.n_target_values
        info['n_parameters'] = self.n_parameters
        info['n_nonzero_parameters'] = \
            self.n_nonzero_parameters
        return {**info, **self._fit_results}

    def __str__(self):
        width = 54
        s = []
        s.append(' {} '.format(self.__class__.__name__).center(width, '='))
        for key in sorted(self.summary.keys()):
            value = self.summary[key]
            if isinstance(value, (str, int)):
                s.append('{:30} : {}'.format(key, value))
            elif isinstance(value, (float)):
                s.append('{:30} : {:.7g}'.format(key, value))
        s.append(''.center(width, '='))
        return '\n'.join(s)

    def __repr__(self):
        return 'BaseOptimizer((A, y), {}, {}'.format(
            self.fit_method, self.seed)

    @property
    def fit_method(self):
        """ str : fit method """
        return self._fit_method

    @property
    def parameters(self):
        """ np.ndarray : copy of parameter vector """
        if self._fit_results['parameters'] is None:
            return None
        else:
            return self._fit_results['parameters'].copy()

    @property
    def n_nonzero_parameters(self):
        """ int : number of non-zero parameters """
        if self.parameters is None:
            return None
        else:
            return np.count_nonzero(self.parameters)

    @property
    def n_target_values(self):
        """ int : number of target values (=rows in `A` matrix) """
        return self._n_rows

    @property
    def n_parameters(self):
        """ int : number of parameters (=columns in `A` matrix) """
        return self._n_cols

    @property
    def standardize(self):
        """ bool : whether or not to standardize the fit matrix before
        fitting
        """
        return self._standarize

    @property
    def seed(self):
        """ int : seed used to initialize pseudo random number generator """
        return self._seed
