"""
BaseOptimizer serves as base for all optimizers.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union
from .fit_methods import available_fit_methods


class BaseOptimizer(ABC):
    """BaseOptimizer class.

    Serves as base class for all Optimizers solving the linear
    :math:`\\boldsymbol{X}\\boldsymbol{a} = \\boldsymbol{y}` problem.

    Parameters
    ----------
    fit_data : tuple(numpy.ndarray, numpy.ndarray)
        the first element of the tuple represents the `NxM`-dimensional
        fit matrix `A` whereas the second element represents the
        vector of `N`-dimensional target values `y`; here `N` (=rows of
        `A`, elements of `y`) equals the number of target values and
        `M` (=columns of `A`) equals the number of parameters
    fit_method : str
        method to be used for training; possible choice are
        "least-squares", "lasso", "elasticnet", "bayesian-ridge", "ardr",
        "rfe-l2", "split-bregman"
    standardize : bool
        if True the fit matrix is standardized before fitting
    check_condition : bool
        if True the condition number will be checked
        (this can be sligthly more time consuming for larger
        matrices)
    seed : int
        seed for pseudo random number generator
    """

    def __init__(self,
                 fit_data: Tuple[np.ndarray, np.ndarray],
                 fit_method: str,
                 standardize: bool = True,
                 check_condition: bool = True,
                 seed: int = 42):
        """
        Attributes
        ----------
        _A : numpy.ndarray
            fit matrix (N, M)
        _y : numpy.ndarray
            target values (N)
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

    def compute_rmse(self, A: np.ndarray, y: np.ndarray) -> float:
        """
        Returns the root mean squared error (RMSE) using
        :math:`\\boldsymbol{A}`, :math:`\\boldsymbol{y}`, and the vector of
        fitted parameters :math:`\\boldsymbol{x}`, corresponding to
        :math:`\\|\\boldsymbol{A}\\boldsymbol{x}-\\boldsymbol{y}\\|_2`.

        Parameters
        ----------
        A
            fit matrix (`N,M` array) where `N` (=rows of `A`, elements
            of `y`) equals the number of target values and `M`
            (=columns of `A`) equals the number of parameters
            (=elements of `x`)
        y
            vector of target values
        """
        y_predicted = self.predict(A)
        delta_y = y_predicted - y
        rmse = np.sqrt(np.mean(delta_y**2))
        return rmse

    def predict(self, A: np.ndarray) -> Union[np.ndarray, float]:
        """
        Predicts data given an input matrix :math:`\\boldsymbol{A}`,
        i.e., :math:`\\boldsymbol{A}\\boldsymbol{x}`, where
        :math:`\\boldsymbol{x}` is the vector of the fitted parameters.
        The method returns the vector of predicted values or a float
        if a single row provided as input.

        Parameters
        ----------
        A
            fit matrix where `N` (=rows of `A`, elements of `y`) equals the
            number of target values and `M` (=columns of `A`) equals the number
            of parameters
        """
        return np.dot(A, self.parameters)

    def get_contributions(self, A: np.ndarray) -> np.ndarray:
        """
        Returns the average contribution for each row of `A`
        to the predicted values from each element of the parameter vector.

        Parameters
        ----------
        A
            fit matrix where `N` (=rows of `A`, elements of `y`) equals the
            number of target values and `M` (=columns of `A`) equals the number
            of parameters
        """
        return np.mean(np.abs(np.multiply(A, self.parameters)), axis=0)

    @abstractmethod
    def train(self) -> None:
        pass

    @property
    def summary(self) -> Dict[str, Any]:
        """ comprehensive information about the optimizer """
        info = dict()
        info['seed'] = self.seed
        info['fit_method'] = self.fit_method
        info['standardize'] = self.standardize
        info['n_target_values'] = self.n_target_values
        info['n_parameters'] = self.n_parameters
        info['n_nonzero_parameters'] = \
            self.n_nonzero_parameters
        return {**info, **self._fit_results}

    def __str__(self) -> str:
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

    def __repr__(self) -> str:
        return 'BaseOptimizer((A, y), {}, {}'.format(
            self.fit_method, self.seed)

    @property
    def fit_method(self) -> str:
        """ fit method """
        return self._fit_method

    @property
    def parameters(self) -> np.ndarray:
        """ copy of parameter vector """
        if self._fit_results['parameters'] is None:
            return None
        else:
            return self._fit_results['parameters'].copy()

    @property
    def n_nonzero_parameters(self) -> int:
        """ number of non-zero parameters """
        if self.parameters is None:
            return None
        else:
            return np.count_nonzero(self.parameters)

    @property
    def n_target_values(self) -> int:
        """ number of target values (=rows in `A` matrix) """
        return self._n_rows

    @property
    def n_parameters(self) -> int:
        """ number of parameters (=columns in `A` matrix) """
        return self._n_cols

    @property
    def standardize(self) -> bool:
        """ if True standardize the fit matrix before fitting """
        return self._standarize

    @property
    def seed(self) -> int:
        """ seed used to initialize pseudo random number generator """
        return self._seed
