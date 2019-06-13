"""
Optimizer with cross validation score
"""

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from typing import Any, Dict, Tuple
from .base_optimizer import BaseOptimizer
from .optimizer import Optimizer
from .fit_methods import fit
from .tools import ScatterData


validation_methods = {
    'k-fold': KFold,
    'shuffle-split': ShuffleSplit,
}


class CrossValidationEstimator(BaseOptimizer):
    """
    This class provides an optimizer with cross validation for solving the
    linear :math:`\\boldsymbol{A}\\boldsymbol{x} = \\boldsymbol{y}` problem.
    Cross-validation (CV) scores are calculated by splitting the
    available reference data in multiple different ways.  It also produces
    the finalized model (using the full input data) for which the CV score
    is an estimation of its performance.

    Warning
    -------
    Repeatedly setting up a CrossValidationEstimator and training
    *without* changing the seed for the random number generator will yield
    identical or correlated results, to avoid this please specify a different
    seed when setting up multiple CrossValidationEstimator instances.

    Parameters
    ----------
    fit_data : tupe(numpy.ndarray, numpy.ndarray)
        the first element of the tuple represents the fit matrix `A`
        (`N, M` array) while the second element represents the vector
        of target values `y` (`N` array); here `N` (=rows of `A`,
        elements of `y`) equals the number of target values and `M`
        (=columns of `A`) equals the number of parameters
    fit_method : str
        method to be used for training; possible choice are
        "least-squares", "lasso", "elasticnet", "bayesian-ridge", "ardr",
        "rfe-l2", "split-bregman"
    standardize : bool
        if True the fit matrix is standardized before fitting
    validation_method : str
        method to use for cross-validation; possible choices are
        "shuffle-split", "k-fold"
    n_splits : int
        number of times the fit data set will be split for the cross-validation
    check_condition : bool
        if True the condition number will be checked
        (this can be sligthly more time consuming for larger
        matrices)
    seed : int
        seed for pseudo random number generator

    Attributes
    ----------
    train_scatter_data : ScatterData
        contains target and predicted values from each individual
        traininig set in the cross-validation split;
        :class:`ScatterData` is a namedtuple.
    validation_scatter_data : ScatterData
        contains target and predicted values from each individual
        validation set in the cross-validation split;
        :class:`ScatterData` is a namedtuple.

    """

    def __init__(self,
                 fit_data: Tuple[np.ndarray, np.ndarray],
                 fit_method: str = 'least-squares',
                 standardize: bool = True,
                 validation_method: str = 'k-fold',
                 n_splits: int = 10,
                 check_condition: bool = True,
                 seed: int = 42,
                 **kwargs) -> None:

        super().__init__(fit_data, fit_method, standardize, check_condition,
                         seed)

        if validation_method not in validation_methods.keys():
            msg = ['Validation method not available']
            msg += ['Please choose one of the following:']
            for key in validation_methods:
                msg += [' * ' + key]
            raise ValueError('\n'.join(msg))
        self._validation_method = validation_method
        self._n_splits = n_splits

        self._set_kwargs(kwargs)

        # data set splitting object
        self._splitter = validation_methods[validation_method](
            n_splits=self.n_splits, random_state=seed,
            **self._split_kwargs)

        self.train_scatter_data = None
        self.validation_scatter_data = None

        self._parameters_splits = None
        self._rmse_train_splits = None
        self._rmse_valid_splits = None
        self._rmse_train_final = None

    def train(self) -> None:
        """ Constructs the final model using all input data available. """
        self._fit_results = fit(self._A, self._y, self.fit_method,
                                self.standardize, self._check_condition,
                                **self._fit_kwargs)
        self._rmse_train_final = self.compute_rmse(self._A, self._y)

    def validate(self) -> None:
        """ Runs validation. """
        train_target, train_predicted = [], []
        valid_target, valid_predicted = [], []
        rmse_train_splits, rmse_valid_splits = [], []
        parameters_splits = []
        for train_set, test_set in self._splitter.split(self._A):
            opt = Optimizer((self._A, self._y), self.fit_method,
                            standardize=self.standardize,
                            train_set=train_set,
                            test_set=test_set,
                            check_condition=self._check_condition,
                            **self._fit_kwargs)
            opt.train()

            parameters_splits.append(opt.parameters)
            rmse_train_splits.append(opt.rmse_train)
            rmse_valid_splits.append(opt.rmse_test)
            train_target.extend(opt.train_scatter_data.target)
            train_predicted.extend(opt.train_scatter_data.predicted)
            valid_target.extend(opt.test_scatter_data.target)
            valid_predicted.extend(opt.test_scatter_data.predicted)

        self._parameters_splits = np.array(parameters_splits)
        self._rmse_train_splits = np.array(rmse_train_splits)
        self._rmse_valid_splits = np.array(rmse_valid_splits)
        self.train_scatter_data = ScatterData(
            target=np.array(train_target), predicted=np.array(train_predicted))
        self.validation_scatter_data = ScatterData(
            target=np.array(valid_target), predicted=np.array(valid_predicted))

    def _set_kwargs(self, kwargs: dict) -> None:
        """
        Sets up fit_kwargs and split_kwargs.
        Different split methods need different keywords.
        """
        self._fit_kwargs = {}
        self._split_kwargs = {}

        if self.validation_method == 'k-fold':
            for key, val in kwargs.items():
                if key in ['shuffle']:
                    self._split_kwargs[key] = val
                else:
                    self._fit_kwargs[key] = val
        elif self.validation_method == 'shuffle-split':
            for key, val in kwargs.items():
                if key in ['test_size', 'train_size']:
                    self._split_kwargs[key] = val
                else:
                    self._fit_kwargs[key] = val

    @property
    def summary(self) -> Dict[str, Any]:
        """ comprehensive information about the optimizer """
        info = super().summary

        # Add class specific data
        info['validation_method'] = self.validation_method
        info['n_splits'] = self.n_splits
        info['rmse_train_final'] = self.rmse_train_final
        info['rmse_train'] = self.rmse_train
        info['rmse_train_splits'] = self.rmse_train_splits
        info['rmse_validation'] = self.rmse_validation
        info['rmse_validation_splits'] = self.rmse_validation_splits
        info['train_scatter_data'] = self.train_scatter_data
        info['validation_scatter_data'] = self.validation_scatter_data

        # add kwargs used for fitting and splitting
        info = {**info, **self._fit_kwargs, **self._split_kwargs}
        return info

    def __repr__(self) -> str:
        kwargs = dict()
        kwargs['fit_method'] = self.fit_method
        kwargs['validation_method'] = self.validation_method
        kwargs['n_splits'] = self.n_splits
        kwargs['seed'] = self.seed
        kwargs = {**kwargs, **self._fit_kwargs, **self._split_kwargs}
        return 'CrossValidationEstimator((A, y), {})'.format(
            ', '.join('{}={}'.format(*kwarg) for kwarg in kwargs.items()))

    @property
    def validation_method(self) -> str:
        """ validation method name """
        return self._validation_method

    @property
    def n_splits(self) -> int:
        """ number of splits (folds) used for cross-validation """
        return self._n_splits

    @property
    def parameters_splits(self) -> np.ndarray:
        """ all parameters obtained during cross-validation """
        return self._parameters_splits

    @property
    def n_nonzero_parameters_splits(self) -> list:
        """ number of non-zero parameters for each split """
        if self.parameters_splits is None:
            return None
        else:
            return [np.count_nonzero(p) for p in self.parameters_splits]

    @property
    def rmse_train_final(self) -> float:
        """
        root mean squared error when using the full set of input data
        """
        return self._rmse_train_final

    @property
    def rmse_train(self) -> float:
        """
        average root mean squared training error obtained during
        cross-validation
        """
        if self._rmse_train_splits is None:
            return None
        return np.sqrt(np.mean(self._rmse_train_splits**2))

    @property
    def rmse_train_splits(self) -> np.ndarray:
        """
        root mean squared training errors obtained during
        cross-validation
        """
        return self._rmse_train_splits

    @property
    def rmse_validation(self) -> float:
        """ average root mean squared cross-validation error """
        if self._rmse_valid_splits is None:
            return None
        return np.sqrt(np.mean(self._rmse_valid_splits**2))

    @property
    def rmse_validation_splits(self) -> np.ndarray:
        """
        root mean squared validation errors obtained during
        cross-validation
        """
        return self._rmse_valid_splits
