"""
Ensemble Optimizer

https://en.wikipedia.org/wiki/Bootstrap_aggregating
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html  # NOQA
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Union
from .base_optimizer import BaseOptimizer
from .optimizer import Optimizer


class EnsembleOptimizer(BaseOptimizer):
    """
    The ensemble optimizer carries out a series of single optimization runs
    using the :class:`Optimizer` class in order to solve the linear
    :math:`\\boldsymbol{A}\\boldsymbol{x} = \\boldsymbol{y}` problem.
    Subsequently, it provides access to various ensemble averaged
    quantities such as errors and parameters.

    Warning
    -------
    Repeatedly setting up a EnsembleOptimizer and training
    *without* changing the seed for the random number generator will yield
    identical or correlated results, to avoid this please specify a different
    seed when setting up multiple EnsembleOptimizer instances.

    Parameters
    ----------
    fit_data : tuple(numpy.ndarray, numpy.ndarray)
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
    ensemble_size : int
        number of fits in the ensemble
    train_size : float or int
        if float represents the fraction of `fit_data` (rows) to be used for
        training; if int, represents the absolute number of rows to be used for
        training
    bootstrap : bool
        if True sampling will be carried out with replacement
    check_condition : bool
        if True the condition number will be checked
        (this can be sligthly more time consuming for larger
        matrices)
    seed : int
        seed for pseudo random number generator
    """

    def __init__(self,
                 fit_data: Tuple[np.ndarray, np.ndarray],
                 fit_method: str = 'least-squares',
                 standardize: bool = True,
                 ensemble_size: int = 50,
                 train_size: Union[int, float] = 1.0,
                 bootstrap: bool = True,
                 check_condition: bool = True,
                 seed: int = 42,
                 **kwargs) -> None:

        super().__init__(fit_data, fit_method, standardize, check_condition,
                         seed)

        # set training size
        if isinstance(train_size, float):
            self._train_size = int(np.round(train_size * self.n_target_values))
        elif isinstance(train_size, int):
            self._train_size = train_size
        else:
            raise TypeError('Training size must be int or float')

        self._ensemble_size = ensemble_size
        self._bootstrap = bootstrap
        self._kwargs = kwargs
        self._train_set_list = None
        self._test_set_list = None
        self._parameter_vectors = None
        self._parameters_std = None
        self._rmse_train_ensemble = None
        self._rmse_test_ensemble = None

    def train(self) -> None:
        """
        Carries out ensemble training and construct the final model by
        averaging over all models in the ensemble.
        """
        self._run_ensemble()
        self._construct_final_model()

    def _run_ensemble(self) -> None:
        """ Constructs an ensemble of models. """

        rs = np.random.RandomState(self.seed)
        optimizers = []
        for _ in range(self.ensemble_size):
            # construct training and test sets
            train_set = rs.choice(np.arange(self.n_target_values),
                                  self.train_size, replace=self.bootstrap)
            test_set = np.setdiff1d(
                range(self.n_target_values), train_set)

            # train
            opt = Optimizer((self._A, self._y), self.fit_method,
                            standardize=self.standardize,
                            train_set=train_set, test_set=test_set,
                            check_condition=self._check_condition,
                            **self._kwargs)
            opt.train()
            optimizers.append(opt)

        # collect data from each fit

        self._parameter_vectors = np.array(
            [opt.parameters for opt in optimizers])
        self._train_set_list = [opt.train_set for opt in optimizers]
        self._test_set_list = [opt.test_set for opt in optimizers]
        self._rmse_train_ensemble = np.array(
            [opt.rmse_train for opt in optimizers])
        self._rmse_test_ensemble = np.array(
            [opt.rmse_test for opt in optimizers])

    def _construct_final_model(self) -> None:
        """
        Constructs final model by averaging over all models in the ensemble.
        """
        self._fit_results['parameters'] = np.mean(
            self.parameter_vectors, axis=0)
        self._parameters_std = np.std(self.parameter_vectors, axis=0)

    def predict(self,
                A: np.ndarray,
                return_std: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predicts data given an input matrix :math:`\boldsymbol{A}`,
        i.e., :math:`\\boldsymbol{A}\\boldsymbol{x}`, where
        :math:`\\boldsymbol{x}` is the vector of the fitted parameters.
        The method returns the vector of predicted values and optionally also
        the vector of standard deviations.

        By using all parameter vectors in the ensemble a standard deviation of
        the prediction can be obtained.

        Parameters
        ----------
        A
            fit matrix where `N` (=rows of `A`, elements of `y`) equals the
            number of target values and `M` (=columns of `A`) equals the number
            of parameters
        return_std
            whether or not to return the standard deviation of the prediction
        """
        prediction = np.dot(A, self.parameters)
        if return_std:
            predictions = np.dot(A, self.parameter_vectors.T)
            if len(predictions.shape) == 1:  # shape is (N, )
                std = np.std(predictions)
            else:  # shape is (N, M)
                std = np.std(predictions, axis=1)
            return prediction, std
        else:
            return prediction

    @property
    def error_matrix(self) -> np.ndarray:
        """
        matrix of fit errors where `N` is the number of target values and
        `M` is the number of fits (i.e., the size of the ensemble)
        """
        if self.parameter_vectors is None:
            return None
        error_matrix = np.zeros((self._n_rows, self.ensemble_size))
        for i, parameters in enumerate(self.parameter_vectors):
            error_matrix[:, i] = np.dot(self._A, parameters) - self._y
        return error_matrix

    @property
    def summary(self) -> Dict[str, Any]:
        """ comprehensive information about the optimizer """
        info = super().summary

        # Add class specific data
        info['parameters_std'] = self.parameters_std
        info['ensemble_size'] = self.ensemble_size
        info['rmse_train'] = self.rmse_train
        info['rmse_train_ensemble'] = self.rmse_train_ensemble
        info['rmse_test'] = self.rmse_test
        info['rmse_test_ensemble'] = self.rmse_test_ensemble
        info['train_size'] = self.train_size
        info['bootstrap'] = self.bootstrap

        # add kwargs used for fitting
        info = {**info, **self._kwargs}
        return info

    def __repr__(self) -> str:
        kwargs = dict()
        kwargs['fit_method'] = self.fit_method
        kwargs['ensemble_size'] = self.ensemble_size
        kwargs['train_size'] = self.train_size
        kwargs['bootstrap'] = self.bootstrap
        kwargs['seed'] = self.seed
        kwargs = {**kwargs, **self._kwargs}
        return 'EnsembleOptimizer((A, y), {})'.format(
            ', '.join('{}={}'.format(*kwarg) for kwarg in kwargs.items()))

    @property
    def parameters_std(self) -> np.ndarray:
        """ standard deviation for each parameter """
        return self._parameters_std

    @property
    def parameter_vectors(self) -> List[np.ndarray]:
        """ all parameter vectors in the ensemble """
        return self._parameter_vectors

    @property
    def ensemble_size(self) -> int:
        """ number of train rounds """
        return self._ensemble_size

    @property
    def rmse_train(self) -> float:
        """
        ensemble average of root mean squared error over train sets
        """
        if self.rmse_train_ensemble is None:
            return None
        return np.sqrt(np.mean((self.rmse_train_ensemble)**2))

    @property
    def rmse_train_ensemble(self) -> np.ndarray:
        """
        root mean squared train errors obtained during for
        each fit in ensemble
        """
        return self._rmse_train_ensemble

    @property
    def rmse_test(self) -> float:
        """
        ensemble average of root mean squared error over test sets
        """
        if self.rmse_test_ensemble is None:
            return None
        return np.sqrt(np.mean((self.rmse_test_ensemble)**2))

    @property
    def rmse_test_ensemble(self) -> np.ndarray:
        """
        root mean squared test errors obtained during for
        each fit in ensemble
        """
        return self._rmse_test_ensemble

    @property
    def train_size(self) -> int:
        """
        number of rows included in train sets; note that this will
        be different from the number of unique rows if boostrapping
        """
        return self._train_size

    @property
    def train_fraction(self) -> float:
        """
        fraction of input data used for training; this value can differ
        slightly from the value set during initialization due to
        rounding
        """
        return self.train_set_size / self._n_rows

    @property
    def bootstrap(self) -> bool:
        """ True if sampling is carried out with replacement """
        return self._bootstrap
