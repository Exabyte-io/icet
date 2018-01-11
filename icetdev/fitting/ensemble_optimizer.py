'''
Ensemble Optimizer

https://en.wikipedia.org/wiki/Bootstrap_aggregating
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html  # NOQA
'''

import numpy as np
from .base_optimizer import BaseOptimizer
from .optimizer import Optimizer


class EnsembleOptimizer(BaseOptimizer):
    '''Ensemble optimizer that carries out a series of single optimization runs
    using the :class:`Optimizer` class and then provides access to various
    ensemble averaged quantities including e.g., errors and parameters.

    Parameters
    ----------
    fit_data : tuple of (N, M) NumPy array and (N) NumPy array
        the first element of the tuple represents the fit matrix `A`
        whereas the second element represents the vector of target
        values `y`; here `N` (=rows of `A`, elements of `y`) equals the number
        of target values and `M` (=columns of `A`) equals the number of
        parameters
    fit_method : string
        method to be used for training; possible choice are
        "least-squares", "lasso", "bayesian-ridge", "ardr"
    ensemble_size : int
        number of fits in the ensemble
    training_size : float or int
        If float represents the fraction of `fit_data` (rows) to be used for
        training. If int, represents the absolute number of rows to be used for
        training.
    bootstrap : boolean
        if True sampling will be carried out with replacement
    seed : int
        seed for pseudo random number generator
    '''

    def __init__(self, fit_data, fit_method='least-squares', ensemble_size=50,
                 training_size=0.75, bootstrap=True, seed=42, **kwargs):

        BaseOptimizer.__init__(self, fit_data, fit_method, seed)

        # set training size
        if isinstance(training_size, float):
            self._training_size = int(
                np.round(training_size * self.number_of_target_values))
        elif isinstance(training_size, int):
            self._training_size = training_size
        else:
            raise TypeError('Training size must be int or float')

        self._ensemble_size = ensemble_size
        self._bootstrap = bootstrap
        self._kwargs = kwargs
        self._parameters_stddev = None

    def train(self):
        '''
        Carry out ensemble training and construct the final model by averaging
        over all models in the ensemble.
        '''
        self._run_ensemble()
        self._construct_final_model()

    def _run_ensemble(self):
        ''' Construct an ensemble of models. '''

        np.random.seed(self.seed)
        optimizers = []
        for _ in range(self.ensemble_size):
            # construct training and test sets
            training_set = np.random.choice(
                np.arange(self.number_of_target_values), self.training_size,
                replace=self.bootstrap)
            test_set = np.setdiff1d(
                range(self.number_of_target_values), training_set)

            # train
            opt = Optimizer(
                (self._A, self._y), self.fit_method, training_set=training_set,
                test_set=test_set, **self._kwargs)
            opt.train()
            optimizers.append(opt)

        # collect data from each fit
        self._parameters_set = np.array([opt.parameters for opt in optimizers])
        self._training_set_list = [opt.training_set for opt in optimizers]
        self._test_set_list = [opt.test_set for opt in optimizers]
        self._rmse_training_ensemble = np.array(
            [opt.rmse_training for opt in optimizers])
        self._rmse_test_ensemble = np.array(
            [opt.rmse_test for opt in optimizers])

    def _construct_final_model(self):
        '''
        Construct final model by averaging over all models in the ensemble.
        '''
        self._fit_results['parameters'] = np.mean(
            self.parameter_vectors, axis=0)
        self._parameters_stddev = np.std(self.parameter_vectors, axis=0)

    def get_errors(self):
        ''' Get the errors for each fit and each target value.

        Returns
        -------
        NumPy (N,M) array
            matrix of fit errors where `N` is the number of target values and
            `M` is the number of fits (i.e., the size of the ensemble)
        '''
        error_matrix = np.zeros((self._Nrows, self.ensemble_size))
        for i, parameters in enumerate(self.parameter_vectors):
            error_matrix[:, i] = np.dot(self._A, parameters) - self._y
        return error_matrix

    @property
    def summary(self):
        ''' dict : Comprehensive information about the optimizer '''
        info = super().get_info

        # Add class specific data
        info['parameters_stddev'] = self.parameters_stddev
        info['ensemble_size'] = self.ensemble_size
        info['rmse_training'] = self.rmse_training
        info['rmse_training_ensemble'] = self.rmse_training_ensemble
        info['rmse_test'] = self.rmse_test
        info['rmse_test_ensemble'] = self.rmse_test_ensemble
        info['training_size'] = self.training_size
        info['bootstrap'] = self.bootstrap
        return info

    @property
    def parameters_stddev(self):
        ''' NumPy array : standard deviation for each parameter '''
        return self._parameters_stddev

    @property
    def parameter_vectors(self):
        ''' list : all parameter vectors in the ensemble '''
        return self._parameters_set

    @property
    def ensemble_size(self):
        ''' int : number of training rounds '''
        return self._ensemble_size

    @property
    def rmse_training(self):
        '''
        float : ensemble average of root mean squared error over training sets
        '''
        return np.mean(self.rmse_training_ensemble)

    @property
    def rmse_training_ensemble(self):
        ''' list : root mean squared training errors obtained during for each
                   fit in ensemble '''
        return self._rmse_training_ensemble

    @property
    def rmse_test(self):
        '''
        float : ensemble average of root mean squared error over test sets
        '''
        return np.mean(self.rmse_test_ensemble)

    @property
    def rmse_test_ensemble(self):
        ''' list : root mean squared test errors obtained during for each
                   fit in ensemble '''
        return self._rmse_test_ensemble

    @property
    def training_size(self):
        ''' int : number of rows included in training sets. Note that this will
        be different from the number of unique rows if boostrapping '''
        return self._training_size

    @property
    def training_fraction(self):
        ''' float : fraction of input data used for training; this value can
                    differ slightly from the value set during initialization
                    due to rounding '''
        return self.training_set_size / self._Nrows

    @property
    def bootstrap(self):
        ''' boolean : True if sampling is carried out with replacement '''
        return self._bootstrap
