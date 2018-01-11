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
    train_fraction : float
        fraction of input data (=rows) to be used for training
    test_fraction : float
        fraction of input data (=rows) to be used for testing
    bootstrap : boolean
        if True sampling will be carried out with replacement
    seed : int
        seed for pseudo random number generator
    '''

    def __init__(self, fit_data, fit_method='least-squares', ensemble_size=50,
                 train_fraction=0.7, test_fraction=None,
                 bootstrap=True, seed=42, **kwargs):

        BaseOptimizer.__init__(self, fit_data, fit_method, seed)
        self._ensemble_size = ensemble_size
        self._training_set_size = int(np.round(train_fraction * self._Nrows))
        if test_fraction is not None:
            self._test_set_size = int(np.round(test_fraction * self._Nrows))
            assert (self._training_set_size +
                    self._test_set_size <= self._Nrows), \
                'Size of training and test sets exceeds available data'
        else:
            self._test_set_size = self._Nrows - self._training_set_size

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

        parameters_list = []
        rmse_training_list, rmse_test_list = [], []
        training_set_list, test_set_list = [], []
        for _ in range(self.ensemble_size):
            # construct training and test sets
            rows = np.random.choice(
                self._Nrows, self.training_set_size + self.test_set_size,
                replace=self.bootstrap)
            training_set = np.random.choice(rows, self.training_set_size,
                                            replace=self.bootstrap)
            test_set = np.setdiff1d(rows, training_set)

            # train
            opt = Optimizer((self._A, self._y), self.fit_method,
                            training_set=training_set, test_set=test_set,
                            **self._kwargs)
            opt.train()

            # collect results
            parameters_list.append(opt.parameters)
            rmse_training_list.append(opt.rmse_training)
            rmse_test_list.append(opt.rmse_test)
            training_set_list.append(training_set)
            test_set_list.append(test_set)

        self._parameters_set = np.array(parameters_list)
        self._training_set_list = training_set_list
        self._test_set_list = test_set_list
        self._average_rmse_training = np.average(rmse_training_list)
        self._average_rmse_test = np.average(rmse_test_list)

    def _construct_final_model(self):
        '''
        Construct final model by averaging over all models in the ensemble.
        '''
        self._fit_results['parameters'] = np.mean(self.parameter_vectors,
                                                  axis=0)

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
    def parameters_stddev(self):
        ''' NumPy array : standard deviation for each parameter '''
        if self.fit_results['parameters'] is None:
            return None
        else:
            if self._parameters_stddev is None:
                self._parameters_stddev = np.std(self.parameter_vectors,
                                                 axis=0)
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
        return self._average_rmse_training

    @property
    def rmse_test(self):
        '''
        float : ensemble average of root mean squared error over test sets
        '''
        return self._average_rmse_test

    @property
    def training_set_size(self):
        ''' int : number of rows included in training sets '''
        return self._training_set_size

    @property
    def test_set_size(self):
        ''' int : number of rows included in test sets '''
        return self._test_set_size

    @property
    def fractional_training_set_size(self):
        ''' float : fraction of input data used for training; this value can
                    differ slightly from the value set during initialization
                    due to rounding '''
        return float(self.training_set_size) / self._Nrows

    @property
    def bootstrap(self):
        ''' boolean : True if sampling is carried out with replacement '''
        return self._bootstrap
