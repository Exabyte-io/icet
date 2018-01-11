'''
Optimizer with cross validation score
'''

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from .optimizer import Optimizer
from .base_optimizer import BaseOptimizer
from .tools import ScatterData


validation_methods = {
    'k-fold': KFold,
    'shuffle-split': ShuffleSplit,
}


class CrossValidationEstimator(BaseOptimizer):
    '''
    Optimizer with cross validation.

    This optimizer first computes a cross-validation score and finally
    generates a model using the full set of input data.

    Parameters
    ----------
    fit_data : tuple of NumPy (N, M) array and NumPy (N) array
        the first element of the tuple represents the fit matrix `A`
        whereas the second element represents the vector of target
        values `y`; here `N` (=rows of `A`, elements of `y`) equals the number
        of target values and `M` (=columns of `A`) equals the number of
        parameters
    fit_method : string
        method to be used for training; possible choice are
        "least-squares", "lasso", "bayesian-ridge", "ardr"
    validation_method : string
        method to use for cross-validation
    number_of_splits : int
        number of times the fit data set will be split for the cross-validation
    seed : int
        seed for pseudo random number generator

    Attributes
    ----------
    train_scatter_data : ScatterData object (namedtuple)
        contains target and predicted values from training against the full set
        of input data
    validation_scatter_data : ScatterData object (namedtuple)
        contains target and predicted values from each individual
        cross-validation split
    '''

    def __init__(self, fit_data, fit_method='least-squares',
                 validation_method='k-fold', number_of_splits=10,
                 seed=42, **kwargs):

        BaseOptimizer.__init__(self, fit_data, fit_method, seed)

        if validation_method not in validation_methods.keys():
            msg = ['Validation method not available']
            msg += ['Please choose one of the following:']
            for key in validation_methods:
                msg += [' * ' + key]
            raise ValueError('\n'.join(msg))
        self._validation_method = validation_method
        self._number_of_splits = number_of_splits

        self._set_kwargs(kwargs)

        # data set splitting object
        self._splitter = validation_methods[validation_method](
            n_splits=self.number_of_splits, random_state=seed,
            **self._split_kwargs)

    def train(self):
        '''
        Carry out cross-validation and construct final model, where the
        latter is obtained using all available training data
        '''
        self._validate()
        self._construct_final_model()

    def _validate(self):
        ''' Run validation '''
        valid_target, valid_predicted = [], []
        rmse_train_splits, rmse_valid_splits = [], []
        for training_set, test_set in self._splitter.split(self._A):
            opt = Optimizer((self._A, self._y), self.fit_method,
                            training_set=training_set,
                            test_set=test_set,
                            **self._fit_kwargs)
            opt.train()

            rmse_train_splits.append(opt.rmse_training)
            rmse_valid_splits.append(opt.rmse_test)
            valid_target.extend(opt.test_scatter_data.target)
            valid_predicted.extend(opt.test_scatter_data.predicted)

        self._rmse_train_splits = np.array(rmse_train_splits)
        self._rmse_valid_splits = np.array(rmse_valid_splits)
        self.validation_scatter_data = ScatterData(
            target=np.array(valid_target), predicted=np.array(valid_predicted))

    def _construct_final_model(self):
        ''' Construct the final model using all input data available '''
        self._fit_results = self._optimizer_function(self._A, self._y,
                                                     **self._fit_kwargs)
        self._rmse_train_final = self.compute_rmse(self._A, self._y)
        self.train_scatter_data = ScatterData(self._y, self.predict(self._A))

    def _set_kwargs(self, kwargs):
        ''' Sets up fit_kwargs and split_kwargs

        The different split methods need different keywords.
        '''
        self._fit_kwargs = {}
        self._split_kwargs = {}

        if self.validation_method == 'k-fold':
            self._fit_kwargs = kwargs
        elif self.validation_method == 'shuffle-split':
            for key, val in kwargs.items():
                if key in ['test_size', 'train_size']:
                    self._split_kwargs[key] = val
                else:
                    self._fit_kwargs[key] = val

    def __str__(self):
        s = []
        s.append('Validation method: {}'.format(self.validation_method))
        s.append(super().__str__())
        s.append('Number of splits {}'.format(self.number_of_splits))
        return '\n'.join(s)

    @property
    def validation_method(self):
        ''' string : validation method name '''
        return self._validation_method

    @property
    def number_of_splits(self):
        ''' string : number of splits (folds) used for cross-validation '''
        return self._number_of_splits

    @property
    def rmse_training(self):
        '''
        float : root mean squared error when using the full set of input data
        '''
        return self._rmse_train_final

    @property
    def rmse_training_splits(self):
        ''' list : root mean squared training errors obtained during
                   cross-validation '''
        return self._rmse_train_splits

    @property
    def rmse_validation(self):
        ''' float : average root mean squared cross-validation error '''
        return np.mean(self._rmse_valid_splits)

    @property
    def rmse_validation_splits(self):
        ''' list : root mean squared validation errors obtained during
                   cross-validation '''
        return self._rmse_valid_splits
