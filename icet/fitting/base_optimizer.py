'''
BaseOptimizer serves as base for all optimizers.
'''

import numpy as np
from collections import OrderedDict
from .tools import compute_rmse
from .fit_methods import (fit_least_squares,
                          fit_lasso,
                          fit_bayesian_ridge,
                          fit_ardr)


fit_methods = OrderedDict([
    ('least-squares', fit_least_squares),
    ('lasso', fit_lasso),
    ('bayesian-ridge', fit_bayesian_ridge),
    ('ardr', fit_ardr),
    ])


class BaseOptimizer:
    '''
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
    fit_method : string
        method to be used for training; possible choice are
        "least-squares", "lasso", "bayesian-ridge", "ardr"
    seed : int
        seed for pseudo random number generator
    '''

    def __init__(self, fit_data, fit_method, seed):
        '''
        Attributes
        ----------
        _A : NumPy (N, M) array
            fit matrix
        _y : NumPy (N) array
            target values
        _optimizer_function : function
            optimizer function to be called when training
        '''

        if fit_method not in fit_methods.keys():
            raise ValueError('Fit method not available')

        if fit_data[0].shape[0] != fit_data[1].shape[0]:
            raise ValueError('Invalid fit data, shape did not match')

        self._A, self._y = fit_data
        self._Nrows = self._A.shape[0]
        self._Ncols = self._A.shape[1]
        self._fit_method = fit_method
        self._seed = seed
        self._optimizer_function = fit_methods[self.fit_method]
        self._fit_results = {'parameters': None}

    def compute_rmse(self, A, y):
        '''
        Compute the root mean square error using the `A`, `y`, and the
        vector of fitted parameters `x` corresponding to `||Ax-y||_2`.

        Parameters
        ----------
        A : NumPy (N, M) array
            fit matrix where `N` (=rows of `A`, elements of `y`) equals the
            number of target values and `M` (=columns of `A`) equals the number
            of parameters (=elements of `x`)
        y : NumPy (N) array
            vector of target values

        Returns
        -------
        float
            root mean squared error
        '''
        return compute_rmse(A, self.parameters, y)

    def predict(self, A):
        '''
        Predict data given an input matrix `A`, i.e., `Ax`, where `x` is
        the vector of the fitted parameters.

        Parameters
        ----------
        A : NumPy (N, M) array
            fit matrix where `N` (=rows of `A`, elements of `y`) equals the
            number of target values and `M` (=columns of `A`) equals the number
            of parameters

        Returns
        -------
        NumPy (N) array
            vector of predicted values
        '''
        return np.dot(A, self.parameters)

    def get_contributions(self, A):
        '''
        Compute the average contribution to the predicted values from each
        element of the parameter vector.

        Parameters
        ----------
        A : NumPy (N, M) array
            fit matrix where `N` (=rows of `A`, elements of `y`) equals the
            number of target values and `M` (=columns of `A`) equals the number
            of parameters

        Returns
        -------
        NumPy (N, M) array
            average contribution for each row of `A` from each parameter
        '''
        return np.mean(np.abs(np.multiply(A, self.parameters)), axis=0)

    def get_info(self):
        ''' Get comprehensive information concerning the optimization process.

        Returns
        -------
        dict
        '''
        info = dict()
        info['parameters'] = self.parameters
        info['fit-method'] = self.fit_method
        info['number-of-target-values'] = self.number_of_target_values
        info['number-of-parameters'] = self.number_of_parameters
        return info

    def __str__(self):
        s = []
        for key, value in self.get_info().items():
            if type(value) in [str, int, float]:
                s.append('{:25} : {}'.format(key, value))
        return '\n'.join(s)

    def __repr__(self):
        return(str(self))

    @property
    def fit_method(self):
        ''' string : fit method '''
        return self._fit_method

    @property
    def parameters(self):
        ''' NumPy array : copy of parameter vector '''
        if self.fit_results['parameters'] is None:
            return None
        else:
            return self.fit_results['parameters'].copy()

    @property
    def number_of_target_values(self):
        ''' int : number of target values (=rows in `A` matrix) '''
        return self._Nrows

    @property
    def number_of_parameters(self):
        ''' int : number of parameters (=columns in `A` matrix) '''
        return self._Ncols

    @property
    def seed(self):
        ''' int : seed used to initialize pseudo random number of generator '''
        return self._seed

    @property
    def fit_results(self):
        ''' dict : results obtained during training '''
        return self._fit_results
