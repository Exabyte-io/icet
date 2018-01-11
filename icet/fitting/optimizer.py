'''
Optimizer
'''
import numpy as np
from sklearn.model_selection import train_test_split
from .tools import ScatterData
from .base_optimizer import BaseOptimizer


class Optimizer(BaseOptimizer):
    '''
    Optimizer for single `Ax = y` fit.

    One has to specify either `train_fraction`/`test_fraction` or
    `training_set`/`test_set` If either `training_set` or `test_set` (or both)
    is specified the fractions will be ignored.

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
    train_fraction : float
        fraction of input data (=rows) to be used for training
    test_fraction : float
        fraction of input data (=rows) to be used for testing
    training_set : tuple/list of ints
        indices of rows of `A`/`y` to be used for training
    test_set : tuple/list of ints
        indices of rows of `A`/`y` to be used for testing
    seed : int
        seed for pseudo random number generator

    Attributes
    ----------
    training_scatter_data : ScatterData object (namedtuple)
        target and predicted value for each row in the training set
    test_scatter_data : ScatterData object (namedtuple)
        target and predicted value for each row in the test set
    '''

    def __init__(self, fit_data, fit_method='least-squares',
                 train_fraction=0.75, test_fraction=None,
                 training_set=None, test_set=None, seed=42, **kwargs):

        super().__init__(fit_data, fit_method, seed)

        self._seed = seed
        self._kwargs = kwargs

        # setup training and test sets
        self._setup_rows(train_fraction, test_fraction,
                         training_set, test_set)

        # will be populate once running train
        self._rmse_training = None
        self._rmse_test = None
        self.training_scatter_data = None
        self.test_scatter_data = None

    def train(self):
        ''' Carry out training. '''
        print('{s:-^{n}}'.format(s='Training', n=45))

        # select training data
        A_train = self._A[self.training_set, :]
        y_train = self._y[self.training_set]

        # perform training
        print('Fit Method {}, N_params {}'.format(self.fit_method,
                                                  self.number_of_parameters))
        print('Train size {}, Test size {} '
              .format(self.training_set_size, self.test_set_size))

        self._fit_results = self._optimizer_function(A_train, y_train,
                                                     **self._kwargs)
        self._rmse_training = self.compute_rmse(A_train, y_train)
        self.training_scatter_data = ScatterData(y_train,
                                                 self.predict(A_train))
        print('Train RMSE  {:5.5f}'.format(self.rmse_training))

        # perform validation
        if self.test_set is not None:
            A_test = self._A[self.test_set, :]
            y_test = self._y[self.test_set]
            self._rmse_test = self.compute_rmse(A_test, y_test)
            self.test_scatter_data = ScatterData(y_test,
                                                 self.predict(A_test))
            print('Test  RMSE  {:5.5f}'.format(self.rmse_test))
        else:
            self._rmse_test = None
            self.test_scatter_data = None
            print('Test  RMSE  NaN')
        print('{s:-^{n}}'.format(s='Done', n=45))

    def _setup_rows(self, train_fraction, test_fraction, training_set,
                    test_set):
        '''
        Set up training and test rows depending on which arguments are
        specified.

        If `training_set` and `test_set` are `None` then `train_fraction` and
        `test_fraction` are used.
        '''

        if training_set is None and test_set is None:
            # get rows from fractions
            training_set, test_set = \
                self._get_rows_via_fractions(train_fraction, test_fraction)
        else:  # get rows from specified rows
            training_set, test_set = \
                self._get_rows_from_indices(training_set, test_set)

        if len(training_set) == 0:
            raise ValueError('No training rows was selected from fit_data')

        self._training_set = training_set
        self._test_set = test_set

    def _get_rows_via_fractions(self, train_fraction, test_fraction):
        ''' Gets row via fractions. '''

        # Handle special cases
        if test_fraction is None and train_fraction is None:
            raise ValueError('Both train fraction and test fraction are None')
        elif train_fraction is None and abs(test_fraction - 1.0) < 1e-10:
            raise ValueError('train rows is empty for these fractions')
        elif test_fraction is None and abs(train_fraction - 1.0) < 1e-10:
            training_set = np.arange(self._Nrows)
            test_set = None
            return training_set, test_set

        # split
        training_set, test_set = \
            train_test_split(np.arange(self._Nrows),
                             train_size=train_fraction,
                             test_size=test_fraction,
                             random_state=self.seed)
        if len(test_set) == 0:
            test_set = None
        if len(training_set) == 0:
            raise ValueError('train rows is empty, too small train_fraction')

        return training_set, test_set

    def _get_rows_from_indices(self, training_set, test_set):
        ''' Gets row via indices '''
        if training_set is None and test_set is None:
            raise ValueError('Both training and test set are None')
        elif test_set is None:
            test_set = [i for i in range(self._Nrows)
                        if i not in training_set]
        elif training_set is None:
            training_set = [i for i in range(self._Nrows)
                            if i not in test_set]
        return np.array(training_set), np.array(test_set)

    def get_info(self):
        '''
        Get comprehensive information concerning the optimization process.

        Returns
        -------
        dict
        '''
        info = BaseOptimizer.get_info(self)
        info['rmse-training-set'] = self.rmse_training
        info['rmse-test-set'] = self.rmse_test
        info['training-set-size'] = self.training_set_size
        info['test-set-size'] = self.test_set_size
        info['training-set'] = self.training_set
        info['test-set'] = self.test_set
        return info

    @property
    def rmse_training(self):
        ''' float : root mean squared error for training set '''
        return self._rmse_training

    @property
    def rmse_test(self):
        ''' float : root mean squared error for test set '''
        return self._rmse_test

    @property
    def training_set(self):
        ''' list : indices of the rows included in the training set '''
        return self._training_set

    @property
    def test_set(self):
        ''' list : indices of the rows included in the test set '''
        return self._test_set

    @property
    def training_set_size(self):
        ''' int : number of rows included in training set '''
        if self.training_set is None:
            return 0
        return len(self.training_set)

    @property
    def train_fraction(self):
        ''' float : fraction of rows included in training set '''
        if self.training_set is None:
            return 0.0
        return float(self.training_set_size) / self._Nrows

    @property
    def test_set_size(self):
        ''' int : number of rows included in test set '''
        if self.test_set is None:
            return 0
        return len(self.test_set)

    @property
    def test_fraction(self):
        ''' float : fraction of rows included in test set '''
        if self.test_set is None:
            return 0.0
        return float(self.test_set_size) / self._Nrows
