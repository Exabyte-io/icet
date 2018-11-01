import numpy as np
import unittest

from icet.fitting import CrossValidationEstimator
from icet.fitting.cross_validation import validation_methods


class TestCrossValidationEstimator(unittest.TestCase):
    """Unittest class for CrossValidationEstimator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_rows = 200
        self.n_cols = 50
        self.tol = 1.0 / self.n_rows
        self.float_tol = 1e-10

        # set up dummy linear problem data
        self.A = np.random.normal(0, 1, (self.n_rows, self.n_cols))
        self.x = np.random.normal(0, 5, (self.n_cols, ))
        self.noise = np.random.normal(0, 0.1, (self.n_rows, ))
        self.y = np.dot(self.A, self.x) + self.noise

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_init(self):
        """Tests initializing CrossValidationEstimator."""

        # assert valid cv-method
        with self.assertRaises(ValueError):
            CrossValidationEstimator((self.A, self.y), validation_method='asd')

    def test_set_kwargs(self):
        """Tests set_kwargs."""
        kwargs = dict(value1=1.0, value2=2.0, value3='3')

        # test with k-fold
        for validation_method in validation_methods.keys():
            cve = CrossValidationEstimator((self.A, self.y),
                                           validation_method=validation_method)
            cve._set_kwargs(kwargs)
            self.assertDictEqual(cve._fit_kwargs, kwargs)

    def test_train(self):
        """Tests train."""
        cve = CrossValidationEstimator((self.A, self.y))
        self.assertIsNone(cve.parameters)
        self.assertIsNone(cve._rmse_train_final)
        cve.train()
        self.assertIsNotNone(cve.parameters)
        self.assertIsNotNone(cve._rmse_train_final)

    def test_validate(self):
        """Tests validate."""
        n_splits = 7
        for validation_method in validation_methods:
            cve = CrossValidationEstimator(
                (self.A, self.y), number_of_splits=n_splits,
                validation_method=validation_method)

            self.assertIsNone(cve._rmse_train_splits)
            self.assertIsNone(cve._rmse_valid_splits)
            self.assertIsNone(cve.train_scatter_data)
            self.assertIsNone(cve.validation_scatter_data)
            cve.validate()
            self.assertEqual(len(cve._rmse_train_splits), n_splits)
            self.assertEqual(len(cve._rmse_valid_splits), n_splits)
            self.assertIsNotNone(cve.train_scatter_data)
            self.assertIsNotNone(cve.validation_scatter_data)

    def test_summary_property(self):
        """Tests summary property."""

        # without having trained
        cve = CrossValidationEstimator((self.A, self.y))
        self.assertIsInstance(cve.summary, dict)

        # with having validated and trained
        cve.validate()
        cve.train()
        self.assertIsInstance(cve.summary, dict)
        self.assertIn('rmse_train', cve.summary.keys())
        self.assertIn('rmse_validation', cve.summary.keys())

    def test_repr(self):
        """Tests repr dunder."""
        cve = CrossValidationEstimator((self.A, self.y))
        self.assertIsInstance(repr(cve), str)

    def test_rmse_properties(self):
        """Tests the rmse properties."""

        # without having run anything
        cve = CrossValidationEstimator((self.A, self.y))
        self.assertIsNone(cve.rmse_train)
        self.assertIsNone(cve.rmse_train_splits)
        self.assertIsNone(cve.rmse_train_final)
        self.assertIsNone(cve.rmse_validation)
        self.assertIsNone(cve.rmse_validation_splits)

        # after validation
        cve.validate()
        self.assertIsNotNone(cve.rmse_train)
        self.assertIsNotNone(cve.rmse_train_splits)
        self.assertIsNone(cve.rmse_train_final)
        self.assertIsNotNone(cve.rmse_validation)
        self.assertIsNotNone(cve.rmse_validation_splits)

        # after training
        cve.train()
        self.assertIsNotNone(cve.rmse_train_splits)


if __name__ == '__main__':
    unittest.main()
