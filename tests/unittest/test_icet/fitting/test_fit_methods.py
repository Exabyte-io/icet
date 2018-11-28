import numpy as np
import unittest

from icet.fitting import fit, available_fit_methods


class TestFitMethods(unittest.TestCase):
    """Unittest class for fit_methods module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        np.random.seed(42)

        # set up dummy linear problem data
        N, M = 200, 50
        self.n_rows, self.n_cols = N, M
        self.A = np.random.normal(0, 1.0, (N, M))
        self.x = np.random.normal(0.0, 10.0, M)
        noise = np.random.normal(0.0, 0.2, N)
        self.y = np.dot(self.A, self.x) + noise

    def shortDescription(self):
        """Prevents unittest from printing docstring in test cases."""
        return None

    def test_all_available_fit_methods(self):
        """Tests all available fit_methods."""

        # with standardize
        for fit_method in available_fit_methods:
            res = fit(self.A, self.y, fit_method=fit_method, standardize=True)
            self.assertLess(np.linalg.norm(self.x - res['parameters']), 0.2)

        # without standardize
        for fit_method in available_fit_methods:
            res = fit(self.A, self.y, fit_method=fit_method, standardize=False)
            self.assertLess(np.linalg.norm(self.x - res['parameters']), 0.2)

    def test_other_fit_methods(self):
        """Tests fit methods which are not run via available_fit_methods."""

        # lasso with alpha
        res = fit(self.A, self.y, fit_method='lasso', alpha=1e-5)
        self.assertIsNotNone(res['parameters'])

        # elasticnet with alpha
        res = fit(self.A, self.y, fit_method='elasticnet', alpha=1e-5)
        self.assertIsNotNone(res['parameters'])

        # rfe-l2 with n_features
        n_features = int(0.5 * self.n_cols)
        res = fit(self.A, self.y, fit_method='rfe-l2', n_features=n_features)
        self.assertIsNotNone(res['parameters'])
        self.assertEqual(len(res['parameters']), self.n_cols)
        self.assertEqual(sum(res['features']), n_features)

    def test_fit_with_invalid_fit_method(self):
        """Tests correct raise with unavailable fit_method."""
        bad_fit_methods = ['asd', '123', 42, ['lasso']]
        for fit_method in bad_fit_methods:
            with self.assertRaises(ValueError):
                fit(self.A, self.y, fit_method=fit_method)


if __name__ == '__main__':
    unittest.main()
