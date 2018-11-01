import numpy as np
import unittest

from icet.fitting import fit, available_fit_methods


class TestOptimizer(unittest.TestCase):
    """Unittest class for fit_methods module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        np.random.seed(42)

        # set up dummy linear problem data
        N, M = 200, 50
        self.A = np.random.normal(0, 1.0, (N, M))
        self.x = np.random.normal(0.0, 10.0, M)
        noise = np.random.normal(0.0, 0.2, N)
        self.y = np.dot(self.A, self.x) + noise

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_all_available_fit_methods(self):
        """Tests all available fit_methods."""
        for fit_method in available_fit_methods:
            res = fit(self.A, self.y, fit_method=fit_method)
            self.assertLess(np.linalg.norm(self.x - res['parameters']), 0.2)

    def test_other_fit_methods(self):
        """Tests fit methods which are not run via available_fit_methods."""

        # lasso with alpha
        res = fit(self.A, self.y, fit_method='lasso', alpha=1e-5)
        self.assertIsNotNone(res['parameters'])

        # elasticnet with alpha
        res = fit(self.A, self.y, fit_method='elasticnet', alpha=1e-5)
        self.assertIsNotNone(res['parameters'])


if __name__ == '__main__':
    unittest.main()
