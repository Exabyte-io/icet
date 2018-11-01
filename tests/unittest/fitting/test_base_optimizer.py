import numpy as np
import unittest

from icet.fitting import available_fit_methods
from icet.fitting.base_optimizer import BaseOptimizer


class TestBaseOptimizer(unittest.TestCase):
    """Unittest class for BaseOptimizer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_rows = 200
        self.n_cols = 50

        # set up dummy linear problem data
        self.A = np.random.normal(0, 1, (self.n_rows, self.n_cols))
        self.x = np.random.normal(0, 5, (self.n_cols, ))
        self.noise = np.random.normal(0, 0.1, (self.n_rows, ))
        self.y = np.dot(self.A, self.x) + self.noise

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        pass

    def test_init(self):
        """Tests initializing BaseOptimizer."""

        # test init with all fit_methods
        for fit_method in available_fit_methods:
            BaseOptimizer((self.A, self.y), fit_method)

        # test init with a fit_method not available
        with self.assertRaises(ValueError):
            BaseOptimizer((self.A, self.y), 'asdasd')

        # test init with a non-aligned fit data
        with self.assertRaises(ValueError):
            A_faulty = np.random.normal(0, 1, (self.n_rows+20, self.n_cols))
            BaseOptimizer((A_faulty, self.y), 'least-squares')

    def test_str(self):
        """Tests str dunder."""
        bopt = BaseOptimizer((self.A, self.y), 'least-squares')
        self.assertIsInstance(str(bopt), str)

    def test_get_contributions(self):
        """Tests get_contributions."""
        A = np.array([[1, 2], [-3, -4]])
        parameters = np.array([1, 10])
        target = np.array([2, 30])

        bopt = BaseOptimizer((self.A, self.y), 'least-squares')
        bopt._fit_results['parameters'] = parameters
        np.testing.assert_almost_equal(target, bopt.get_contributions(A))

    def test_summary_property(self):
        """Tests summary property."""
        bopt = BaseOptimizer((self.A, self.y), 'least-squares')
        self.assertIn('parameters', bopt.summary.keys())
        self.assertIn('fit_method', bopt.summary.keys())


if __name__ == '__main__':
    unittest.main()
