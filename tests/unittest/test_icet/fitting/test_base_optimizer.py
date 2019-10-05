import numpy as np
import unittest
import tempfile

from icet.fitting import available_fit_methods, read_summary
from icet.fitting.base_optimizer import BaseOptimizer


class RandomOptimizer(BaseOptimizer):
    def __init__(self, fit_data, fit_method, standardize=True, seed=42):
        super().__init__(fit_data, fit_method, standardize, seed)

    def train(self):
        self._fit_results['parameters'] = np.random.random(self._n_cols)


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
        """Prevents unittest from printing docstring in test cases."""
        return None

    def setUp(self):
        pass

    def test_init(self):
        """Tests initializing BaseOptimizer."""

        # test init with all fit_methods
        for fit_method in available_fit_methods:
            RandomOptimizer((self.A, self.y), fit_method)

        # test init with a fit_method not available
        with self.assertRaises(ValueError):
            RandomOptimizer((self.A, self.y), 'asdasd')

        # test init with a non-aligned fit data
        with self.assertRaises(ValueError):
            A_faulty = np.random.normal(0, 1, (self.n_rows+20, self.n_cols))
            RandomOptimizer((A_faulty, self.y), 'least-squares')

    def test_str(self):
        """Tests str dunder."""
        bopt = RandomOptimizer((self.A, self.y), 'least-squares')
        self.assertIsInstance(str(bopt), str)

    def test_get_contributions(self):
        """Tests get_contributions."""

        A = np.array([[1, 2], [-3, -4]])
        parameters = np.array([1, 10])
        target = np.array([2, 30])

        bopt = RandomOptimizer((self.A, self.y), 'least-squares')
        bopt._fit_results['parameters'] = parameters
        np.testing.assert_almost_equal(target, bopt.get_contributions(A))

    def test_summary_property(self):
        """Tests summary property."""
        bopt = RandomOptimizer((self.A, self.y), 'least-squares')
        self.assertIn('parameters', bopt.summary.keys())
        self.assertIn('fit_method', bopt.summary.keys())

    def test_write_summary_and_read(self):
        """ Tests write and read summary functionality """
        bopt = RandomOptimizer((self.A, self.y), 'least-squares')
        bopt.train()

        # write and read summary
        with tempfile.NamedTemporaryFile() as file:
            bopt.write_summary(file.name)
            summary_read = read_summary(file.name)

        # compare summary to read summary
        summary = bopt.summary
        self.assertEqual(len(summary), len(summary_read))
        self.assertSequenceEqual(sorted(summary.keys()), sorted(summary_read.keys()))
        for key in summary.keys():
            if isinstance(summary[key], (int, float, str)):
                self.assertEqual(summary[key], summary_read[key])
            else:
                np.testing.assert_almost_equal(summary[key], summary_read[key])


if __name__ == '__main__':
    unittest.main()
