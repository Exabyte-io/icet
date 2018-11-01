import numpy as np
import unittest

from icet.fitting.tools import compute_correlation_matrix


class TestFittingTools(unittest.TestCase):
    """Unittest class for tools module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_compute_correlation_matrix(self):
        """Tests compute_correlation_matrix."""

        v0 = np.array([1, 1, 1])
        v1 = np.array([1, 1, -2])
        v2 = np.array([-1, -1, -1])

        A = np.array([v0, v1, v2])
        n_rows = len(A)
        C = compute_correlation_matrix(A)

        # check that correlation matrix is symmetric
        for i in range(n_rows):
            for j in range(n_rows):
                self.assertAlmostEqual(C[i][j], C[j][i])

        # check diagonal elements are zero
        for i in range(n_rows):
            self.assertAlmostEqual(C[i][i], 0)

        # check v0-v1 and v1-v2 correlations are zero
        self.assertAlmostEqual(C[0][1], 0)
        self.assertAlmostEqual(C[2][1], 0)

        # check v0-v2 correlation is minus one
        self.assertAlmostEqual(C[0][2], -1)


if __name__ == '__main__':
    unittest.main()
