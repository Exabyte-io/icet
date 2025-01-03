#!/usr/bin/env python3
import unittest
import numpy as np
from icet.tools import ConvexHull


class TestConvexHull(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestConvexHull, self).__init__(*args, **kwargs)
        self.concentrations = [0.0, 1.0, 0.4, 0.6]
        self.energies = [0.0, 10.0, -10.0, 20.0]

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiates class before each test."""
        self.ch = ConvexHull(self.concentrations, self.energies)

    def test_init(self):
        """Tests that initialization of tested class works."""
        self.assertEqual(self.ch.dimensions, 1)
        self.assertTrue(np.allclose(self.ch.energies,
                                    np.array([0.0, -10.0, 10.0])))

    def test_get_energy_at_convex_hull(self):
        """Tests energy at convex hull retrieval functionality."""
        convex_hull_energies = self.ch.get_energy_at_convex_hull([0.2, 0.7])
        self.assertTrue(np.allclose(convex_hull_energies, np.array([-5, 0])))

    def test_extract_low_energy_structures(self):
        """
        Tests extracting of structures that are sufficiently close to convex
        hull.
        """
        concentrations = [0.3, 0.7]
        energies = [-7.0, 0.1]
        energy_tolerance = 0.3
        extracted = self.ch.extract_low_energy_structures(concentrations,
                                                          energies,
                                                          energy_tolerance)
        self.assertEqual(len(extracted), 1)

    def test_str(self):
        target = """============= Convex Hull ==============
 dimensions               : 1
 number of points         : 3
 smallest concentration   : 0.0
 largest concentration    : 1.0
========================================"""
        ret = str(self.ch)
        self.assertEqual(ret, target)

    def test_repr_html(self):
        target = """<h4>Convex Hull</h4><table border="1" class="dataframe"><tbody><tr><td style="text-align: left;">Dimensions</td><td>1</td></tr><tr><td style="text-align: left;">Number of points</td><td>3</td></tr><tr><td style="text-align: left;">Smallest concentration</td><td>0.0</td></tr><tr><td style="text-align: left;">Largest concentration</td><td>1.0</td></tr></tbody></table>"""   # noqa
        ret = self.ch._repr_html_()
        self.assertEqual(ret, target)


class TestConvexHullTernary(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestConvexHullTernary, self).__init__(*args, **kwargs)
        self.concentrations = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                               [0.1, 0.1], [0.3, 0.3]]
        self.energies = [0.0, 10.0, -10.0, 3.0, -7.0]

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiates class before each test."""
        self.ch = ConvexHull(self.concentrations, self.energies)

    def test_init(self):
        """Tests that initialization of tested class works."""
        self.assertEqual(self.ch.dimensions, 2)
        self.assertTrue(np.allclose(self.ch.energies,
                                    np.array([0.0, 10.0, -10.0, -7.0])))

    def test_get_energy_at_convex_hull(self):
        """Tests energy at convex hull retrieval functionality."""
        convex_hull_energies = self.ch.get_energy_at_convex_hull(
            [[0.0, 0.0], [0.15, 0.15]])
        self.assertTrue(np.allclose(convex_hull_energies,
                                    np.array([0.0, -3.5])))

    def test_extract_low_energy_structures(self):
        """
        Tests extracting of structures that are sufficiently close to convex
        hull.
        """
        concentrations = [[0.0, 0.0], [0.15, 0.15]]
        energies = [0.5, -3.3]
        energy_tolerance = 0.4
        extracted = self.ch.extract_low_energy_structures(concentrations,
                                                          energies,
                                                          energy_tolerance)
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0], 1)

    def test_str(self):
        target = """============= Convex Hull ==============
 dimensions               : 2
 number of points         : 4
 smallest concentration   : 0.0
 largest concentration    : 1.0
========================================"""
        ret = str(self.ch)
        self.assertEqual(ret, target)

    def test_repr_html(self):
        target = """<h4>Convex Hull</h4><table border="1" class="dataframe"><tbody><tr><td style="text-align: left;">Dimensions</td><td>2</td></tr><tr><td style="text-align: left;">Number of points</td><td>4</td></tr><tr><td style="text-align: left;">Smallest concentration</td><td>0.0</td></tr><tr><td style="text-align: left;">Largest concentration</td><td>1.0</td></tr></tbody></table>"""   # noqa
        ret = self.ch._repr_html_()
        self.assertEqual(ret, target)


if __name__ == '__main__':
    unittest.main()
