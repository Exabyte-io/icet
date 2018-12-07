import unittest
import tempfile
from icet import ClusterSpace, ClusterExpansion
from ase.build import bulk
from io import StringIO
import numpy as np


def strip_surrounding_spaces(input_string):
    """
    Helper function that removes both leading and trailing spaces from a
    multi-line string.

    Returns
    -------
    str
        original string minus surrounding spaces and empty lines
    """
    s = []
    for line in StringIO(input_string):
        if len(line.strip()) == 0:
            continue
        s += [line.strip()]
    return '\n'.join(s)


class TestClusterExpansion(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterExpansion, self).__init__(*args, **kwargs)
        self.atoms = bulk('Au')
        self.cutoffs = [3.0] * 3
        chemical_symbols = ['Au', 'Pd']
        self.cs = ClusterSpace(self.atoms, self.cutoffs, chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        params_len = self.cs.get_cluster_space_size()
        self.parameters = np.arange(params_len)
        self.ce = ClusterExpansion(self.cs, self.parameters)

    def test_init(self):
        """Tests that initialization works."""
        self.assertIsInstance(self.ce, ClusterExpansion)

        # test whether method raises Exception
        with self.assertRaises(ValueError) as context:
            ClusterExpansion(self.cs, [0.0])
        self.assertTrue('cluster_space (5) and parameters (1) must have the'
                        ' same length' in str(context.exception))

    def test_predict(self):
        """Tests predict function."""
        predicted_val = self.ce.predict(self.atoms)
        self.assertEqual(predicted_val, 10.0)

    def test_property_orders(self):
        """Tests orders property."""
        self.assertEqual(self.ce.orders, list(range(len(self.cutoffs) + 2)))

    def test_property_parameters_as_dataframe(self):
        """Tests parameters_as_dataframe property."""
        df = self.ce.parameters_as_dataframe
        self.assertIn('radius', df.columns)
        self.assertIn('order', df.columns)
        self.assertIn('eci', df.columns)
        self.assertEqual(len(df), len(self.parameters))

    def test_property_clusterspace(self):
        """Tests cluster space property."""
        self.assertEqual(self.ce.cluster_space, self.cs)

    def test_property_parameters(self):
        """Tests parameters properties."""
        self.assertEqual(list(self.ce.parameters), list(self.parameters))

    def test_len(self):
        """Tests len functionality."""
        self.assertEqual(self.ce.__len__(), len(self.parameters))

    def test_read_write(self):
        """Tests read and write functionalities."""
        # save to file
        temp_file = tempfile.NamedTemporaryFile()
        self.ce.write(temp_file.name)

        # read from file
        temp_file.seek(0)
        ce_read = ClusterExpansion.read(temp_file.name)

        # check cluster space
        self.assertEqual(self.cs._atoms, ce_read.cluster_space._atoms)
        self.assertEqual(self.cs._cutoffs, ce_read.cluster_space._cutoffs)
        self.assertEqual(self.cs._chemical_symbols,
                         ce_read.cluster_space._chemical_symbols)

        self.assertIsInstance(ce_read.parameters, np.ndarray)
        # check parameters
        self.assertEqual(list(ce_read.parameters), list(self.parameters))

    def test_read_write_pruned(self):
        """Tests read and write functionalities."""
        # save to file
        temp_file = tempfile.NamedTemporaryFile()
        self.ce.prune(indices=[2, 3])
        self.ce.prune(tol=3)
        pruned_params = self.ce.parameters
        pruned_cs_len = len(self.ce.cluster_space)
        self.ce.write(temp_file.name)

        # read from file
        temp_file.seek(0)
        ce_read = ClusterExpansion.read(temp_file.name)
        params_read = self.ce.parameters
        cs_len_read = len(self.ce.cluster_space)

        # check cluster space
        self.assertEqual(cs_len_read, pruned_cs_len)
        self.assertEqual(list(params_read), list(pruned_params))

    def test_prune_cluster_expansion(self):
        """Tests pruning cluster expansion."""

        len_before = len(self.ce)
        self.ce.prune()
        len_after = len(self.ce)

        self.assertEqual(len_before, len_after)

        # Set all to zero except one
        self.ce._parameters = np.array([0] * len_after)
        self.ce._parameters[0] = 1

        self.ce.prune()
        self.assertEqual(len(self.ce), 1)
        self.assertNotEqual(len(self.ce), len_after)

    def test_prune_pairs(self):
        """Tests pruning pairs only"""

        df = self.ce.parameters_as_dataframe
        pair_indices = df.index[df['order'] == 2].tolist()
        self.ce.prune(indices=pair_indices)

        df_new = self.ce.parameters_as_dataframe
        pair_indices_new = df_new.index[df_new['order'] == 2].tolist()
        self.assertEqual(pair_indices_new, [])

    def test_prune_zerolet(self):
        """Tests pruning zerolet"""
        with self.assertRaises(ValueError) as context:
            self.ce.prune(indices=[0])

        self.assertTrue('zerolet may not be pruned' in str(context.exception))

    def test_repr(self):
        """Tests repr functionality."""

        retval = self.ce.__repr__()
        target = """
=================================== Cluster Expansion ====================================
 chemical species: ['Au', 'Pd']
 cutoffs: 3.0000 3.0000 3.0000
 total number of orbits: 5
 number of orbits by order: 0= 1  1= 1  2= 1  3= 1  4= 1
------------------------------------------------------------------------------------------
index | order |  radius  | multiplicity | orbit_index | multi_component_vector |    ECI
------------------------------------------------------------------------------------------
   0  |   0   |   0.0000 |        1     |      -1     |           .            |         0
   1  |   1   |   0.0000 |        1     |       0     |          [0]           |         1
   2  |   2   |   1.4425 |        6     |       1     |         [0, 0]         |         2
   3  |   3   |   1.6657 |        8     |       2     |       [0, 0, 0]        |         3
   4  |   4   |   1.7667 |        2     |       3     |      [0, 0, 0, 0]      |         4
==========================================================================================
"""  # noqa

        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_get_string_representation(self):
        """Tests _get_string_representation functionality."""

        retval = self.ce._get_string_representation(print_threshold=2,
                                                    print_minimum=1)
        target = """
=================================== Cluster Expansion ====================================
 chemical species: ['Au', 'Pd']
 cutoffs: 3.0000 3.0000 3.0000
 total number of orbits: 5
 number of orbits by order: 0= 1  1= 1  2= 1  3= 1  4= 1
------------------------------------------------------------------------------------------
index | order |  radius  | multiplicity | orbit_index | multi_component_vector |    ECI
------------------------------------------------------------------------------------------
   0  |   0   |   0.0000 |        1     |      -1     |           .            |         0
 ...
   4  |   4   |   1.7667 |        2     |       3     |      [0, 0, 0, 0]      |         4
==========================================================================================
"""  # noqa
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))


class TestClusterExpansionTernary(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterExpansionTernary, self).__init__(*args, **kwargs)
        self.atoms = bulk('Au')
        self.cutoffs = [3.0] * 3
        chemical_symbols = ['Au', 'Pd', 'Ag']
        self.cs = ClusterSpace(self.atoms, self.cutoffs, chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        params_len = self.cs.get_cluster_space_size()
        self.parameters = np.arange(params_len)
        self.ce = ClusterExpansion(self.cs, self.parameters)

    def test_prune_cluster_expansion(self):
        """Tests pruning cluster expansion."""

        self.ce.prune(indices=[1, 2, 3, 4, 5])

    def test_prune_cluster_expansion(self):
        """Tests pruning cluster expansion."""
        # Prune everything
        self.ce.prune(tol=1e3)
        self.assertEqual(len(self.ce), 1)

    def test_prune_pairs(self):
        """Tests pruning pairs only"""

        df = self.ce.parameters_as_dataframe
        pair_indices = df.index[df['order'] == 2].tolist()
        self.ce.prune(indices=pair_indices)

        df_new = self.ce.parameters_as_dataframe
        pair_indices_new = df_new.index[df_new['order'] == 2].tolist()
        self.assertEqual(pair_indices_new, [])


if __name__ == '__main__':
    unittest.main()
