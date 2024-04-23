import os
import tempfile
import unittest
from io import StringIO

import numpy as np
import pytest
from ase.build import bulk
from ase import Atoms  # NOQA (needed for eval(retval))
from icet import ClusterSpace, ClusterExpansion
from pandas import DataFrame


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
        self.structure = bulk('Au')
        self.cutoffs = [3.0] * 3
        chemical_symbols = ['Au', 'Pd']
        self.cs = ClusterSpace(self.structure, self.cutoffs, chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        params_len = len(self.cs)
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
        predicted_val = self.ce.predict(self.structure)
        self.assertEqual(predicted_val, 10.0)

    def test_property_orders(self):
        """Tests orders property."""
        self.assertEqual(self.ce.orders, list(range(len(self.cutoffs) + 2)))

    def test_to_dataframe(self):
        """Tests orbits_as_dataframe property."""
        df = self.ce.to_dataframe()
        self.assertIn('radius', df.columns)
        self.assertIn('order', df.columns)
        self.assertIn('eci', df.columns)
        self.assertEqual(len(df), len(self.parameters))

    def test_get__clusterspace_copy(self):
        """Tests get cluster space copy."""
        self.assertEqual(str(self.ce.get_cluster_space_copy()), str(self.cs))

    def test_cutoffs(self):
        """Tests cutoffs property."""
        self.assertEqual(self.ce.cutoffs, self.cutoffs)

    def test_chemical_symbols(self):
        """Tests chemical_symbols property."""
        target = [['Au', 'Pd']]
        self.assertEqual(self.ce.chemical_symbols, target)

    def test_property_parameters(self):
        """Tests parameters properties."""
        self.assertEqual(list(self.ce.parameters), list(self.parameters))

    def test_len(self):
        """Tests len functionality."""
        self.assertEqual(self.ce.__len__(), len(self.parameters))

    def test_read_write(self):
        """Tests read and write functionalities."""
        # save to file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        self.ce.write(temp_file.name)

        # read from file
        ce_read = ClusterExpansion.read(temp_file.name)
        os.remove(temp_file.name)

        # check cluster space
        self.assertEqual(self.cs._input_structure, ce_read._cluster_space._input_structure)
        self.assertEqual(self.cs._cutoffs, ce_read._cluster_space._cutoffs)
        self.assertEqual(
            self.cs._input_chemical_symbols, ce_read._cluster_space._input_chemical_symbols)

        # check parameters
        self.assertIsInstance(ce_read.parameters, np.ndarray)
        self.assertEqual(list(ce_read.parameters), list(self.parameters))

        # check metadata
        self.assertEqual(len(self.ce.metadata), len(ce_read.metadata))
        self.assertSequenceEqual(sorted(self.ce.metadata.keys()), sorted(ce_read.metadata.keys()))
        for key in self.ce.metadata.keys():
            self.assertEqual(self.ce.metadata[key], ce_read.metadata[key])

    def test_read_write_pruned(self):
        """Tests read and write functionalities."""
        # save to file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.ce.prune(indices=[2, 3])
        self.ce.prune(tol=3)
        pruned_params = self.ce.parameters
        pruned_cs_len = len(self.ce._cluster_space)
        temp_file.close()
        self.ce.write(temp_file.name)

        # read from file
        ce_read = ClusterExpansion.read(temp_file.name)
        params_read = ce_read.parameters
        cs_len_read = len(ce_read._cluster_space)
        os.remove(temp_file.name)

        # check cluster space
        self.assertEqual(cs_len_read, pruned_cs_len)
        self.assertEqual(list(params_read), list(pruned_params))

    def test_prune_cluster_expansion(self):
        """Tests pruning cluster expansion."""
        len_before = len(self.ce)
        self.ce.prune()
        len_after = len(self.ce)
        self.assertEqual(len_before, len_after)

        # Set all parameters to zero except three
        self.ce._parameters = np.array([0.0] * len_after)
        self.ce._parameters[0] = 1.0
        self.ce._parameters[1] = 2.0
        self.ce._parameters[2] = 0.5
        self.ce.prune()
        self.assertEqual(len(self.ce), 3)
        self.assertNotEqual(len(self.ce), len_after)

    def test_prune_cluster_expansion_tol(self):
        """Tests pruning cluster expansion with tolerance."""
        len_before = len(self.ce)
        self.ce.prune()
        len_after = len(self.ce)
        self.assertEqual(len_before, len_after)

        # Set all parameters to zero except two, one of which is
        # non-zero but below the tolerance
        self.ce._parameters = np.array([0.0] * len_after)
        self.ce._parameters[0] = 1.0
        self.ce._parameters[1] = 0.01
        self.ce.prune(tol=0.02)
        self.assertEqual(len(self.ce), 1)
        self.assertNotEqual(len(self.ce), len_after)

    def test_prune_pairs(self):
        """Tests pruning pairs only."""
        df = self.ce.to_dataframe()
        pair_indices = df.index[df['order'] == 2].tolist()
        self.ce.prune(indices=pair_indices)

        df_new = self.ce.to_dataframe()
        pair_indices_new = df_new.index[df_new['order'] == 2].tolist()
        self.assertEqual(pair_indices_new, [])

    def test_prune_zerolet(self):
        """Tests pruning zerolet."""
        with self.assertRaises(ValueError) as context:
            self.ce.prune(indices=[0])
        self.assertTrue('zerolet may not be pruned' in str(context.exception))

    def test_repr(self):
        """Tests __repr__ method."""
        retval = self.ce.__repr__()
        self.assertIn('ClusterExpansion', retval)
        self.assertIn('ClusterSpace', retval)
        ret = eval(retval)
        self.assertIsInstance(ret, ClusterExpansion)

    def test_str(self):
        """Tests __str__ method."""
        retval = self.ce.__str__()
        target = """
================================================ Cluster Expansion ================================================
 space group                            : Fm-3m (225)
 chemical species                       : ['Au', 'Pd'] (sublattice A)
 cutoffs                                : 3.0000 3.0000 3.0000
 total number of parameters             : 5
 number of parameters by order          : 0= 1  1= 1  2= 1  3= 1  4= 1
 fractional_position_tolerance          : 2e-06
 position_tolerance                     : 1e-05
 symprec                                : 1e-05
 total number of nonzero parameters     : 4
 number of nonzero parameters by order  : 0= 0  1= 1  2= 1  3= 1  4= 1
-------------------------------------------------------------------------------------------------------------------
index | order |  radius  | multiplicity | orbit_index | multicomponent_vector | sublattices | parameter |    ECI
-------------------------------------------------------------------------------------------------------------------
   0  |   0   |   0.0000 |        1     |      -1     |           .           |      .      |         0 |         0
   1  |   1   |   0.0000 |        1     |       0     |          [0]          |      A      |         1 |         1
   2  |   2   |   1.4425 |        6     |       1     |        [0, 0]         |     A-A     |         2 |     0.333
   3  |   3   |   1.6657 |        8     |       2     |       [0, 0, 0]       |    A-A-A    |         3 |     0.375
   4  |   4   |   1.7667 |        2     |       3     |     [0, 0, 0, 0]      |   A-A-A-A   |         4 |         2
===================================================================================================================
"""  # noqa

        self.assertEqual(strip_surrounding_spaces(target), strip_surrounding_spaces(retval))

    def test_get_string_representation(self):
        """Tests _get_string_representation functionality."""
        retval = self.ce._get_string_representation(print_threshold=2, print_minimum=1)
        target = """
================================================ Cluster Expansion ================================================
 space group                            : Fm-3m (225)
 chemical species                       : ['Au', 'Pd'] (sublattice A)
 cutoffs                                : 3.0000 3.0000 3.0000
 total number of parameters             : 5
 number of parameters by order          : 0= 1  1= 1  2= 1  3= 1  4= 1
 fractional_position_tolerance          : 2e-06
 position_tolerance                     : 1e-05
 symprec                                : 1e-05
 total number of nonzero parameters     : 4
 number of nonzero parameters by order  : 0= 0  1= 1  2= 1  3= 1  4= 1
-------------------------------------------------------------------------------------------------------------------
index | order |  radius  | multiplicity | orbit_index | multicomponent_vector | sublattices | parameter |    ECI
-------------------------------------------------------------------------------------------------------------------
   0  |   0   |   0.0000 |        1     |      -1     |           .           |      .      |         0 |         0
 ...
   4  |   4   |   1.7667 |        2     |       3     |     [0, 0, 0, 0]      |   A-A-A-A   |         4 |         2
===================================================================================================================
"""  # noqa

        self.assertEqual(strip_surrounding_spaces(target), strip_surrounding_spaces(retval))


class TestClusterExpansionTernary(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestClusterExpansionTernary, self).__init__(*args, **kwargs)
        self.structure = bulk('Au')
        self.cutoffs = [3.0] * 3
        chemical_symbols = ['Au', 'Pd', 'Ag']
        self.cs = ClusterSpace(self.structure, self.cutoffs, chemical_symbols)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        params_len = len(self.cs)
        self.parameters = np.arange(params_len)
        self.ce = ClusterExpansion(self.cs, self.parameters)

    def test_prune_cluster_expansion_with_indices(self):
        """Tests pruning cluster expansion."""

        self.ce.prune(indices=[1, 2, 3, 4, 5])

    def test_prune_cluster_expansion_with_tol(self):
        """Tests pruning cluster expansion."""
        # Prune everything
        self.ce.prune(tol=1e3)
        self.assertEqual(len(self.ce), 1)

    def test_prune_pairs(self):
        """Tests pruning pairs only"""

        df = self.ce.to_dataframe()
        pair_indices = df.index[df['order'] == 2].tolist()
        self.ce.prune(indices=pair_indices)

        df_new = self.ce.to_dataframe()
        pair_indices_new = df_new.index[df_new['order'] == 2].tolist()
        self.assertEqual(pair_indices_new, [])

    def test_property_metadata(self):
        """ Test metadata property. """

        user_metadata = dict(parameters=[1, 2, 3], fit_method='ardr')
        ce = ClusterExpansion(self.cs, self.parameters, metadata=user_metadata)
        metadata = ce.metadata

        # check for user metadata
        self.assertIn('parameters', metadata.keys())
        self.assertIn('fit_method', metadata.keys())

        # check for default metadata
        self.assertIn('date_created', metadata.keys())
        self.assertIn('username', metadata.keys())
        self.assertIn('hostname', metadata.keys())
        self.assertIn('icet_version', metadata.keys())

    def test_property_primitive_structure(self):
        """ Test primitive_structure property.. """
        prim = self.cs.primitive_structure
        self.assertEqual(prim, self.ce.primitive_structure)


@pytest.fixture
def cluster_expansion_fcc():
    prim = bulk('Au', crystalstructure='fcc', a=4.01)
    cs = ClusterSpace(prim, [8, 5], ['Au', 'Pd'])
    return ClusterExpansion(cs, list(range(len(cs))))


def test_to_dataframe_fcc(cluster_expansion_fcc):
    df = cluster_expansion_fcc.to_dataframe()
    orders = [0, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    radii = [0.000000, 0.000000, 1.417749, 2.005000, 2.455613,
             2.835498, 3.170183, 3.472762, 3.751012, 1.637076, 1.854526,
             1.982172, 2.262583, 2.453891, 2.663409, 2.835498]
    multiplicities = [1, 1, 6, 3, 12, 6, 12, 4, 24, 8, 12, 24, 24, 24, 24, 8]
    ecis = [0.000000, 1.000000, 0.333333, 1.000000, 0.333333,
            0.833333, 0.500000, 1.750000, 0.333333, 1.125000, 0.833333,
            0.458333, 0.500000, 0.541667, 0.583333, 1.875000]
    assert isinstance(df, DataFrame)
    assert len(df) == 16
    assert all(df.order == orders)
    assert all(df.multiplicity == multiplicities)
    assert np.allclose(df.radius, radii)
    assert np.allclose(df.eci, ecis)


def test_repr_html_fcc(cluster_expansion_fcc):
    s = cluster_expansion_fcc._repr_html_()
    target = """<h4>Cluster Expansion</h4><table border="1" class="dataframe"><thead><tr><th style="text-align: left;">Field</th><th>Value</th></tr></thead><tbody><tr><td style="text-align: left;">Space group</td><td>Fm-3m (225)</td></tr><tr><td style="text-align: left;">Sublattice A</td><td>('Au', 'Pd')</td></tr><tr><td style="text-align: left;">Cutoffs</td><td>[8, 5]</td></tr><tr><td style="text-align: left;">Total number of parameters (nonzero)</td><td>16 (15)</td></tr><tr><td style="text-align: left;">Number of parameters of order 0 (nonzero)</td><td>1 (0)</td></tr><tr><td style="text-align: left;">Number of parameters of order 1 (nonzero)</td><td>1 (1)</td></tr><tr><td style="text-align: left;">Number of parameters of order 2 (nonzero)</td><td>7 (7)</td></tr><tr><td style="text-align: left;">Number of parameters of order 3 (nonzero)</td><td>7 (7)</td></tr><tr><td style="text-align: left;">fractional_position_tolerance</td><td>2e-06</td></tr><tr><td style="text-align: left;">position_tolerance</td><td>1e-05</td></tr><tr><td style="text-align: left;">symprec</td><td>1e-05</td></tr></tbody></table>"""  # noqa
    assert s.startswith(target)


if __name__ == '__main__':
    unittest.main()
