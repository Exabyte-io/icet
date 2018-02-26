"""
This test checks that a cluster expansion model can be initialized
with any structure in the test database and can predict a property.
"""

from ase.db import connect
from icet import ClusterSpace, ClusterExpansion
import unittest
from ase.build import bulk


class TestClusterExpansion(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestClusterExpansion, self).__init__(*args, **kwargs)
        self.db = connect('structures_for_testing.db')
        self.subelements = ['H', 'He', 'Pb']
        self.cutoffs = [1.4] * 3

    def test_all_systems_in_database(self):
        """
        Test all structures in database
        """
        for row in self.db.select():
            atoms_row = row.toatoms()
            atoms_row.set_chemical_symbols(
                len(atoms_row) * [self.subelements[0]])
            self._test_clusterexpansion_model(
                atoms_row, self.cutoffs, self.subelements)

    def _test_clusterexpansion_model(self, atoms, cutoffs, subelements):
        """
        Test clusterexpansion init and prediction

        Parameters
        ----------
        atoms : ASE Atoms object
            atomic configuration
        subelements : list of strings (chemical symbols)
            list of elements that are allowed
        """
        cs = ClusterSpace(atoms, cutoffs, subelements)
        params_len = cs.get_cluster_space_size()
        params = list(range(params_len))

        ce = ClusterExpansion(cs, params)
        predicted_val = ce.predict(atoms)

        self.assertIsInstance(predicted_val, float)

    def test_read_write(self):
        """
        Test read write functionality.
        """
        atoms = bulk("Al")
        cluster_space = ClusterSpace(atoms, self.cutoffs, self.subelements)
        params = list(range(len(cluster_space)))
        ce = ClusterExpansion(cluster_space, params)

        import tempfile
        f = tempfile.NamedTemporaryFile()
        ce.write(f.name)
        f.seek(0)
        ce_read = ClusterExpansion.read(f.name)

        # Test clusterspace in cluster expansion
        self.assertEqual(cluster_space._input_atoms,
                         ce_read.cluster_space._input_atoms)
        self.assertEqual(list(cluster_space._cutoffs),
                         list(ce_read.cluster_space._cutoffs))
        self.assertEqual(cluster_space._chemical_symbols,
                         ce_read.cluster_space._chemical_symbols)
        self.assertEqual(cluster_space._mi, ce_read.cluster_space._mi)
        self.assertEqual(cluster_space._verbosity,
                         ce_read.cluster_space._verbosity)

        # Test equal parameters
        self.assertEqual(ce_read.parameters, params)


if __name__ == '__main__':
    unittest.main()
