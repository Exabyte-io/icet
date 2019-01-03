import numpy as np
import unittest

from ase.build import bulk
from ase.neighborlist import NeighborList as ASENeighborList
from icet.core.neighbor_list import NeighborList, get_neighbor_lists
from icet import Structure


class TestNeighborList(unittest.TestCase):
    """Container for test of the module functionality."""
    def __init__(self, *args, **kwargs):
        super(TestNeighborList, self).__init__(*args, **kwargs)

        self.atoms = bulk('Ni', 'hcp', a=1.0).repeat([3, 3, 1])
        self.cutoff = 1.4
        self.structure = Structure.from_atoms(self.atoms)
        self.ase_nl = ASENeighborList(len(self.atoms) * [self.cutoff / 2],
                                      skin=1e-8, bothways=True,
                                      self_interaction=False)
        self.ase_nl.update(self.atoms)
        self.ase_indices, self.ase_offsets = self.ase_nl.get_neighbors(0)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.nl = NeighborList(self.cutoff)
        self.nl.build(self.structure)
        neighbors = self.nl.get_neighbors(0)
        self.indices = []
        self.offsets = []
        for ngb in neighbors:
            self.indices.append(ngb.index)
            self.offsets.append(ngb.unitcell_offset)

    def test_build(self):
        """
        Tests build gives the same number of neighbors as ASE neigborlist.
        """
        for index in range(len(self.atoms)):
            neighbors = self.nl.get_neighbors(index)
            ase_neighbors = self.ase_nl.get_neighbors(index)
            self.assertEqual(len(neighbors), len(ase_neighbors[0]))

    def test_indices(self):
        """Tests indices are equal to those in ASE neighbor list."""
        self.assertEqual(len(self.indices), len(self.ase_indices))
        for index in self.indices:
            self.assertIn(index, self.ase_indices)

    def test_offsets(self):
        """Tests offsets are equal to those in ASE neighbor list."""
        self.assertEqual(len(self.offsets), len(self.ase_offsets))
        for offset in self.offsets:
            self.assertIn(offset, self.ase_offsets)

    def test_equivalent_indices(self):
        """Tests equivalent indices."""
        for index, offset in zip(self.indices, self.offsets):
            equiv_indices = \
                [i for i, ase_offset in enumerate(self.ase_offsets) if
                 self.ase_indices[i] == index and (ase_offset == offset).all()]
            self.assertEqual(len(equiv_indices), 1)
            self.assertEqual(self.ase_indices[equiv_indices[0]], index)

    def test_neighbors_lists(self):
        """
        Tests return neighbor positions from neighborlist
        againts ASE neighborlist.
        """
        for i in range(len(self.atoms)):
            index = [ngb.index for ngb in self.nl.get_neighbors(i)]
            offset = [ngb.unitcell_offset for ngb in self.nl.get_neighbors(i)]
            pos = self.atoms.positions[index] + np.dot(offset,
                                                       self.atoms.get_cell())
            index2, offset2 = self.ase_nl.get_neighbors(i)
            pos2 = self.atoms.positions[index2] + np.dot(offset2,
                                                         self.atoms.get_cell())
            self.assertCountEqual(pos2.tolist(), pos.tolist())

    def test_get_neighbors_lists(self):
        """Tests get_neighbor_lists functionality."""
        list_of_nl = get_neighbor_lists(self.structure, [self.cutoff] * 4)
        self.assertEqual(len(list_of_nl), 4)
        self.assertEqual(len(list_of_nl[0]), len(self.nl))

    def test_neighbors_non_pbc(self):
        """
        Tests indices and offset of neighborlist for a
        non-pbc structure under the same cutoff as above.
        """
        atoms = self.atoms.copy()
        atoms.pbc = [True, True, False]
        atoms.center(4.0, axis=[2])

        structure = Structure.from_atoms(atoms)

        nl = NeighborList(self.cutoff)
        nl.build(structure)
        indices = [ngb.index for ngb in nl.get_neighbors(0)]
        offsets = [ngb.unitcell_offset for ngb in nl.get_neighbors(0)]

        self.ase_nl.update(atoms)
        ase_indices, ase_offsets = self.ase_nl.get_neighbors(0)

        for index in indices:
            self.assertIn(index, ase_indices)

        for offset in offsets:
            self.assertIn(offset, ase_offsets)

        self.assertLess(len(indices), len(self.indices))
        self.assertLess(len(offsets), len(self.offsets))


if __name__ == '__main__':
    unittest.main()
