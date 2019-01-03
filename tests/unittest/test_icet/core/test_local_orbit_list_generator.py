import unittest
from ase.build import bulk, make_supercell
import numpy as np

from icet import OrbitList, Structure
from icet.core.lattice_site import LatticeSite
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator


class TestLocalOrbitListGenerator(unittest.TestCase):
    """Container for test of class functionality."""
    def __init__(self, *args, **kwargs):
        super(TestLocalOrbitListGenerator, self).__init__(*args, **kwargs)
        # corner case: tilted structure
        prim_atoms = bulk('Al')
        cutoffs = [4.2, 4.2]
        self.orbit_list = OrbitList(prim_atoms, cutoffs)
        self.primitive = self.orbit_list.get_primitive_structure()
        super_atoms = make_supercell(prim_atoms, [[2, 0, 1000],
                                                  [0, 2, 0],
                                                  [0, 0, 2]])
        self.supercell = Structure.from_atoms(super_atoms)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiate class for each test case."""
        self.lolg = LocalOrbitListGenerator(self.orbit_list, self.supercell)

    def test_generate_local_orbit_list_from_index(self):
        """
        Tests that function generates an orbit list from
        an index of a specific offset of the primitive structure.
        """
        unique_offsets = self.lolg._get_unique_primcell_offsets()

        for index, offset in enumerate(unique_offsets):
            local_orbit_list = self.lolg.generate_local_orbit_list(index)
            for orbit_prim, orbit_super in zip(self.orbit_list.orbits,
                                               local_orbit_list.orbits):
                for site_p, site_s in zip(orbit_prim.representative_sites,
                                          orbit_super.representative_sites):
                    site_p.unitcell_offset += offset
                    pos_super = self.supercell.get_position(site_s)
                    pos_prim = self.primitive.get_position(site_p)
                    self.assertTrue(np.all(np.isclose(pos_super, pos_prim)))

    def test_generate_local_orbit_list_from_offset(self):
        """
        Tests that function generates an orbit list for the given
        offset of the primitive structure.
        """
        unique_offsets = self.lolg._get_unique_primcell_offsets()
        for offset in unique_offsets:
            local_orbit_list = self.lolg.generate_local_orbit_list(offset)
            for orbit_prim, orbit_super in zip(self.orbit_list.orbits,
                                               local_orbit_list.orbits):
                for site_p, site_s in zip(orbit_prim.representative_sites,
                                          orbit_super.representative_sites):
                    site_p.unitcell_offset += offset
                    pos_super = self.supercell.get_position(site_s)
                    pos_prim = self.primitive.get_position(site_p)
                    self.assertTrue(np.all(np.isclose(pos_super, pos_prim)))

    def test_generate_full_orbit_list(self):
        """
        Tests that equivalent sites of all local orbit lists are listed
        as equivalent sites in the full orbit list.
        """
        fol = self.lolg.generate_full_orbit_list()
        for offset in self.lolg._get_unique_primcell_offsets():
            lol = self.lolg.generate_local_orbit_list(offset)
            for orbit, orbit_ in zip(lol.orbits, fol.orbits):
                for sites in orbit.equivalent_sites:
                    self.assertIn(sites, orbit_.equivalent_sites)

    def test_clear(self):
        """
        Tests vector of unique offsets and primitive to supercell mapping
        are cleared.
        """
        self.lolg.generate_local_orbit_list(2)
        self.lolg.clear()
        offsets_count = self.lolg.get_number_of_unique_offsets()
        self.assertEqual(offsets_count, 0)
        mapping = self.lolg._get_primitive_to_supercell_map()
        self.assertEqual(len(mapping), 0)

    def test_unique_offset_count(self):
        """
        Tests number of unique offsets corresponds to the number of atoms
        in the supercell given that there is one atom in the primitive cell.
        """
        self.assertEqual(self.lolg.get_number_of_unique_offsets(),
                         len(self.supercell))

    def test_get_primitive_to_supercell_map(self):
        """Tests primitive to supercell mapping."""
        unique_offsets = self.lolg._get_unique_primcell_offsets()

        for offset in unique_offsets:
            self.lolg.generate_local_orbit_list(offset)
            mapping = self.lolg._get_primitive_to_supercell_map()
            for sites_prim, sites_super in mapping.items():
                pos_super = self.supercell.get_position(sites_super)
                pos_prim = self.primitive.get_position(sites_prim)
                self.assertTrue(np.all(np.isclose(pos_super, pos_prim)))

    def test_unique_primcell_offsets(self):
        """
        Tests primitive offsets are unique and take to positions that
        match atoms positions in the supercell.
        """
        unique_offsets = self.lolg._get_unique_primcell_offsets()
        super_pos = self.supercell.positions

        for k, offset in enumerate(unique_offsets):
            pos_prim = self.primitive.get_position(LatticeSite(0, offset))
            self.assertTrue(
                np.any(np.isclose(pos_prim, pos) for pos in super_pos))
            for i in range(k + 1, len(unique_offsets)):
                self.assertFalse(np.all(np.isclose(offset, unique_offsets[i])))


class TestLocalOrbitListGeneratorHCP(unittest.TestCase):
    """
    Container for test of class functionality for hcp structure,
    which contains two atoms per unitcell.
    """
    def __init__(self, *args, **kwargs):
        super(TestLocalOrbitListGeneratorHCP, self).__init__(*args, **kwargs)
        prim_atoms = bulk('Ni', 'hcp', a=4.0)
        cutoffs = [4.2, 4.2]
        self.orbit_list = OrbitList(prim_atoms, cutoffs)
        self.primitive = self.orbit_list.get_primitive_structure()
        super_atoms = make_supercell(prim_atoms, [[2, 0, 1000],
                                                  [0, 2, 0],
                                                  [0, 0, 2]])
        self.supercell = Structure.from_atoms(super_atoms)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiate class for each test case."""
        self.lolg = LocalOrbitListGenerator(self.orbit_list, self.supercell)

    def test_generate_local_orbit_list(self):
        """
        Tests that function generates an orbit list for the given
        offset of the primitive structure.
        """
        unique_offsets = self.lolg._get_unique_primcell_offsets()
        for offset in unique_offsets:
            local_orbit_list = self.lolg.generate_local_orbit_list(offset)
            for orbit_prim, orbit_super in zip(self.orbit_list.orbits,
                                               local_orbit_list.orbits):
                for site_p, site_s in zip(orbit_prim.representative_sites,
                                          orbit_super.representative_sites):
                    site_p.unitcell_offset += offset
                    pos_super = self.supercell.get_position(site_s)
                    pos_prim = self.primitive.get_position(site_p)
                    self.assertTrue(np.all(np.isclose(pos_super, pos_prim)))

    def test_unique_offset_count(self):
        """
        Tests number of unique offsets corresponds to half of the total number
        of atoms in the supercell given that there is two atoms per unitcell.
        """
        self.assertEqual(self.lolg.get_number_of_unique_offsets(),
                         len(self.supercell) / 2)

    def test_get_primitive_to_supercell_map(self):
        """Tests primitive to supercell mapping."""
        unique_offsets = self.lolg._get_unique_primcell_offsets()

        for offset in unique_offsets:
            self.lolg.generate_local_orbit_list(offset)
            mapping = self.lolg._get_primitive_to_supercell_map()
            for sites_prim, sites_super in mapping.items():
                pos_super = self.supercell.get_position(sites_super)
                pos_prim = self.primitive.get_position(sites_prim)
                self.assertTrue(np.all(np.isclose(pos_super, pos_prim)))

    def test_unique_primcell_offsets(self):
        """
        Tests primitive offsets are unique and take to positions that
        match atoms positions in the supercell.
        """
        unique_offsets = self.lolg._get_unique_primcell_offsets()
        super_pos = self.supercell.positions

        for k, offset in enumerate(unique_offsets):
            pos_prim = self.primitive.get_position(LatticeSite(0, offset))
            self.assertTrue(
                np.any(np.isclose(pos_prim, pos) for pos in super_pos))
            for i in range(k + 1, len(unique_offsets)):
                self.assertFalse(np.all(np.isclose(offset, unique_offsets[i])))


if __name__ == '__main__':
    unittest.main()
