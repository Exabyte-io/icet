from icet.core.lattice_site import LatticeSite

import unittest


class TestLatticeSite(unittest.TestCase):
    '''
    Container for test of the module functionality.
    '''

    def setUp(self):
        '''
        SetUp
        '''
        pass

    def test_hash(self):
        '''
        Test that lattice site is hashable
        '''
        index = 1
        unitcell_offset = [0., 0., 0.]
        lattice_site = LatticeSite(index, unitcell_offset)
        lattice_site_map = {}
        lattice_site_map[lattice_site] = 1
        self.assertIn(lattice_site, lattice_site_map)


if __name__ == '__main__':
    unittest.main()
