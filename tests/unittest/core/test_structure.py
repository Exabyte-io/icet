import unittest


class TestStructure(unittest.TestCase):
    '''
    Container for test of the module functionality.
    Test is done againts ASE Atoms object
    '''

    def setUp(self):
        '''
        SetUp
        '''
        pass

    def test_structure_from_atoms(self):
        '''
        Test that Structure object is returned form an
        ASE Atoms object
        '''
        pass

    def test_structure_to_atoms(self):
        '''
        Test that Structure is returned as an ASE
        Atoms object
        '''
        pass

    def test_repr_function(self):
        '''
        Test representation of a Structure object
        '''
        pass


if __name__ == '__main__':
    unittest.main()
