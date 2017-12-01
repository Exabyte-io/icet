#!/usr/bin/env python3

'''
This file contains unit tests and other tests. It can be executed by
simply executing this file from a shell prompt:
    
    $ ./test_structure_container.py
    
In which case it will use the system's default python version. If a specific
python version should be used, run that python version with this file as input,
e.g.:
    
    python3 test_structure_container.py

For a description of the python unit testing framework, see this link:
https://docs.python.org/3/library/unittest.html

When executing this file doc testing is also performed on all doc tests in 
the structure_container.py file

'''

import unittest

from icetdev import ClusterSpace, StructureContainer
from ase.build import bulk
from ase.calculators.emt import EMT
from ase import Atoms

subelements = ['Ag', 'Au']
cutoffs = [4.0] * 3
atoms_prim = bulk('Ag')
atoms_supercell = atoms_prim.repeat(2)

cs = ClusterSpace(atoms_supercell, cutoffs, subelements)

atoms_list=[]

# structure #1
conf_1 = atoms_supercell.copy()
atoms_list.append(conf_1)

# structure #2
conf_2 = atoms_supercell.copy()
conf_2[0].symbol = 'Au'
atoms_list.append(conf_2)

# structure #3
conf_3 = atoms_supercell.copy()
conf_3[0].symbol = 'Au'
conf_3[1].symbol = 'Au'
atoms_list.append(conf_3)

calc = EMT()
properties = []

for conf in atoms_list:
    conf.set_calculator(calc)
    conf.properties = {'energy':conf.get_potential_energy()/len(conf), 
                       'volume':conf.get_volume()}
    properties.append(conf.properties)

class TestStructureContainer(unittest.TestCase):
    '''
    Container for tests of the class functionality
    '''

    def setUp(self):
        '''
        Instantiate class before each test.

        '''
        self.sc = StructureContainer(cs, atoms_list, properties)

    def test_init(self):
        '''
        Just testing that the setup
        (initialization) of tested class work

        '''
        self.sc = StructureContainer(cs, atoms_list, properties)

    def test_len(self):
        '''
        Testing len functionality

        '''
        number_structures = self.sc.__len__()
        self.assertEqual(number_structures, len(atoms_list))

    def test_getitem(self):
        '''
        Testing getitem functionality

        '''
        structure = self.sc.__getitem__(1)
        self.assertIsNotNone(structure)

    def test_get_structure_indices(self):
        '''
        Testing get_structure_indices functionality

        '''
        list_index = [x for x in range(len(atoms_list))]
        self.assertEqual(self.sc.get_structure_indices(), list_index)

    def test_add_structure(self):
        '''
        Testing add_structure functionality

        '''
        #structure #4
        conf_4 = atoms_supercell.copy()
        conf_4[0].symbol = 'Au'
        conf_4[1].symbol = 'Au'
        conf_4[2].symbol = 'Au'
        conf_4.set_calculator(calc)
        conf_4.get_potential_energy()

        self.sc.add_structure(conf_4)

        list_index = [x for x in range(len(atoms_list)+1)]
        new_indices = self.sc.get_structure_indices()
        self.assertEqual(new_indices, list_index)

    def test_get_fit_data(self):
        '''
        Testing get_fit_data functionality
        '''
        clustervectors, properties = self.sc.get_fit_data()
        self.assertTrue(isinstance(prop, float) for prop in properties)
        self.assertTrue(isinstance(cv, float) for cv in clustervectors)


    def test_repr(self):
        '''
        Testing repr functionality

        '''
        retval = self.sc.__repr__()
        target = """------------------- Structure Container --------------------
Total number of structures: 3
index |   user_tag   | natoms | fit_ready | energy | volume 
------------------------------------------------------------
   0  | None         |     8  |   True    | 0.002  | 136.836
   1  | None         |     8  |   True    | -0.001 | 136.836
   2  | None         |     8  |   True    | -0.003 | 136.836"""
        self.assertEqual(target, retval)


    def test_get_properties(self):
        '''
        Testing get_properties functionality

        '''
        p_list = self.sc.get_properties()
        self.assertTrue(isinstance(properties, float) for properties in p_list)

    def test_add_properties(self):
        '''
        Testing load_properties functionality

        '''
        extra_properties = []
        for conf in atoms_list:
            extra_properties.append({'total_energy': conf.get_total_energy()})

        self.sc.add_properties(properties=extra_properties)
        p_list = self.sc.get_properties(key='total_energy')
        self.assertTrue(isinstance(properties, float) for properties in p_list)

    def test_get_structure(self):
        '''
        Testing get_structures functionality
        '''
        s_list = self.sc.get_structure()
        self.assertTrue(isinstance(atoms, Atoms) for atoms in s_list)

    def test_get_cluster_space(self):
        '''
        Testing get_clusterspace functionality
        '''
        cs_onlyread = self.sc.get_cluster_space
        self.assertEqual(cs_onlyread, cs)



def suite():
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestStructureContainer)
    return test_suite


if __name__ == '__main__':
    unittest.main()
