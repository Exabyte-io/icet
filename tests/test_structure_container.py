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
from icetdev.structure_container import FitStructure
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
    conf.properties = {'energy':conf.get_potential_energy(), 
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
        TODO: 
            This unit test is not completely isolated since  
            `get_structure_indices` method is being called here.        
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
        clustervectors, target_properties = self.sc.get_fit_data()
        self.assertTrue(isinstance(prop, float) for prop in target_properties)
        self.assertTrue(isinstance(cv, float) for cv in clustervectors)


    def test_repr(self):
        '''
        Testing repr functionality

        '''
        retval = self.sc.__repr__()
        target = """------------- Structure Container --------------
Total number of structures: 3
index |   user_tag   | natoms | energy | volume 
------------------------------------------------
   0  | None         |     8  | 0.013  | 136.836
   1  | None         |     8  | -0.007 | 136.836
   2  | None         |     8  | -0.026 | 136.836"""
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

    def test_cluster_space(self):
        '''
        Testing clusterspace functionality
        '''
        cs_onlyread = self.sc.cluster_space
        self.assertEqual(cs_onlyread, cs)


class TestFitStructure(unittest.TestCase):
    def setUp(self):
        '''
        Instantiate class before each test
        '''
        cv = cs.get_clustervector(conf_1)
        prop = properties[0]
        self.fit_structure = FitStructure(conf_1, "conf_1", cv, prop)

    def test_init(self):
        '''
        Just testing that the setup
        (initialization) of tested class work
        '''
        self.fit_structure = FitStructure(conf_1, None)

    def test_clustervector(self):
        '''
        Testing clustervector attribute
        '''
        cv  = self.fit_structure.clustervector
        self.assertTrue(all(isinstance(val, float) for val in cv))

    
    def test_atoms(self):
        '''
        Testing atoms attribute
        '''
        atoms  = self.fit_structure.atoms
        self.assertTrue(isinstance(atoms, Atoms))

    def test_user_tag(self):
        '''
        Testing user_tag attribute
        '''
        user_tag = self.fit_structure.user_tag
        self.assertTrue(isinstance(user_tag, str))
        
    def test_properties(self):
        '''
        Testing properties attribute
        '''
        properties_dict = self.fit_structure.properties
        self.assertTrue(isinstance(properties_dict, dict))

    def test_set_properties(self):
        '''
        Testing set_properties functionality
        '''
        extra_prop = {'total_energy': conf_1.get_total_energy()}
        self.fit_structure.set_properties(extra_prop)
        properties_dict = self.fit_structure.properties
        self.assertTrue(isinstance(properties_dict, dict))

    def test_set_clustervector(self):
        '''
        Testing set_clustervector functionality
        '''
        self.fit_structure.set_clustervector(None)
        cv  = self.fit_structure.clustervector
        self.assertTrue(cv is None)


def suite():
    test_classes_to_run = [TestStructureContainer, TestFitStructure]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    test_suite = unittest.TestSuite(suites_list)
    return test_suite


if __name__ == '__main__':
    unittest.main()
