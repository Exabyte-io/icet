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
import doctest

import structure_container as icet
from icetdev import ClusterSpace
from ase.build import bulk
from ase import Atoms


subelements = ['Ag', 'Au']
cutoffs = [4.0] * 3
atoms_prim = bulk('Ag')
atoms_supercell = atoms_prim.repeat(3)

cs = ClusterSpace(atoms_supercell, cutoffs, subelements)


def _random_configuration(atoms_supercell, subelements):
    '''
    Generate random structures by filling randomly a supercell
    with elements in subelement list and then calculate the 
    energy using ASE EMT calculator.

    '''
    from ase.calculators.emt import EMT
    import random
    atoms_copy = atoms_supercell.copy()
    for atom in atoms_copy:
        elem = random.choice(subelements)
        atom.symbol = elem

    calc = EMT()
    atoms_copy.set_calculator(calc)
    atoms_copy.get_total_energy()
    atoms_copy.properties = {'energy':atoms_copy.get_potential_energy()/len(atoms_copy), 
                             'volumen':atoms_copy.get_volume()}
    atoms_copy.total_energy = {'total_energy': atoms_copy.get_total_energy()}


    return atoms_copy

atoms_list = []
properties = []
properties_to_add = []
number_configurations = 10

for i in range(number_configurations):
    conf = _random_configuration(atoms_supercell, subelements)
    atoms_list.append(conf)
    properties.append(conf.properties)
    properties_to_add.append(conf.total_energy)


class TestIceT(unittest.TestCase):
    '''
    Base class for testing the icet classes
    '''
    pass


class TestStructureContainer(TestIceT):
    '''
    Container for tests of the class functionality
    '''
    
    def setUp(self):
        '''
        Instantiate class before each test.

        '''
        self.sc = icet.StructureContainer(cs, atoms_list, properties)
    
    def test_init(self):
        '''
        Just testing that the setup 
        (initialization) of tested class work

        '''
        self.sc = icet.StructureContainer(cs, atoms_list, properties)

    def test_len(self):
        '''
        Testing len functionality

        '''
        number_structures = self.sc.__len__()
        self.assertEqual(number_structures, number_configurations)

    def test_getitem(self):
        '''
        Testing getitem functionality

        TODO: Improve this test
        '''
        structure = self.sc.__getitem__(1)
        self.assertIsNotNone(structure)

    def test_get_structure_indices(self):
        '''
        Testing get_structure_indices functionality

        '''
        list_index = [x for x in range(number_configurations)]
        self.assertEqual(self.sc.get_structure_indices(), list_index)

    def test_add_structure(self):
        '''
        Testing add_structure functionality

        '''
        new_structure = _random_configuration(atoms_supercell, subelements)
        self.sc.add_structure(new_structure)
        
        list_index = [x for x in range(number_configurations+1)]
        new_indices = self.sc.get_structure_indices()
        self.assertEqual(new_indices,list_index)

    def test_repr(self):
        '''
        Testing repr functionality

        TODO: Improve this test 

        '''
        retval = self.sc.__repr__()
        print('Printing repr\n{}'.format(retval))
        self.assertIn('energy',retval)

    def test_get_properties(self):
        '''
        Testing gte_properties functionality
        
        '''
        p_list = self.sc.get_properties()
        self.assertTrue(isinstance(properties,float) for properties in p_list)

    def test_load_properties(self):
        '''
        Testing load_properties functionality

        '''
        self.sc.load_properties(properties=properties_to_add)
        p_list = self.sc.get_properties(key='total_energy')
        self.assertTrue(isinstance(properties,float) for properties in p_list)

    def test_get_structures(self):
        '''
        Testing get_strcutures functionality
        '''
        s_list = self.sc.get_structures()
        self.assertTrue(isinstance(atoms, Atoms) for atoms in s_list)

    def test_get_cluster_space(self):
        '''
        Testing get_clusterspace functionality
        '''
        sc_cs = self.sc.get_cluster_space
        self.assertEqual(sc_cs, cs)

if __name__ == '__main__':

    def load_tests(loader, tests, ignore):
        tests.addTests(doctest.DocTestSuite(icet))
        return tests
    
    unittest.main()