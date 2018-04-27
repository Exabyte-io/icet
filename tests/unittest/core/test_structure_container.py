#!/usr/bin/env python3

"""
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

"""

import unittest
import tempfile

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from icet import ClusterSpace, StructureContainer
from icet.core.structure_container import FitStructure


def strip_surrounding_spaces(input_string):
    """
    Helper function that removes both leading and trailing spaces from a
    multi-line string.

    Returns
    -------
    str
        original string minus surrounding spaces and empty lines
    """
    from io import StringIO
    s = []
    for line in StringIO(input_string):
        if len(line.strip()) == 0:
            continue
        s += [line.strip()]
    return '\n'.join(s)


class TestStructureContainer(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestStructureContainer, self).__init__(*args, **kwargs)
        self.subelements = ['Ag', 'Au']
        self.cutoffs = [4.0] * 3
        self.atoms_prim = bulk('Ag', a=4.09)
        self.structure_list = []
        self.user_tags = []
        for k in range(4):
            atoms = self.atoms_prim.repeat(2)
            symbols = [self.subelements[0]] * len(atoms)
            symbols[:k] = [self.subelements[1]] * k
            atoms.set_chemical_symbols(symbols)
            self.structure_list.append(atoms)
            self.user_tags.append('Structure {}'.format(k))

        self.cs = ClusterSpace(self.atoms_prim, self.cutoffs, self.subelements)
        self.properties_list = []
        self.add_properties_list = []
        for k, atoms in enumerate(self.structure_list):
            atoms.set_calculator(EMT())
            properties = {'energy': atoms.get_potential_energy(),
                          'volume': atoms.get_volume()}
            self.properties_list.append(properties)
            add_properties = {'total_energy': atoms.get_total_energy()}
            self.add_properties_list.append(add_properties)

    def setUp(self):
        """
        Instantiate class before each test.
        """
        self.sc = StructureContainer(self.cs,
                                     list(zip(self.structure_list,
                                              self.user_tags)),
                                     self.properties_list)

    def test_init(self):
        """
        Just testing that the setup
        (initialization) of tested class work
        """
        sc = StructureContainer(self.cs, self.structure_list,
                                self.properties_list)
        self.assertIsInstance(sc, StructureContainer)
        # with only list of atoms (properties read from calc)
        sc = StructureContainer(self.cs, self.structure_list)
        self.assertIsInstance(sc, StructureContainer)
        # add atoms along with tags
        structure_list_with_tags = []
        for k, atoms in enumerate(self.structure_list, start=1):
            structure_list_with_tags.append((atoms, 'struct{}'.format(k)))
        sc = StructureContainer(self.cs, structure_list_with_tags,
                                self.properties_list)
        self.assertIsInstance(sc, StructureContainer)
        # check that other types fails
        with self.assertRaises(AssertionError) as context:
            sc = StructureContainer(self.cs, ['something'])
        msg = 'atoms has not ASE Atoms format'
        self.assertTrue(msg in str(context.exception))
        # check empty initialization
        sc = StructureContainer(self.cs)
        self.assertIsInstance(sc, StructureContainer)

    def test_len(self):
        """
        Testing len functionality
        """
        len_structure_container = self.sc.__len__()
        self.assertEqual(len_structure_container, len(self.structure_list))

    def test_getitem(self):
        """
        Testing getitem functionality
        """
        structure = self.sc.__getitem__(1)
        self.assertIsNotNone(structure)

    def test_get_structure_indices(self):
        """
        Testing get_structure_indices functionality
        """
        list_index = [x for x in range(len(self.structure_list))]
        self.assertEqual(self.sc.get_structure_indices(), list_index)

    def test_add_structure(self):
        """
        Testing add_structure functionality
        """
        atoms = self.structure_list[0].copy()
        properties = self.properties_list[0]
        tag = "struct5"
        self.sc.add_structure(atoms, tag, properties)
        self.assertEqual(self.sc.__len__(), len(self.structure_list)+1)
        # check that adding structures without properties fails
        with self.assertRaises(AssertionError):
            self.sc.add_structure(atoms)

    def test_get_fit_data(self):
        """
        Testing get_fit_data functionality
        """
        import numpy as np
        cluster_vectors, properties = self.sc.get_fit_data()
        # testing outputs have ndarray type
        self.assertIsInstance(cluster_vectors, np.ndarray)
        self.assertIsInstance(properties, np.ndarray)
        # testing values of cluster_vectors and properties
        for atoms, cv in zip(self.structure_list, cluster_vectors):
            retval = list(cv)
            target = list(self.cs.get_cluster_vector(atoms))
            self.assertAlmostEqual(retval, target, places=9)
        for target, retval in zip(self.properties_list, properties):
            self.assertEqual(retval, target['energy'])
        # passing a list of indexes
        cluster_vectors, properties = self.sc.get_fit_data([0])
        retval = list(cluster_vectors[0])
        atoms = self.structure_list[0]
        target = list(self.cs.get_cluster_vector(atoms))
        self.assertAlmostEqual(retval, target, places=9)
        retval2 = properties[0]
        target2 = self.properties_list[0]
        self.assertEqual(retval2, target2['energy'])

    def test_repr(self):
        """
        Testing repr functionality
        """
        retval = self.sc.__repr__()
        target = """
============================= Structure Container =============================
Total number of structures: 4
-------------------------------------------------------------------------------
index |       user_tag        | natoms | chemical formula |  energy  |  volume
-------------------------------------------------------------------------------
   0  | Structure 0           |     8  | Ag8              |    0.013 |  136.836
   1  | Structure 1           |     8  | Ag7Au            |   -0.007 |  136.836
   2  | Structure 2           |     8  | Ag6Au2           |   -0.026 |  136.836
   3  | Structure 3           |     8  | Ag5Au3           |   -0.038 |  136.836
===============================================================================
"""
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_get_string_representation(self):
        """
        Testing _get_string_representation functionality
        """
        retval = self.sc._get_string_representation(print_threshold=2,
                                                    print_minimum=1)
        target = """
============================= Structure Container =============================
Total number of structures: 4
-------------------------------------------------------------------------------
index |       user_tag        | natoms | chemical formula |  energy  |  volume
-------------------------------------------------------------------------------
   0  | Structure 0           |     8  | Ag8              |    0.013 |  136.836
 ...
   3  | Structure 3           |     8  | Ag5Au3           |   -0.038 |  136.836
===============================================================================
"""
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_print_overview(self):
        """
        Testing print_overview functionality
        """
        # this runs the function but since the latter merely invokes another
        # function to print some information to stdout there is not much to
        # test here (other than the function not throwing an exception)
        self.sc.print_overview()

    def test_get_properties(self):
        """
        Testing get_properties functionality
        """
        p_list = self.sc.get_properties()
        self.assertTrue(isinstance(properties, float) for properties in p_list)
        # passing a list of indexes
        p_list = self.sc.get_properties([0])
        self.assertTrue(isinstance(properties, float) for properties in p_list)

    def test_add_properties(self):
        """
        Testing load_properties functionality
        """
        self.sc.add_properties([0], properties=[self.add_properties_list[0]])
        p_list = self.sc.get_properties([0], key='total_energy')
        self.assertTrue(isinstance(properties, float) for properties in p_list)
        # adding a list of properties
        self.sc.add_properties(properties=self.add_properties_list)
        p_list = self.sc.get_properties(key='total_energy')
        self.assertTrue(isinstance(properties, float) for properties in p_list)

    def test_get_structures(self):
        """
        Testing get_structures functionality
        """
        s_list = self.sc.get_structures()
        self.assertTrue(isinstance(atoms, Atoms) for atoms in s_list)
        # passing a list of indexes
        s_list = self.sc.get_structures([0])
        self.assertTrue(isinstance(atoms, Atoms) for atoms in s_list)

    def test_get_user_tags(self):
        """
        Testing get_user_tags function
        """
        target = ['Structure 0', 'Structure 1', 'Structure 2', 'Structure 3']
        self.assertEqual(target, self.sc.get_user_tags())

    def test_cluster_space(self):
        """
        Testing cluster space functionality
        """
        cs_onlyread = self.sc.cluster_space
        self.assertEqual(cs_onlyread, self.cs)

    def test_read_write(self):
        """
        Test the read write functionality.
        """
        temp_file = tempfile.NamedTemporaryFile()
        self.sc.write(temp_file.name)
        sc_read = self.sc.read(temp_file.name)

        self.assertEqual(len(self.sc), len(sc_read))
        self.assertEqual(self.sc.__str__(), sc_read.__str__())

        for fs, fs_read in zip(self.sc.fit_structures, sc_read.fit_structures):
            self.assertEqual(list(fs.cluster_vector),
                             list(fs_read.cluster_vector))
            self.assertEqual(fs.atoms, fs_read.atoms)
            self.assertEqual(fs.user_tag, fs_read.user_tag)
            self.assertEqual(fs.properties, fs_read.properties)


class TestFitStructure(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestFitStructure, self).__init__(*args, **kwargs)
        self.subelements = ['Ag', 'Au']
        self.cutoffs = [4.0] * 3
        self.atoms_prim = bulk('Ag', a=4.09)
        self.cs = ClusterSpace(self.atoms_prim, self.cutoffs, self.subelements)

    def setUp(self):
        """
        Instantiate class before each test
        """
        atoms = self.atoms_prim.repeat(2)
        prop = {'energy': 0.0126746}
        cv = self.cs.get_cluster_vector(atoms)
        tag = "struct1"
        self.fit_structure = FitStructure(atoms, tag, cv, prop)

    def test_init(self):
        """
        Just testing that the setup
        (initialization) of tested class work
        """
        atoms = self.atoms_prim.repeat(2)
        tag = "struct1"
        self.fit_structure = FitStructure(atoms, tag)

    def test_cluster_vector(self):
        """
        Testing cluster vector attribute
        """
        atoms = self.atoms_prim.repeat(2)
        cv_from_cluster_space = list(self.cs.get_cluster_vector(atoms))
        cv = list(self.fit_structure.cluster_vector)
        self.assertEqual(cv, cv_from_cluster_space)

    def test_atoms(self):
        """
        Testing atoms attribute
        """
        atoms = self.fit_structure.atoms
        self.assertTrue(isinstance(atoms, Atoms))

    def test_user_tag(self):
        """
        Testing user_tag attribute
        """
        user_tag = self.fit_structure.user_tag
        self.assertTrue(isinstance(user_tag, str))

    def test_properties(self):
        """
        Testing properties attribute
        """
        properties = self.fit_structure.properties
        self.assertTrue(isinstance(properties, dict))

    def test_set_properties(self):
        """
        Testing set_properties functionality
        """
        add_prop = {'total_energy': 0.0126746}
        self.fit_structure.set_properties(add_prop)
        properties = self.fit_structure.properties
        self.assertTrue(isinstance(properties, dict))
        prop_value = self.fit_structure.properties['total_energy']
        self.assertEqual(prop_value, add_prop['total_energy'])

    def test_set_cluster_vector(self):
        """
        Testing set_cluster_vector functionality
        """
        self.fit_structure.set_cluster_vector(None)
        cv = self.fit_structure.cluster_vector
        self.assertTrue(cv is None)


if __name__ == '__main__':
    unittest.main()
