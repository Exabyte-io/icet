'''
Test structure numeration by checking that it yields the correct
number of structure.
'''

from icetdev.enumeration import enumerate_structures
from ase.build import bulk
from ase import Atom


def count_structures(atoms, sizes, species, correct_count, tag):
    '''
    Count structures given by structure enumeration and assert that the
    right number is given.

    Paramters
    ---------
    atoms : ASE Atoms
        Primitive structure for the enumeration.
    sizes : list of ints
        Cell sizes to be included in the enumeration.
    species : list of str
        Species to be passed to the enumeration.
    correct_count : int
        Expected number of structures.
    tag : str
        Describes the structure.
    '''
    count = 0
    for _ in enumerate_structures(atoms, sizes, species):
        count += 1
    msg = 'Structure enumeration failed for {}'.format(tag)
    assert count == correct_count, msg

tag = 'FCC, 3 elements'
atoms = bulk('Au', crystalstructure='fcc')
species = ['Au', 'Pd', 'Cu']
sizes = range(1, 7)
correct_count = 1081
count_structures(atoms, sizes, species, correct_count, tag)

tag = 'FCC, elongated cell, two sites'
atoms = bulk('Au', crystalstructure='fcc', a=4.0)
cell = atoms.cell
cell[0] = 1.33*cell[0]
atoms.cell = cell
atoms.append(Atom('H', (2.0, 2.0, 2.0)))
species = [['Au', 'Pd'], ['H', 'V']]
sizes = range(1, 5)
correct_count = 1500
count_structures(atoms, sizes, species, correct_count, tag)

tag = 'HCP'
atoms = bulk('Au', crystalstructure='hcp', a=4.0)
species = ['Au', 'Pd']
sizes = range(1, 7)
correct_count = 5777
count_structures(atoms, sizes, species, correct_count, tag)
