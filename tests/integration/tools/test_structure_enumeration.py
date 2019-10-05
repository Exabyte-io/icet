"""
Test structure numeration by checking that it yields the correct
number of structure.
"""

from ase.build import bulk, fcc100
from ase import Atom
from icet.tools import (enumerate_structures,
                        enumerate_supercells)


def count_structures(structure, sizes, species, correct_count, tag,
                     conc_rest=None):
    """
    Count structures given by structure enumeration and assert that the
    right number is given.

    Parameters
    ----------
    structure : ASE Atoms
        Primitive structure for the enumeration.
    sizes : list of ints
        Cell sizes to be included in the enumeration.
    species : list of str
        Species to be passed to the enumeration.
    correct_count : int
        Expected number of structures.
    tag : str
        Describes the structure.
    """
    count = 0
    for _ in enumerate_structures(structure, sizes, species,
                                  concentration_restrictions=conc_rest):
        count += 1
    msg = 'Structure enumeration failed for {}'.format(tag)
    assert count == correct_count, msg


tag = 'FCC, 3 elements'
structure = bulk('Au', crystalstructure='fcc')
species = ['Au', 'Pd', 'Cu']
sizes = range(1, 7)
correct_count = 1081
count_structures(structure, sizes, species, correct_count, tag)

tag = 'FCC, elongated cell, two sites'
structure = bulk('Au', crystalstructure='fcc', a=4.0)
cell = structure.cell
cell[0] = 1.33 * cell[0]
structure.cell = cell
structure.append(Atom('H', (2.0, 2.0, 2.0)))
species = [['Au', 'Pd'], ['H', 'V']]
sizes = range(1, 5)
correct_count = 1500
count_structures(structure, sizes, species, correct_count, tag)

tag = 'HCP'
structure = bulk('Au', crystalstructure='hcp', a=4.0)
species = ['Au', 'Pd']
sizes = range(1, 6)
correct_count = 984
count_structures(structure, sizes, species, correct_count, tag)

tag = 'Surface'
structure = fcc100('Au', (1, 1, 1), a=4.0, vacuum=2.0)
species = ['Au', 'Pd']
sizes = range(1, 9)
correct_count = 271
count_structures(structure, sizes, species, correct_count, tag)

tag = 'Chain'
structure = bulk('Au', a=4.0)
structure.set_pbc((False, False, True))
species = ['Au', 'Pd']
sizes = range(1, 9)
correct_count = 62
count_structures(structure, sizes, species, correct_count, tag)

tag = 'FCC, concentration restricted'
structure = bulk('Au', crystalstructure='fcc')
species = ['Au', 'Pd']
sizes = range(1, 9)
concentration_restrictions = {'Au': [0.0, 0.36]}
correct_count = 134
count_structures(structure, sizes, species, correct_count, tag,
                 conc_rest=concentration_restrictions)

tag = 'FCC'
msg = 'Supercell enumeration failed for {}'.format(tag)
structure = bulk('Au', crystalstructure='fcc')
count = len(list(enumerate_supercells(structure, [6])))
msg = 'Supercell enumeration failed for {}'.format(tag)
assert count == 10, msg

tag = 'FCC'
msg = 'Supercell enumeration failed for {}'.format(tag)
structure = bulk('Au', crystalstructure='fcc')
count = len(list(enumerate_supercells(structure, [6], niggli_reduce=False)))
msg = 'Supercell enumeration failed for {}'.format(tag)
assert count == 10, msg

tag = 'FCC'
msg = 'Supercell enumeration failed for {}'.format(tag)
structure = bulk('Au', crystalstructure='fcc')
count = len(list(enumerate_supercells(structure, range(0, 6))))
msg = 'Supercell enumeration failed for {}'.format(tag)
assert count == 18, msg
