'''
This example demonstrates how to enumerate structures, i.e. how to
generate all inequivalent structures derived from a primitive
structure up to a certain size.
'''

# Start import
from ase import Atom
from ase.build import bulk
from ase.db import connect

from icetdev.enumeration import enumerate_structures

# End import

# Generate all binary fcc structures with up to 6 atoms/cell
# and save them in a database
# Start binaries
atoms = bulk('Au')
db = connect('AuPd-fcc.db')
for structure in enumerate_structures(atoms, range(1, 7), ['Pd', 'Au']):
    db.write(structure)
# End binaries

# Enumerate all palladium hydride structures with up to 4 primtive
# cells (= up to 4 Pd atoms and between 0 and 4 H atoms). We want to
# specify that one site should always be Pd while the other can be
# either a hydrogen or a vacancy (vanadium will serve as our "vacancy")
# Start enumeration
a = 4.0
atoms = bulk('Au', a=a)
atoms.append(Atom('H', (a/2, a/2, a/2)))
species = [['Pd'], ['H', 'V']]
db = connect('PdHVac-fcc.db')
for structure in enumerate_structures(atoms, range(1, 5), species):
    db.write(structure)
# End enumeration
