from ase import Atoms
from ase.build import bulk, fcc111
from ase.cluster.icosahedron import Icosahedron
from ase.db import connect

"""
Database containing different type of structures used for testing.
Some tests may use partially the following database.
"""

""" FCC (single element, primitive cell, pbc=True) """
atoms_1 = bulk('Al','fcc', a=1.0)

""" FCC (single element, cubic structure, pbc=True) """
atoms_2 = bulk('Al', 'fcc', a=1.0).repeat(2)

""" FCC (single element, distorted structure, pbc=True) """
atoms_3 = bulk('Al', 'fcc', a=1.0).repeat(2).rattle(stdev=0.01, seed=42)

""" BCC (two elements, cubic structure, pbc=True) """
atoms_4 = bulk('Ti','bcc', a=1.0).repeat(2)
for atom in atoms_4:
    if atom.index % 2 == 0:
        atom.symbol='W'

""" rocksalt (two elements, cubic structure) """
atoms_5 = bulk('NaCl', 'rocksalt', a=1.0).repeat(1)

""" HCP (single element, hexagonal structure) """
atoms_6 = bulk('Ni', 'hcp', a=0.625, c=1.0).repeat(1)

""" perovskite (three elements, cubic structure) """
a=1.0
b=a/2
atoms_7 = Atoms('BaZrO3', positions=[(0, 0, 0), (b,b,b), 
                (b,b,0), (b,0,b), (0,b,b)], cell=[a, a, a], pbc=True).repeat(1)

""" surface slab (two elements, pbc=[True, True, False]) """
atoms_8 = fcc111('Pd', a=1.0, size=(2,2,3))
atoms_8.center(vacuum=4.0, axis=2)

""" Nanoparticle (single element, pbc=False) """
lc=1.0
atoms_9 = Icosahedron('Au', noshells=3, latticeconstant=lc)
atoms_9.center(vacuum=6.0, axis=(0, 1, 2))


db = connect('structures_for_testing.db')

db.write(atoms_1, tag='Al-fcc-primitive_cell')
db.write(atoms_2, tag='Al-fcc-supercell')
db.write(atoms_3, tag='Al-fcc-distorted')
db.write(atoms_4, tag='WTi-bcc-cubic_cell')
db.write(atoms_5, tag='NaCl-rocksalt-cubic-cell')
db.write(atoms_6, tag='Ni-hcp-hexagonal-cell')
db.write(atoms_7, tag='BaZrO3-perovskite')
db.write(atoms_8, tag='Pd-slab-surface')
db.write(atoms_9, tag='Au-nanoparticle')