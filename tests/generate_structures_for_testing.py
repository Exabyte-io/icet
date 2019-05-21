from ase import Atoms
from ase.build import bulk
from ase.db import connect


def create_database():
    """
    Database containing different type of structures used for testing.
    Some tests may use partially the following database.
    """

    db = connect('structures_for_testing.db', append=False)

    """ FCC (single element, primitive cell, pbc=True) """
    atoms = bulk('Al', 'fcc', a=1.0)
    db.write(atoms, tag='Al-fcc-primitive_cell')

    """ FCC (single element, supercell, pbc=True) """
    atoms = bulk('Al', 'fcc', a=1.0).repeat(2)
    db.write(atoms, tag='Al-fcc-supercell')

    """ FCC (single element, distorted structure, pbc=True) """
    atoms = bulk('Al', 'fcc', a=1.0).repeat(2)
    atoms.rattle(stdev=0.001, seed=42)
    db.write(atoms, tag='Al-fcc-distorted')

    """ BCC (two elements, cubic structure, pbc=True) """
    atoms = bulk('Ti', 'bcc', a=1.0).repeat(2)
    for atom in atoms:
        if atom.index % 2 == 0:
            atom.symbol = 'W'
    db.write(atoms, tag='WTi-bcc-supercell')

    """ rocksalt (two elements, cubic structure) """
    atoms = bulk('NaCl', 'rocksalt', a=1.0)
    db.write(atoms, tag='NaCl-rocksalt-cubic-cell')

    """ HCP (single element, hexagonal structure) """
    atoms = bulk('Ni', 'hcp', a=0.625, c=1.0)
    db.write(atoms, tag='Ni-hcp-hexagonal-cell')

    """ perovskite (three elements, cubic structure) """
    a = 1.0
    b = 0.5 * a
    atoms = Atoms('BaZrO3',
                  positions=[(0, 0, 0), (b, b, b),
                             (b, b, 0), (b, 0, b), (0, b, b)],
                  cell=[a, a, a], pbc=True)
    db.write(atoms, tag='BaZrO3-perovskite')


if __name__ == "__main__":
    create_database()
