from ase.build import bulk
from ase.db import connect
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from icet.tools import enumerate_structures

# step 1: Prepare database and set up basic structure
db = connect('structures.db')
prim = bulk('Ag')
species = ['Ag', 'Au']

# step 2: Enumerate structures, then relax and add them to the database
sizes = range(1, 7)
for k, atoms in enumerate(enumerate_structures(prim, sizes, species)):
    # skip if structure is already present in database
    if any(db.select(structure_id=k)):
        continue
    # remember the original (unrelaxed) cell metric and positions
    original_cell = atoms.get_cell()
    original_positions = atoms.get_positions()
    # relax the structure
    atoms.calc = EMT()
    dyn = BFGS(atoms)
    dyn.run(fmax=0.01)
    # store energy
    relaxed_energy = atoms.get_potential_energy()
    # restore ideal cell metric and positions
    atoms.set_cell(original_cell)
    atoms.set_positions(original_positions)
    # add the structure to the database
    db.write(atoms, structure_id=k, relaxed_energy=relaxed_energy)

# step 3: get reference energies for the species
eref = {}
for elem in species:
    for row in db.select('{}=1'.format(elem), natoms=1):
        eref[elem] = row.relaxed_energy / row.natoms
        break

# step 4: compute mixing energies and add them to the database
for row in db.select():
    conc = float(row.count_atoms().get('Ag', 0)) / row.natoms
    emix = row.relaxed_energy / row.natoms
    emix -= conc * eref['Ag'] + (1.0 - conc) * eref['Au']
    db.update(row.id, emix=emix, conc=conc)
