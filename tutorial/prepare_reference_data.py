# import modules
from ase.build import bulk
from ase.db import connect
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from icet.tools import enumerate_structures

# step 1: Prepare database and set up basic structure
db = connect('structures.db')
prim = bulk('Au')
subelements = ['Ag', 'Au']

# step 2: Enumerate structures, then relax and add them to the database
sizes = range(1, 7)
for k, atoms in enumerate(enumerate_structures(prim, sizes, subelements)):
    # skip if structure is already present in database
    if any(db.select(structure_id=k)):
        continue
    # remember the original (unrelaxed) positions
    original_positions = atoms.get_positions()
    # relax the structure
    atoms.calc = EMT()
    dyn = BFGS(atoms)
    dyn.run(fmax=0.01)
    # add the structure to the database
    db.write(atoms, structure_id=k,
             data={'original_positions': original_positions})
