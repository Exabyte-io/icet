'''
This test checks that a structure container object can be initialized
with a list of random structures given as ASE-Atoms objects along with
properties and label tags
'''

from icetdev import ClusterSpace, StructureContainer
from ase.build import bulk
from ase import Atoms
import numpy as np


subelements = ['Ag', 'Au']
cutoffs = [4.0] * 3
atoms_prim = bulk('Ag')

atoms_supercell = atoms_prim.repeat(3)


def random_configuration(atoms_supercell, subelements):
    '''
    Generate a random structure with atoms_supercell as a base
    and fill it randomly with elements in subelements

    '''

    from ase.calculators.emt import EMT
    import random
    random.seed(23)
    conf = atoms_supercell.copy()
    for atom in conf:
        elem = random.choice(subelements)
        atom.symbol = elem

    calc = EMT()
    conf.set_calculator(calc)
    conf.get_total_energy()
    conf.keys = {'energy':conf.get_total_energy(), 'efermi':0.1, 'eLUMO':3.0}

    formula = conf.get_chemical_formula()
    tag = '{}'.format(formula)
    conf.tag = tag

    return conf


atoms_list = []
number_of_structures = 10

for index in range(number_of_structures):
    conf = random_configuration(atoms_supercell, subelements)
    atoms_list.append(conf)

cs = ClusterSpace(atoms_prim, cutoffs, subelements)
sc = StructureContainer(cs, atoms_list)

# testing total number of structures
assert sc.get_number_of_structures() == number_of_structures

# testing structure getter
assert isinstance(sc.get_structures(conf.tag), Atoms)

# testing clustervector 
cv_target = np.array(cs.get_clustervector(conf))
cv = sc.get_clustervector(conf.tag)
assert np.all(np.abs(cv_target - cv) < 1e-6)

# testing list of available properties
en = sc.get_properties('energy', conf.tag)
assert en == conf.keys['energy']
