from ase.build import make_supercell
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
import numpy as np
from multiprocessing import Pool

# step 1: Set up structure to simulate as well as calculator
ce = ClusterExpansion.read('../basic/mixing_energy.ce')
chemical_symbols = ce.cluster_space.chemical_symbols[0]
atoms = make_supercell(ce.cluster_space.primitive_structure,
                       3 * np.array([[-1, 1, 1],
                                     [1, -1, 1],
                                     [1, 1, -1]]))
atoms.set_chemical_symbols([chemical_symbols[0]] * len(atoms))
calculator = ClusterExpansionCalculator(atoms, ce)

# step 2: Define a function that handles MC run of one set of parameters
def run_mc(args):
    temperature = args['temperature']
    dmu = args['dmu']    
    mc = SemiGrandCanonicalEnsemble(
        atoms=atoms,
        calculator=calculator,
        temperature=temperature,
        data_container='sgc-T{}-dmu{:+.3f}.dc'
                       .format(temperature, dmu),
        chemical_potentials={chemical_symbols[0]: 0,
                             chemical_symbols[1]: dmu})
    mc.run(number_of_trial_steps=len(atoms) * 30)

# step 3: Define all sets of parameters to be run
args = []
for temperature in range(600, 199, -100):
    for dmu in np.arange(-0.6, 0.6, 0.05):
        args.append({'temperature': temperature,
                     'dmu': dmu})

# step 4: Define a Pool object with the desired number of processes and run
pool = Pool(processes=4)
pool.map(run_mc, args)
