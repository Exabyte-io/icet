from ase.build import make_supercell
from numpy import arange, array

from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble

# step 1: set up the structure to simulate and the calculator
ce = ClusterExpansion.read('mixing_energy.ce')
chemical_symbols = ce.cluster_space.chemical_symbols[0]
atoms = make_supercell(ce.cluster_space.primitive_structure,
                       3 * array([[-1, 1, 1],
                                 [1, -1, 1],
                                 [1, 1, -1]]))
# TODO: Remove this line once cs.primitive cell is not longer decorated
# with H atoms
atoms.numbers = [47]*len(atoms)

calculator = ClusterExpansionCalculator(atoms, ce)

# step 2: carry out Monte Carlo simulations
for temperature in [900, 600, 300]:
    # Evolve configuration through the entire composition range
    for dmu in arange(-0.6, 0.51, 0.05):
        # Initialize MC ensemble
        mc = SemiGrandCanonicalEnsemble(
            atoms=atoms,
            calculator=calculator,
            temperature=temperature,
            chemical_potentials={chemical_symbols[0]: 0,
                                 chemical_symbols[1]: dmu})

        mc.run(number_of_trial_steps=len(atoms) * 30)
        mc.data_container.write('sgc-T{}-dmu{:.3f}.dc'
                                .format(temperature, dmu))
