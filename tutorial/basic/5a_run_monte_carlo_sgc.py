from ase.build import make_supercell
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
import numpy as np
from os import mkdir

# step 1: Set up structure to simulate as well as calculator
ce = ClusterExpansion.read('mixing_energy.ce')
atoms = make_supercell(ce.cluster_space.primitive_structure,
                       3 * np.array([[-1, 1, 1],
                                     [1, -1, 1],
                                     [1, 1, -1]]))
calculator = ClusterExpansionCalculator(atoms, ce)

# step 2: Carry out Monte Carlo simulations
# Make sure output directory exists
output_directory = 'monte_carlo_data'
try:
    mkdir(output_directory)
except FileExistsError:
    pass
for temperature in [900, 300]:
    # Evolve configuration through the entire composition range
    for dmu in np.arange(-0.7, 0.51, 0.05):
        # Initialize MC ensemble
        mc = SemiGrandCanonicalEnsemble(
            atoms=atoms,
            calculator=calculator,
            temperature=temperature,
            data_container='{}/sgc-T{}-dmu{:+.3f}.dc'
                           .format(output_directory, temperature, dmu),
            chemical_potentials={'Ag': 0, 'Pd': dmu})

        mc.run(number_of_trial_steps=len(atoms) * 30)
