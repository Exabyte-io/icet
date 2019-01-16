from ase.build import make_supercell
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
from numpy import arange, array

# step 1: Set up structure to simulate as well as calculator
ce = ClusterExpansion.read('mixing_energy.ce')
chemical_symbols = ce.cluster_space.chemical_symbols[0]
atoms = make_supercell(ce.cluster_space.primitive_structure,
                       3 * array([[-1, 1, 1],
                                  [1, -1, 1],
                                  [1, 1, -1]]))
atoms.set_chemical_symbols([chemical_symbols[0]] * len(atoms))
calculator = ClusterExpansionCalculator(atoms, ce)

# step 2: Carry out Monte Carlo simulations
for temperature in [900, 300]:
    # Evolve configuration through the entire composition range
    for dmu in arange(-0.6, 0.51, 0.05):
        # Initialize MC ensemble
        mc = SemiGrandCanonicalEnsemble(
            atoms=atoms,
            calculator=calculator,
            temperature=temperature,
            data_container='monte_carlo_data/sgc-T{}-dmu{:+.3f}.dc'
                           .format(temperature, dmu),
            chemical_potentials={chemical_symbols[0]: 0,
                                 chemical_symbols[1]: dmu})

        mc.run(number_of_trial_steps=len(atoms) * 30)
