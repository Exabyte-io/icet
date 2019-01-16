from ase.build import make_supercell
from numpy import arange, array
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import VCSGCEnsemble

# step 1: set up the structure to simulate and the calculator
ce = ClusterExpansion.read('mixing_energy.ce')
chemical_symbols = ce.cluster_space.chemical_symbols[0]
atoms = make_supercell(ce.cluster_space.primitive_structure,
                       3 * array([[-1, 1, 1],
                                  [1, -1, 1],
                                  [1, 1, -1]]))
# TODO: Remove this line once atoms is not longer decorated with H atoms
atoms.numbers = [47] * len(atoms)

calculator = ClusterExpansionCalculator(atoms, ce)

# step 2: carry out Monte Carlo simulations
for temperature in [900, 300]:
    # Evolve configuration through the entire composition range
    for phi in arange(-2.1, 0.11, 0.1):
        # Initialize MC ensemble
        mc = VCSGCEnsemble(
            atoms=atoms,
            calculator=calculator,
            temperature=temperature,
            data_container='monte-carlo-data/vcsgc-T{}-phi{:+.3f}.dc'.format(
                temperature, phi),
            phis={chemical_symbols[0]: -2.0 - phi,
                  chemical_symbols[1]: phi},
            kappa=200)

        mc.run(number_of_trial_steps=len(atoms) * 30)
