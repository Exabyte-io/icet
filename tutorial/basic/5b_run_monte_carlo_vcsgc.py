from ase.build import make_supercell
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import VCSGCEnsemble
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
    for phi in arange(-2.1, 0.11, 0.1):
        # Initialize MC ensemble
        mc = VCSGCEnsemble(
            atoms=atoms,
            calculator=calculator,
            temperature=temperature,
            data_container='monte_carlo_data/vcsgc-T{}-phi{:+.3f}.dc'
                           .format(temperature, phi),
            phis={chemical_symbols[0]: -2.0 - phi, chemical_symbols[1]: phi},
            kappa=200)

        mc.run(number_of_trial_steps=len(atoms) * 30)
