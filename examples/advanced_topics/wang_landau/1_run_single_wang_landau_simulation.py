from ase import Atoms
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import WangLandauEnsemble

# Prepare cluster expansion
prim = Atoms('Au', positions=[[0, 0, 0]], cell=[1, 1, 10], pbc=True)
cs = ClusterSpace(prim, cutoffs=[1.01], chemical_symbols=['Ag', 'Au'])
ce = ClusterExpansion(cs, [0, 0, 2])

# Prepare initial configuration
structure = prim.repeat((4, 4, 1))
for k in range(8):
    structure[k].symbol = 'Ag'

# Set up and run MC simulation
calculator = ClusterExpansionCalculator(structure, ce)
mc = WangLandauEnsemble(structure=structure,
                        calculator=calculator,
                        energy_spacing=1,
                        data_container='wl_n16.dc')
mc.run(number_of_trial_steps=1000000)
