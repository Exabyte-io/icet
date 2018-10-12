from ase.db import connect

from icet import (StructureContainer,
                  CrossValidationEstimator,
                  ClusterExpansion)

from mchammer.ensembles import SemiGrandCanonicalEnsemble
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.observers import ClusterExpansionObserver


# step 1: Read cluster expansion for mixing energies from file
ce_mix_energies = ClusterExpansion.read('mixing_energy.ce')
cs = ce_mix_energies.cluster_space
atoms = cs.primitive_structure.repeat(3)

# step 2: Parse the input structures and set up a structure container
db = connect('reference_data.db')
sc = StructureContainer(cluster_space=cs)
for row in db.select('natoms<=6'):
    sc.add_structure(atoms=row.toatoms(),
                     user_tag=row.tag,
                     properties={'lattice_parameter': row.lattice_parameter})

# step 3: Construct cluster expansion for lattice parameter
opt = CrossValidationEstimator(
    fit_data=sc.get_fit_data(key='lattice_parameter'), fit_method='lasso')
opt.validate()
opt.train()
ce_latt_param = ClusterExpansion(cluster_space=cs,
                                 parameters=opt.parameters)

# step 4: Set up the calculator and a canonical ensemble
calculator = ClusterExpansionCalculator(atoms=atoms,
                                        cluster_expansion=ce_mix_energies)
ensemble = \
    SemiGrandCanonicalEnsemble(calculator=calculator, atoms=atoms,
                               random_seed=42, temperature=900.0,
                               chemical_potentials={'Ag': 0, 'Pd': 0},
                               ensemble_data_write_interval=10)

# step 5: Attach observer and run
observer = ClusterExpansionObserver(cluster_expansion=ce_latt_param,
                                    interval=10)
ensemble.attach_observer(observer=observer, tag='lattice_parameter')
ensemble.run(number_of_trial_steps=1000)

# step 6: Print data
print(ensemble.data_container.data)
