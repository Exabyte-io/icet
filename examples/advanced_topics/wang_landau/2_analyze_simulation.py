import matplotlib.pyplot as plt
import numpy as np
from mchammer import WangLandauDataContainer
from mchammer.data_containers import (get_average_observables_wl,
                                      get_density_of_states_wl)

# Get density and entropy
dc = WangLandauDataContainer.read('wl_n16.dc')
df, _ = get_density_of_states_wl(dc)

# Plot density
_, ax = plt.subplots()
ax.semilogy(df.energy, df.density, marker='o')
ax.set_xlabel('Energy')
ax.set_ylabel('Density of states')
plt.savefig(f'wang_landau_density.svg', bbox_inches='tight')

# Compute thermodynamic averages
df = get_average_observables_wl(dc,
                                temperatures=np.arange(0.4, 6, 0.05),
                                observables=['sro_Ag_1'],
                                boltzmann_constant=1)

# Plot heat capacity and short-range order parameter
n_sites = dc.ensemble_parameters['n_atoms']
_, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(df.temperature, n_sites * df.potential_std ** 2 / df.temperature ** 2)
axes[0].set_xlabel('Temperature')
axes[0].set_ylabel('Heat capacity')
axes[1].plot(df.temperature, df.sro_Ag_1_mean, label='mean')
axes[1].plot(df.temperature, df.sro_Ag_1_std, label='stddev')
axes[1].set_xlabel('Temperature')
axes[1].set_ylabel('Short-range order parameter')
axes[1].legend()
plt.subplots_adjust(hspace=0)
plt.savefig(f'wang_landau_heat_capacity_sro.svg', bbox_inches='tight')
