import matplotlib.pyplot as plt
from mchammer import DataContainer
from mchammer.ensembles import get_averages_wang_landau, get_density_wang_landau
from numpy import arange

# Read data container
dc = DataContainer.read('wl_n16.dc')
print(dc.data)

# Plot density
df, _ = get_density_wang_landau(dc)
_, ax = plt.subplots()
ax.semilogy(df.energy, df.density, marker='o')
ax.set_xlabel('Energy')
ax.set_ylabel('Density of states')
plt.show()

# Compute thermodynamic averages
df = get_averages_wang_landau(dc, temperatures=arange(0.4, 6, 0.05),
                              boltzmann_constant=1)

# Plot heat capacity
_, ax = plt.subplots()
n_sites = dc.ensemble_parameters['n_atoms']
ax.plot(df.temperature, n_sites * df.potential_std ** 2 / df.temperature ** 2)
ax.set_xlabel('Temperature')
ax.set_ylabel('Heat capacity')
plt.show()
