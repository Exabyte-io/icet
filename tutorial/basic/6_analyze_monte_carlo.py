from glob import glob
from mchammer import DataContainer
import matplotlib.pyplot as plt
import pandas as pd

# step 1: Collect data
data = {}
for fname in sorted(glob('sgc*.dc')):
    dc = DataContainer.read(fname)
    row = dc.data.T[0]
    nsites = row['Pd_count'] + row['Ag_count']

    temperature = row['temperature']
    if temperature not in data:
        data[temperature] = {'dmu': [], 'concentration': [],
                             'mixing_energy': [], 'acceptance_ratio': []}

    dmu = row['mu_Pd'] - row['mu_Ag']
    data[temperature]['dmu'].append(dmu)

    nequil = nsites * 10
    conc = dc.get_average('Pd_count', start=nequil)[0] / nsites
    data[temperature]['concentration'].append(conc)

    emix = dc.get_average('potential', start=nequil)[0] / nsites
    data[temperature]['mixing_energy'].append(emix)

    accratio = dc.get_average('acceptance_ratio', start=nequil)[0]
    data[temperature]['acceptance_ratio'].append(accratio)

# step 2: Plot chemical potential difference vs composition
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.set_xlabel('Pd concentration')
ax.set_ylabel('Chemical potential difference (meV/atom)')
ax.set_xlim([-0.02, 1.02])
for temperature, series in reversed(sorted(data.items())):
    series = pd.DataFrame.from_dict(series).sort_values('dmu')
    ax.plot(series['concentration'], 1e3 * series['dmu'],
            marker='o', markersize=2.5, label='{} K'.format(temperature))
plt.legend()
plt.savefig('chemical_potential_difference.png', bbox_inches='tight')

# step 3: Plot mixing energy
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.set_xlabel('Pd concentration')
ax.set_ylabel('Mixing energy (meV/atom)')
ax.set_xlim([-0.02, 1.02])
for temperature, series in reversed(sorted(data.items())):
    series = pd.DataFrame.from_dict(series).sort_values('dmu')
    ax.plot(series['concentration'], 1e3 * series['mixing_energy'],
            marker='o', markersize=2.5, label='{} K'.format(temperature))
plt.legend()
plt.savefig('mixing_energy.png', bbox_inches='tight')

# step 4: Plot acceptance ratio
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.set_xlabel('Pd concentration')
ax.set_ylabel('Acceptance ratio')
ax.set_xlim([-0.02, 1.02])
for temperature, series in reversed(sorted(data.items())):
    series = pd.DataFrame.from_dict(series).sort_values('dmu')
    ax.plot(series['concentration'], series['acceptance_ratio'],
            marker='o', markersize=2.5, label='{} K'.format(temperature))
plt.legend()
plt.savefig('acceptance_ratio.png', bbox_inches='tight')
