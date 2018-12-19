import matplotlib.pyplot as plt
import pandas as pd

# step 1: Load data frame
df = pd.read_csv('sgc_collected_data.csv', delimiter='\t')

fig, ax = plt.subplots(figsize=(4, 3.5))
for T in sorted(df.temperature.unique()):
    df_T = df.loc[df['temperature'] == T].sort_values('Pd_concentration')
    ax.plot(df_T['Pd_concentration'], 1e3 * df_T['mu_Pd'],
            marker='o', markersize=2.5, label='{} K'.format(T))
ax.set_xlabel('Pd concentration')
ax.set_ylabel('Chemical potential difference (meV/atom)')
ax.set_xlim([-0.02, 1.02])
ax.legend()
plt.savefig('chemical_potential_difference.png', bbox_inches='tight')

# step 2: Plot mixing energy vs composition
fig, ax = plt.subplots(figsize=(4, 3.5))
for T in sorted(df.temperature.unique()):
    df_T = df.loc[df['temperature'] == T].sort_values('Pd_concentration')
    ax.plot(df_T['Pd_concentration'], 1e3 * df_T['mixing_energy'],
            marker='o', markersize=2.5, label='{} K'.format(T))
ax.set_xlabel('Pd concentration')
ax.set_ylabel('Mixing energy (meV/atom)')
ax.set_xlim([-0.02, 1.02])
ax.legend()
plt.savefig('mixing_energy.png', bbox_inches='tight')

# step 3: Plot acceptance ratio vs composition
fig, ax = plt.subplots(figsize=(4, 3.5))
for T in sorted(df.temperature.unique()):
    df_T = df.loc[df['temperature'] == T].sort_values('Pd_concentration')
    ax.plot(df_T['Pd_concentration'], df_T['acceptance_ratio'],
            marker='o', markersize=2.5, label='{} K'.format(T))
ax.set_xlabel('Pd concentration')
ax.set_ylabel('Acceptance ratio')
ax.set_xlim([-0.02, 1.02])
ax.legend()
plt.savefig('acceptance_ratio.png', bbox_inches='tight')