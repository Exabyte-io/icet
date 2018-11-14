# -*- coding: utf-8 -*-
# This scripts runs in about 2 seconds on an i7-6700K CPU.

import matplotlib.pyplot as plt
from collections import OrderedDict
from icet import ClusterExpansion
from numpy import array, count_nonzero

# step 1: Collect ECIs in dictionary
ce = ClusterExpansion.read('mixing_energy.ce')
ecis = OrderedDict()
for order in range(len(ce.cluster_space.cutoffs)+2):
    for orbit in ce.cluster_space.orbit_data:
        if orbit['order'] != order:
            continue
        if order not in ecis:
            ecis[order] = {'radius': [], 'parameters': []}
        ecis[order]['radius'].append(orbit['radius'])
        ecis[order]['parameters'].append(ce.parameters[orbit['index']])

df_ecis = ce.parameters_as_dataframe

# step 2: Plot ECIs
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(7.5, 3))
for k, order in enumerate(ce.orders):
    df_order = df_ecis.loc[df_ecis['order'] == order]
    if k < 2 or k > 4:
        continue
    ax = axs[k-2]
    ax.set_ylim((-6, 39))
    ax.set_xlabel(r'Cluster radius (Å)')
    if order == 2:
        ax.set_xlim((1.2, 4.2))
        ax.set_ylabel(r'Effective cluster interaction (meV)')
    if order == 3:
        ax.set_xlim((1.5, 3.9))
    if order == 4:
        ax.set_xlim((1.5, 3.9))
        ax.text(0.05, 0.55, 'zerolet: {:.1f} meV'
                .format(1e3*df_order.eci.iloc[0]),
                transform=ax.transAxes)
        ax.text(0.05, 0.45, 'singlet: {:.1f} meV'
                .format(1e3*ecis[1]['parameters'][0]),
                transform=ax.transAxes)
    ax.plot([0, 5], [0, 0], color='black')
    ax.bar(df_order.radius, 1e3*df_order.eci, width=0.05)
    ax.scatter(df_order.radius, len(df_order) * [-5],
               marker='o', s=2.0)
    ax.text(0.05, 0.91, 'order: {}'.format(order),
            transform=ax.transAxes)
    ax.text(0.05, 0.81, '#parameters: {}'.format(len(df_order)),
            transform=ax.transAxes,)
    ax.text(0.05, 0.71, '#non-zero params: {}'
            .format(count_nonzero(df_order.eci)),
            transform=ax.transAxes,)
plt.savefig('ecis.png', bbox_inches='tight')
