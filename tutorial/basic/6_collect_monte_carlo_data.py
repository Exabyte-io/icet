import pandas as pd
from glob import glob
from mchammer import DataContainer

# step 1: Collect data from SGC and VCSGC simulations
for ensemble in ['sgc', 'vcsgc']:
    data = []
    for filename in glob('monte_carlo_data/{}-*.dc'.format(ensemble)):
        dc = DataContainer.read(filename)
        data_row = dc.ensemble_parameters
        data_row['filename'] = filename
        n_atoms = data_row['n_atoms']

        equilibration = 5 * n_atoms
        data_row['Pd_concentration'] = \
            dc.get_average('Pd_count', start=equilibration) / n_atoms
        data_row['mixing_energy'] = \
            dc.get_average('potential', start=equilibration) / n_atoms
        data_row['acceptance_ratio'] = \
            dc.get_average('acceptance_ratio', start=equilibration)
        if ensemble == 'sgc':
            data_row['free_energy_derivative'] = \
                dc.ensemble_parameters['mu_Pd'] - \
                dc.ensemble_parameters['mu_Ag']
        elif ensemble == 'vcsgc':
            data_row['free_energy_derivative'] = \
                dc.get_average('free_energy_derivative', start=equilibration)

        data.append(data_row)

    # step 2: Write data to pandas dataframe in csv format
    df = pd.DataFrame(data)
    df.to_csv('monte-carlo-{}.csv'.format(ensemble), sep='\t')
