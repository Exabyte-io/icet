from mchammer import DataContainer
import pandas as pd
from glob import glob

# step 1: Collect data
data = []
for fname in glob('sgc*.dc'):
    dc = DataContainer.read(fname)
    data_row = dc.ensemble_parameters
    data_row['fname'] = fname
    n_atoms = data_row['n_atoms']
    n_equil = 10 * n_atoms

    data_row['Pd_concentration'] = \
        dc.get_average('Pd_count', start=n_equil) / n_atoms
    data_row['mixing_energy'] = \
        dc.get_average('potential', start=n_equil) / n_atoms
    data_row['acceptance_ratio'] = \
        dc.get_average('acceptance_ratio', start=n_equil)
    data.append(data_row)

# step 2: Save data
df = pd.DataFrame(data)
df.to_csv('sgc_collected_data.csv', sep='\t')
