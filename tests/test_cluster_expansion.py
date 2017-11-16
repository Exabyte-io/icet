'''
This test checks that a cluster expansion model can be initialized
with any structure in the test database and can predict a property.
'''

from ase.db import connect
from icetdev import ClusterSpace, ClusterExpansion


def test_clusterexpansion_model(atoms, cutoffs, subelements):
    '''
    Test clusterexpansion init and prediction

    Parameters
    ----------
    atoms : ASE Atoms object
        atomic configuration
    subelements : list of strings (chemical symbols)
        list of elements that are allowed
    '''
    cs = ClusterSpace(atoms, cutoffs, subelements)
    params_len = cs.get_clusterspace_size() + 1  # plus one for singlet
    params = list(range(params_len))

    ce = ClusterExpansion(cs, params)
    predicted_val = ce.predict(atoms)

    assert isinstance(predicted_val, float), 'Prediction is not of float type'


print('')
db = connect('structures_for_testing.db')
subelements = ['H', 'He', 'Pb']
cutoffs = [1.4] * 3
for row in db.select():
    print(' structure: {}'.format(row.tag))
    atoms_row = row.toatoms()
    atoms_row.set_chemical_symbols(len(atoms_row) * [subelements[0]])
    test_clusterexpansion_model(atoms_row, cutoffs, subelements)
