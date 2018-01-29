'''
This scripts checks the computation of cluster vectors with a variable number
of components on different sites.
'''

import numpy as np
from ase.db import connect
from icet import Structure, ClusterSpace, get_singlet_info
from icet.tools import get_primitive_structure


def test_mi_int_list_and_dict(atoms, subelements, cutoffs, allowed_sites):
    '''
    Test that int Mi, list Mi and dict Mi produces the same cluster vectors.
    '''

    prim_size = len(get_primitive_structure(atoms))

    Mi_int = allowed_sites
    Mi_list = [allowed_sites] * prim_size
    Mi_dict = {}

    singlet_data = get_singlet_info(atoms.copy())
    for singlet in singlet_data:
        Mi_dict[singlet['orbit index']] = allowed_sites

    cluster_space_int = ClusterSpace(atoms, cutoffs, subelements,  Mi=Mi_int)
    cluster_space_list = ClusterSpace(atoms, cutoffs, subelements, Mi=Mi_list)
    cluster_space_dict = ClusterSpace(atoms, cutoffs, subelements, Mi=Mi_dict)

    atoms_prim = cluster_space_int.get_primitive_structure().to_atoms()

    # create and populate a supercell and get cluster vector
    repeat = [1] * 3
    for i, pbc in enumerate(atoms_prim.pbc):
        if pbc:
            repeat[i] = 3
    conf = atoms_prim.repeat(repeat)
    for at in conf:
        at.symbol = np.random.choice(subelements)

    conf = Structure.from_atoms(conf)

    cv_int = cluster_space_int.get_cluster_vector(conf)
    cv_list = cluster_space_list.get_cluster_vector(conf)
    cv_dict = cluster_space_dict.get_cluster_vector(conf)

    msg = 'Cluster vectors were not equal for an equivalent Mi (int vs list)'
    assert (np.array(cv_int) == np.array(cv_list)).all, msg

    msg = 'Cluster vectors were not equal for an equivalent Mi (int vs dict)'
    assert (np.array(cv_int) == np.array(cv_dict)).all, msg


print('')
db = connect('structures_for_testing.db')
subelements = ['H', 'He', 'Pb']
for row in db.select():
    print(' structure: {}'.format(row.tag))
    atoms_row = row.toatoms()
    atoms_tag = row.tag
    cutoffs = [1.4] * 3
    if len(atoms_row) == 0:
        continue
    if atoms_row.get_pbc().all():
        atoms_row.wrap()
    atoms_row.set_chemical_symbols(len(atoms_row) * [atoms_row[0].symbol])
    for allowed_sites in range(2, 4):
        test_mi_int_list_and_dict(atoms_row, subelements,
                                  cutoffs, allowed_sites)
