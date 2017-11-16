from _icetdev import Neighborlist
from icetdev.structure import Structure
from ase import Atoms


def get_neighborlists(atoms, cutoffs=None):
    '''
    Return list of icet neigbhorlists from a configuration and cutoffs

    Parameters
    ----------
    atoms : ASE atoms object / icet structure object (bi-optional)
        atomic configuration
    cutoffs:
        positive floats indicating the cutoffs for the various clusters

    Returns
    -------
    list of neighborlists
    '''

    # deal with different types of structure objects
    if isinstance(atoms, Atoms):
        structure = Structure.from_atoms(atoms)
    elif isinstance(atoms, Structure):
        structure = atoms
    else:
        msg = ['Unknown structure format']
        msg += ['{} (ClusterSpace)'.format(type(atoms))]
        raise Exception(' '.join(msg))

    neighborlists = []
    if cutoffs is None:
        raise Exception('Both n and cutoffs is None in count clusters')
    else:
        for co in cutoffs:
            nl = Neighborlist(co)
            neighborlists.append(nl)

    # build the neighborlists
    for nl in neighborlists:
        nl.build(structure)

    return neighborlists
