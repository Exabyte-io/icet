import numpy as np
import ase.db
from icet import ClusterSpace


class ClusterExpansion(object):
    '''
    Cluster expansion model

    Attributes
    ----------
    cluster_space : ClusterSpace object
        cluster space that was used for constructing the cluster expansion
    parameters : list of floats
        effective cluster interactions (ECIs)
    '''

    def __init__(self, cluster_space, parameters):
        '''
        Initialize a ClusterExpansion object.

        Parameters
        ----------
        cluster_space : ClusterSpace object
            the cluster space to be used for constructing the cluster expansion
        parameters : list of floats
            effective cluster interactions (ECIs)
        '''
        self._cluster_space = cluster_space
        self._parameters = parameters

    def predict(self, structure):
        '''
        Predict the property of interest (e.g., the energy) for the input
        structure using the cluster expansion.

        Parameters
        ----------
        structure : ASE Atoms object / icet Structure (bi-optional)
            atomic configuration

        Returns
        -------
        float
            property value predicted by the cluster expansion
        '''
        cluster_vector = self.cluster_space.get_cluster_vector(structure)
        prop = np.dot(cluster_vector, self.parameters)
        return prop

    @property
    def cluster_space(self):
        '''ClusterSpace object : cluster space the cluster expansion is
        based on'''
        return self._cluster_space

    @property
    def parameters(self):
        '''list of floats : effective cluster interactions (ECIs)'''
        return self._parameters

    def write(self, filename):
        """
        Write Cluster expansion to file.

        Parameters
        ---------
        filename : str with filename to saved
        cluster space.
        """

        self.cluster_space.write(filename)
        db = ase.db.connect(filename, type='db')
        entry = db.get(id=1)
        atoms = entry.toatoms()
        cutoffs = entry.data.cutoffs
        chemical_symbols = entry.data.chemical_symbols
        Mi = entry.data.Mi
        verbosity = entry.data.verbosity

        db = ase.db.connect(filename, type='db', append=False)
        db.write(atoms, data={'cutoffs': cutoffs,
                              'chemical_symbols': chemical_symbols,
                              "Mi": Mi,
                              'verbosity': verbosity,
                              'parameters': self.parameters})

    @staticmethod
    def read(filename):
        """
        Read cluster expansion from file.

        Parameters
        ---------
        filename : str with filename to saved
        cluster space.
        """
        cs = ClusterSpace.read(filename)
        db = ase.db.connect(filename, type='db')
        entry = db.get(id=1)
        parameters = entry.data.parameters
        return ClusterExpansion(cs, parameters)
