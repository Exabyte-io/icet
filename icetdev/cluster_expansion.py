import numpy as np


class ClusterExpansion(object):
    '''
    Cluster expansion model

    Attributes
    ----------
    clusterspace : icet ClusterSpace object
        the cluster space that was used for constructing the cluster expansion
    parameters : list of floats
        effective cluster interactions (ECIs)
    '''

    def __init__(self, clusterspace, parameters):
        '''
        Initialize a ClusterExpansion object.

        Parameters
        ----------
        clusterspace : icet ClusterSpace object
            the cluster space to be used for constructing the cluster expansion
        parameters : list of floats
            effective cluster interactions (ECIs)
        '''
        self._clusterspace = clusterspace
        self._parameters = parameters

    def predict(self, structure):
        '''
        Use the cluster expansion to predict the property of interest
        (e.g., the energy) for the input structure.

        Parameters
        ----------
        structure : icet Structure or ASE Atoms object
            input structure

        Returns
        -------
        float
            property value predicted by the cluster expansion
        '''

        clustervector = self.clusterspace.get_clustervector(structure)
        prop = np.dot(clustervector, self.parameters)
        return prop

    @property
    def clusterspace(self):
        '''icet ClusterSpace object : cluster space the cluster expansion is
        based on'''
        return self._clusterspace

    @property
    def parameters(self):
        '''list of floats : effective cluster interactions (ECIs)'''
        return self._parameters
