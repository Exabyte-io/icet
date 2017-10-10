from __future__ import print_function, division
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.build import bulk
import numpy as np
import pdb 

class manybodyNeighborlistTester():

    """
    This is a python implementation of icet's  manybody neighborlist.
    This is used both as a tester of the c++ version and also as a venue
    to try out new functionalities.

    Functionalities are similar to ASE neigbhorlist but is extended
    for more connections than for pairs.

    Bothways = False will mean that if you collect all neighbors over all indices
    then you will not have any duplicates. I.e. in total you will only generate
    i,j,k and never i,k,j.
    To make sure of this the indices you generate will always be sorted that is:
    i < j < k  etc

    Bothways = True will mean that you will return every neighbor of an indice.
    However you will not return both i,j,k and i,k,j.
    Contrary to bothways = False this is allowed: i>j, i>k etc.. (but always j < k)

    """

    def __init__(self):
        self.initiated = True

    def build(self, neighborlists, index, bothways=False):
        """
        Will take the neighborlist object (nl) and combine the neighbors
        of index "index" up to order "order".

        Args:
        neighborlists : list of ASE neighborlists
        index: index to return neighbors from neighborlist
        bothways: False will return all indices that are bigger than site "index"
                  True will return also return indices that are smaller.
        """
        if not isinstance(neighborlists, list):
            neighborlists = [neighborlists]

        if not neighborlists:
            raise RuntimeError(
                "Neighborlists is empty in manybodyNeighborlistTester::build ")

        manybody_neighbor_indices = []

        self.add_singlet(index, manybody_neighbor_indices)

        self.add_pairs(index, neighborlists[0],
                       manybody_neighbor_indices, bothways)

        """ Add neighbors of higher order (k>=2) """
        for k in range(2, len(neighborlists) + 2):

            """ Get neighbors of index in icet format """
            ngb = self.get_ngb_from_nl(neighborlists[k - 2], index)

            number_of_sites = len(neighborlists[k - 2].positions)

            zero_vector = np.array([0., 0., 0., ])

            current_original_neighbors = [[index, zero_vector]]

            self.combine_to_higher_order(
                neighborlists[k - 2], manybody_neighbor_indices, ngb,
                current_original_neighbors, bothways, k)

        return manybody_neighbor_indices

    def combine_to_higher_order(self, nl, manybody_neighbor_indices, ngb_i, current_original_neighbors, bothways, order):
        """
        For each j in ngb construct the intersect of ngb_j and ngb, call the intersect ngb_ij.
        All neighbors in ngb_ij are then neighbors with i and j
        What is saved then is (i,j) and ngb_ij up to the desired order "order"

        Parameters:
        nl : ASE neighborlist object
        manybody_neighbor_indices: list of neighborlists, each inner list is made up.
        ngb_i : list of neighbors of a chosen index in the icet format [[index,offset]]
        current_original_neighbors :
        bothways : False will return all indices that are bigger than site "index"
                   True will return also return indices that are smaller.
        order : highest order for manybody neighbor indices.
        """
        for j in ngb_i:
            
            originalNeighborCopy = current_original_neighbors.copy()

            # explain this line here
            if len(originalNeighborCopy) == 1 and (not bothways and
                                                   self.compare_neighbors(j, originalNeighborCopy[-1])):
                continue

            originalNeighborCopy.append(j)

            ngb_j_offset = self.translate_all_ngb(
                self.get_ngb_from_nl(nl, j[0]), j[1])

            if not bothways:
                ngb_j_offset = self.filter_ngb_from_smaller(ngb_j_offset, j)

            intersection_with_j = self.get_intersection(ngb_i, ngb_j_offset)

            if len(originalNeighborCopy) + 1 < order:
                self.combine_to_higher_order(
                    nl, manybody_neighbor_indices, intersection_with_j, originalNeighborCopy, bothways, order)

            if len(intersection_with_j) > 0 and len(originalNeighborCopy) == (order - 1):
                manybody_neighbor_indices.append(
                    [originalNeighborCopy, intersection_with_j])

    def add_singlet(self, index, mbn_indices):
        """
        Add singlet to manybody neighbor indices
        """
        offset = [0., 0., 0.]
        mbn_indices.append([index, offset])

    def add_pairs(self, index, neigbhorlist, mbn_indices, bothways):
        """
        Add pairs from neigbhorlist to manybody neighbor indices
        """
        offset = [0., 0., 0.]
        first_site = [index, offset]
        ngb = self.get_ngb_from_nl(neigbhorlist, index)
        if not bothways:
            ngb = self.filter_ngb_from_smaller(ngb, first_site)
        if len(ngb) == 0:
            return
        mbn_indices.append([first_site, ngb])


    def get_intersection(self, ngb_i, ngb_j):
        """
        Return intersection of ngb_i with ngb_j using the is_j_in_ngb bool method
        """
        ngb_ij = []
        for j in ngb_i:
            if self.is_j_in_ngb(j, ngb_j):
                ngb_ij.append(j)
        return ngb_ij

    def is_j_in_ngb(self, j, ngb):
        """
        Returns true if there is an index in ngb that is equal to j
        """
        for k in ngb:
            if k[0] == j[0] and (k[1] == j[1]).all():
                return True
        return False

    def filter_ngb_from_smaller(self, ngb_i, j):
        """
        Returns all k in ngb_i that are bigger than j
        """
        ngb_j_filtered = []
        for k in ngb_i:
            if self.compare_neighbors(j, k):
                ngb_j_filtered.append(k)
        return ngb_j_filtered

    def translate_all_ngb(self, ngb, offset):
        """
        Make a copy of ngb and returns ngb but with all offset "offseted" an addition offset
        """
        ngb_i_offset = ngb.copy()
        for j in ngb_i_offset:
            j[1] += offset
        return ngb_i_offset

    def compare_arrays(self, arr1, arr2):
        """
        Compares two arrays.
        Compares element by element in order.
        Returns true if arr1 < arr2
        """
        assert len(arr1) == len(arr2)
        for i in range(len(arr1)):
            if arr1[i] < arr2[i]:
                return True
            if arr1[i] > arr2[i]:
                return False
        return False

    def compare_neighbors(self, ngb_1, ngb_2):
        """
        Compare two neighbors as defined in this class
        Returns True if ngb_1 < ngb_2
        Returns False otherwise
        """
        if ngb_1[0] < ngb_2[0]:
            return True
        if ngb_1[0] > ngb_2[0]:
            return False
        return self.compare_arrays(ngb_1[1], ngb_2[1])

    def get_ngb_from_nl(self, ase_nl, index):
        """
        Get the neighbors of index in the format used in icet
        """
        indices, offsets = ase_nl.get_neighbors(index)
        ngb = []
        for ind, offs in zip(indices.copy(), offsets.copy()):
            ngb.append([ind, offs])
        return ngb
