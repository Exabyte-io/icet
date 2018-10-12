from icet.core_py.orbit_list import OrbitList


class ClusterSpace(object):
    """
    This class provides functionality for generating and maintaining cluster
    spaces.

    Parameters
    ----------
    atoms : ASE Atoms object
    cutoffs : list of floats
        cutoff radii per order that define the cluster space
    chemical_symbols : list of strings
        list of chemical symbols, each of which must map to an element of
        the periodic table
    Mi : list / dictionary / int
        * if a list is provided, it must contain as many elements as there
          are sites and each element represents the number of allowed
          species on the respective site
        * if a dictionary is provided the key represent the site index and
          the value the number of allowed components
        * if a single `int` is provided each site the number of allowed
          components will be set to `Mi` for sites in the structure

    TODO
    ----
    * Add a method to retrieve a cluster vector
    * When orbitlist is complete:
        * Store OrbitList as a property/attribute
        * Add all the print orbit stuff
        * Fix Mi (allowed components on each site)
        * Write all empty methods that currently are
          only implemented with `pass`
    """

    def __init__(self, atoms, cutoffs, chemical_symbols,
                 Mi=None):

        self._structure = atoms
        self._cutoffs = cutoffs
        self._chemical_symbols = chemical_symbols

        # set up orbit list
        orbit_list = OrbitList(self._structure, self._cutoffs)
        orbit_list.sort()
        self._orbit_list = orbit_list
        self._mi = Mi

    def __repr__(self):
        """ String representation. """
        return self._get_string_representation(print_threshold=50)

    def print_overview(self, print_threshold=None, print_minimum=10):
        """
        Print an overview of the cluster space in terms of the orbits (order,
        radius, multiplicity etc).

        Parameters
        ----------
        print_threshold : int
            if the number of orbits exceeds this number print dots
        print_minimum : int
            number of lines printed from the top and the bottom of the orbit
            list if `print_threshold` is exceeded
        """
        print(self._get_string_representation(print_threshold=print_threshold,
                                              print_minimum=print_minimum))

    def get_cluster_vector(self, atoms):
        """
        Returns the cluster vector for a structure.

        Parameters
        ----------
        atoms : ASE Atoms object / icet Structure object (bi-optional)
            atomic configuration

        Returns
        -------
        NumPy array
            the cluster vector
        """
        pass

    @property
    def structure(self):
        """
        icet Structure object : structure used for initializing the cluster
        space
        """
        return self._structure

    @property
    def cutoffs(self):
        """ list : cutoffs used for initializing the cluster space """
        return self._cutoffs

    @property
    def chemical_symbols(self):
        """ list of species considered in the cluster space """
        return self._chemical_symbols

    @property
    def orbit_list(self):
        """
        Return orbit list object
        """
        return self._orbit_list

    def get_orbit(self, index):
        """
        Return orbit with index from
        orbit list

        parameters
        ----------
        index : int
        """
        return self.orbit_list.orbits[index]

    def get_cluster_space_info(self, index):
        pass

    def __len__(self):
        pass
