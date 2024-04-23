from ase import Atoms
from icet import ClusterSpace
from icet.core.structure import Structure
from mchammer.observers.base_observer import BaseObserver
from typing import List, Dict
import numpy as np


class SiteOccupancyObserver(BaseObserver):
    """
    This class represents a site occupation factor (SOF) observer.

    A SOF observer allows to compute the site occupation factors along the
    trajectory sampled by a Monte Carlo (MC) simulation.

    Parameters
    ----------
    cluster_space
        Cluster space from which the allowed species are extracted.
    structure
        Supercell consistent with primitive structure in :attr:`cluster_space`.
        Used to determine which species are allowed on each site.
    sites
        Dictionary containing lists of sites that are to be considered.
        The keys will be taken as the names of the sites; the indices refer
        to the primitive structure associated with the cluster space.
    interval
        Observation interval. Defaults to ``None`` meaning that if the
        observer is used in a Monte Carlo simulation, then the
        :class:`Ensemble` object will determine the interval.

    Example
    -------
    The following snippet illustrates how to use the site occupancy factor (SOF)
    observer in a Monte Carlo simulation of a surface slab. Here, the SOF
    observer is used to monitor the concentrations of different species at the
    surface, the first subsurface layer, and the remaining 'bulk'. A minimal
    cluster expansion is used with slightly modified surface interactions in
    order to obtain an example that can be run without much ado. In practice,
    one should of course use a proper cluster expansion::

        >>> from ase.build import fcc111
        >>> from icet import ClusterExpansion, ClusterSpace
        >>> from mchammer.calculators import ClusterExpansionCalculator
        >>> from mchammer.ensembles import CanonicalEnsemble
        >>> from mchammer.observers import SiteOccupancyObserver

        >>> # prepare reference structure
        >>> prim = fcc111('Au', size=(1, 1, 10), vacuum=10.0)
        >>> prim.translate((0.1, 0.1, 0.0))
        >>> prim.wrap()
        >>> prim.pbc = True  # icet requires pbc in all directions

        >>> # prepare cluster expansion
        >>> cs = ClusterSpace(prim, cutoffs=[3.7], chemical_symbols=['Ag', 'Au'])
        >>> params = [0] + 5 * [0] + 10 * [0.1]
        >>> params[1] = 0.01
        >>> params[6] = 0.12
        >>> ce = ClusterExpansion(cs, params)
        >>> print(ce)

        >>> # prepare initial configuration based on a 2x2 supercell
        >>> structure = prim.repeat((2, 2, 1))
        >>> for k in range(20):
        >>>     structure[k].symbol = 'Ag'

        >>> # set up MC simulation
        >>> calc = ClusterExpansionCalculator(structure, ce)
        >>> mc = CanonicalEnsemble(structure=structure, calculator=calc, temperature=600,
        ...                        dc_filename='myrun_sof.dc')

        >>> # set up observer and attach it to the MC simulation
        >>> sites = {'surface': [0, 9], 'subsurface': [1, 8],
        ...          'bulk': list(range(2, 8))}
        >>> sof = SiteOccupancyObserver(cs, structure, sites, interval=len(structure))
        >>> mc.attach_observer(sof)

        >>> # run 1000 trial steps
        >>> mc.run(1000)

    After having run this snippet one can access the SOFs via the data
    container::

        >>> print(mc.data_container.data)
    """

    def __init__(self, cluster_space: ClusterSpace,
                 structure: Atoms,
                 sites: Dict[str, List[int]],
                 interval: int = None) -> None:
        super().__init__(interval=interval, return_type=dict, tag='SiteOccupancyObserver')

        self._sites = {site: sorted(indices)
                       for site, indices in sites.items()}

        self._set_allowed_species(cluster_space, structure)

    def _set_allowed_species(self,
                             cluster_space: ClusterSpace,
                             structure: Atoms):
        """
        Set the allowed species for the selected sites in the Atoms object

        Parameters
        ----------
        cluster_space
            Cluster space implicitly defining allowed species.
        structure
            Specific supercell (consistent with cluster_space) whose
            allowed species are to be determined.
        """

        primitive_structure = Structure.from_atoms(cluster_space.primitive_structure)
        chemical_symbols = cluster_space.chemical_symbols

        if len(chemical_symbols) == 1:
            # If the allowed species are the same for all sites no loop is
            # required
            allowed_species = {site: chemical_symbols[0] for
                               site in self._sites.keys()}
        else:
            # Loop over the lattice sites to find the allowed species
            allowed_species = {}
            for site, indices in self._sites.items():
                allowed_species[site] = None
                positions = structure.positions[np.array(indices)]
                lattice_sites = primitive_structure.find_lattice_sites_by_positions(
                    positions=positions,
                    fractional_position_tolerance=cluster_space.fractional_position_tolerance)
                for k, lattice_site in enumerate(lattice_sites):
                    species = chemical_symbols[lattice_site.index]
                    # check that the allowed species are equal for all sites
                    if allowed_species[site] is not None and \
                            species != allowed_species[site]:
                        raise Exception('The allowed species {} for the site'
                                        ' with index {} differs from the'
                                        ' result {} for the previous index'
                                        ' ({})!'.format(species, indices[k],
                                                        allowed_species[site],
                                                        indices[k-1]))
                    allowed_species[site] = species

        self._allowed_species = allowed_species

    def get_observable(self, structure: Atoms) -> Dict[str, List[float]]:
        """
        Returns the site occupation factors for a given atomic configuration.

        Parameters
        ----------
        structure
            Input atomic structure.
        """

        chemical_symbols = np.array(structure.get_chemical_symbols())
        sofs = {}
        for site, indices in self._sites.items():
            counts = {species: 0 for species in self._allowed_species[site]}
            symbols, sym_counts = np.unique(chemical_symbols[indices],
                                            return_counts=True)
            for sym, count in zip(symbols, sym_counts):
                counts[sym] += count

            for species in counts.keys():
                key = 'sof_{}_{}'.format(site, species)
                sofs[key] = float(counts[species]) / len(indices)

        return sofs
