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
    cluster_space : icet.ClusterSpace
        cluster space from which the allowed species are extracted

    sites : dict(str, list(int))
        dictionary containing lists of sites that are to be considered,
        which keys will be taken as the names of the sites

    super_cell : ase.Atoms
        an atoms object that represents a typical super cell, which is used to
        determine the allowed species

    interval : int
        observation interval during the Monte Carlo simulation

    Attributes
    ----------
    tag : str
        name of observer

    interval : int
        observation interval

    Example
    -------
    The following snippet illustrate how to use the site occupancy factor (SOF)
    observer in a Monte Carlo simulation of a surface slab. Here, the SOF
    observer is used to monitor the concentrations of different species at the
    surface, the first subsurface layer, and the remaining "bulk". A minimal
    cluster expansion is used with slightly modified surface interactions in
    order to obtain an example that can be run without much ado. In practice,
    one should of course use a proper cluster expansion::

        from ase.build import fcc111
        from icet import ClusterExpansion, ClusterSpace
        from mchammer.calculators import ClusterExpansionCalculator
        from mchammer.ensembles import CanonicalEnsemble
        from mchammer.observers import SiteOccupancyObserver

        # prepare reference structure
        prim = fcc111('Au', size=(1, 1, 10), vacuum=10.0)
        prim.translate((0.1, 0.1, 0.0))
        prim.wrap()
        prim.pbc = True  # icet requires pbc in all directions

        # prepare cluster expansion
        cs = ClusterSpace(prim, cutoffs=[3.7], chemical_symbols=['Ag', 'Au'])
        params = [0] + 5 * [0] + 10 * [0.1]
        params[1] = 0.01
        params[6] = 0.12
        ce = ClusterExpansion(cs, params)
        print(ce)

        # prepare initial configuration based on a 2x2 supercell
        atoms = prim.repeat((2, 2, 1))
        for k in range(20):
            atoms[k].symbol = 'Ag'

        # set up MC simulation
        calc = ClusterExpansionCalculator(atoms, ce)
        mc = CanonicalEnsemble(atoms=atoms, calculator=calc, temperature=600,
                               data_container='myrun_sof.dc')

        # set up observer and attach it to the MC simulation
        sites = {'surface': [0, 9], 'subsurface': [1, 8],
                 'bulk': list(range(2, 8))}
        sof = SiteOccupancyObserver(cs, sites, atoms, interval=len(atoms))
        mc.attach_observer(sof)

        # run 1000 trial steps
        mc.run(1000)

    After having run this snippet one can access the SOFs via the data
    container::

        print(mc.data_container.data)
    """

    def __init__(self, cluster_space: ClusterSpace,
                 sites: Dict[str, List[int]],
                 super_cell: Atoms,
                 interval: int = None) -> None:
        super().__init__(interval=interval, return_type=dict,
                         tag='SiteOccupancyObserver')

        self._sites = {site: sorted(indices)
                       for site, indices in sites.items()}

        self._set_allowed_species(cluster_space, super_cell)

    def _set_allowed_species(self,
                             cluster_space: ClusterSpace,
                             super_cell: Atoms):
        """
        Set the allowed species for the selected sites in the atoms object

        Parameters
        ----------
        atoms
            input atomic structure.
        """

        primitive_structure = Structure.from_atoms(
            cluster_space.primitive_structure)
        chemical_symbols = cluster_space.get_chemical_symbols()

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
                positions = super_cell.get_positions()[np.array(indices)]
                lattice_sites =\
                    primitive_structure.find_lattice_sites_by_positions(
                        positions)
                for l, lattice_site in enumerate(lattice_sites):
                    species = chemical_symbols[lattice_site.index]
                    # check that the allowed species are equal for all sites
                    if allowed_species[site] is not None and\
                            species != allowed_species[site]:
                        raise Exception("The allowed species {} for the site"
                                        " with index {} differs from the"
                                        " result {} for the previous index"
                                        " ({})!".format(species, indices[l],
                                                        allowed_species[site],
                                                        indices[l-1]))
                    allowed_species[site] = species

        self._allowed_species = allowed_species

    def get_observable(self, atoms: Atoms) -> Dict[str, List[float]]:
        """
        Returns the site occupation factors for a given atomic configuration.

        Parameters
        ----------
        atoms
            input atomic structure.
        """

        chemical_symbols = np.array(atoms.get_chemical_symbols())
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
