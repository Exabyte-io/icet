from copy import deepcopy
from icet.core.structure import Structure
from typing import List
from ase import Atoms
import copy


class Sublattice:
    """
    This class stores and provides information about a specific
    sublattice. A sublattice is always supercell specific since
    it contains lattice indices.

    Parameters
    ----------
    chemical_symbols
        the allowed species on this sublattice
    indices
        the lattice indices the sublattice consists of

    """

    def __init__(self, chemical_symbols: List[str], indices: List[int]):
        self._chemical_symbols = chemical_symbols
        self._indices = indices

    @property
    def chemical_symbols(self):
        return copy.deepcopy(self._chemical_symbols)

    @property
    def indices(self):
        return self._indices.copy()


class Sublattices:
    """
    This class stores and provides information about the sublattices
    of a structure.


    Parameters
    ----------
    allowed_species
        list of the allowed species on each site of the primitve
        structure. For example this can be the chemical_symbols from
        a cluster space
    primitive_structure
        the primitive structure the allowed species reference to
    structure
        the structure that the sublattices will be based on
    """

    def __init__(self, allowed_species: List[List[str]], primitive_structure: Atoms,
                 structure: Atoms):

        # sorted unique sites, this basically decides A, B, C... sublattices

        active_lattices = sorted([tuple(sorted(symbols))
                                  for symbols in allowed_species if len(symbols) > 1])
        inactive_lattices = sorted(
            [tuple(sorted(symbols)) for symbols in allowed_species if len(symbols) == 1])
        self._allowed_species = active_lattices + inactive_lattices

        cpp_prim_structure = Structure.from_atoms(primitive_structure)
        self._sublattices = []
        sublattice_to_indices = [[] for _ in range(len(self._allowed_species))]
        for index, position in enumerate(structure.get_positions()):

            lattice_site = cpp_prim_structure.find_lattice_site_by_position(
                position)

            # Get allowed species on this site
            species = allowed_species[lattice_site.index]

            # Get what sublattice those species correspond to
            sublattice = self._allowed_species.index(tuple(sorted(species)))

            sublattice_to_indices[sublattice].append(index)

        for species, indices in zip(self._allowed_species, sublattice_to_indices):
            sublattice = Sublattice(chemical_symbols=species, indices=indices)
            self._sublattices.append(sublattice)

        # Map lattice index to sublattice index
        self._index_to_sublattice = {}
        for k, sublattice in enumerate(self):
            for index in sublattice.indices:
                self._index_to_sublattice[index] = k

    def __getitem__(self, key: int) -> Sublattice:
        """Returns a sublattice according to key."""
        return self._sublattices[key]

    def __len__(self):
        """Returns number of sublattices."""
        return len(self._sublattices)

    def get_sublattice_index(self, index: int) -> int:
        """ Returns the index of the sublattice the symbol
        or index in the structure belongs to.

        Parameters
        ----------
        index
            index of site in the structure
        """
        return self._index_to_sublattice[index]

    @property
    def allowed_species(self) -> List[List[str]]:
        """Lists of the allowed species on each sublattice, in order."""
        return deepcopy(self._allowed_species)

    def get_sublattice_sites(self, index: int) -> List[int]:
        """Returns the sites that belong to the sublattice with the
        corresponding index.

        Parameters
        ----------
        index
            index of the sublattice
        """
        return self[index].indices

    @property
    def active_sublattices(self) -> List[Sublattice]:
        """Lists of the active sublattices."""
        return [sl for sl in self if len(sl.chemical_symbols) > 1]

    @property
    def inactive_sublattices(self) -> List[Sublattice]:
        """Lists of the active sublattices."""
        return [sl for sl in self if len(sl.chemical_symbols) == 1]
