"""
This module provides the Structure class.
"""

from ase import Atoms

from _icet import Structure


@classmethod
def _structure_from_atoms(self, conf: Atoms):
    """
    Returns the input configuration as an icet Structure object.

    Parameters
    ----------
    conf
        input configuration

    Returns
    -------
    atomic configuration
    """
    return self(conf.positions,
                conf.get_chemical_symbols(),
                conf.cell,
                conf.pbc.tolist())


Structure.from_atoms = _structure_from_atoms


def _structure_to_atoms(self) -> Atoms:
    """
    Returns the structure as an ASE Atoms object.

    Returns
    -------
    atomic configuration
    """
    import ase
    conf = ase.Atoms(pbc=self.pbc)
    conf.set_cell(self.cell)
    for symbol, position in zip(self.chemical_symbols, self.positions):
        conf.append(ase.Atom(symbol, position))
    conf.set_positions(self.get_positions())
    conf.set_chemical_symbols(self.get_chemical_symbols())
    return conf


Structure.to_atoms = _structure_to_atoms


def _repr_function(self) -> str:
    s = ['Cell:']
    s += ['{}\n'.format(self.cell)]
    s += ['Element and positions:']
    for symbol, position in zip(self.chemical_symbols, self.positions):
        s += ['{}  [{:}  {:}  {:}]'.format(symbol, *position)]
    return '\n'.join(s)


Structure.__repr__ = _repr_function
