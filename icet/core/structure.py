from _icet import Structure


@classmethod
def __structure_from_atoms(self, conf):
    '''
    Creates an icet Structure object from an ASE Atoms object.

    Parameters
    ----------
    conf : ASE Atoms object
        input configuration

    Returns
    -------
    icet Structure object
        output configuration
    '''
    return self(conf.positions,
                conf.get_chemical_symbols(),
                conf.cell,
                conf.pbc.tolist())


Structure.from_atoms = __structure_from_atoms


def __structure_to_atoms(self):
    '''
    Returns the structure as an ASE Atoms object.

    Returns
    -------
    ASE Atoms object
        atomic configuration
    '''
    import ase
    conf = ase.Atoms(pbc=self.pbc)
    conf.set_cell(self.cell)
    for symbol, position in zip(self.chemical_symbols, self.positions):
        conf.append(ase.Atom(symbol, position))
    conf.set_positions(self.get_positions())
    conf.set_chemical_symbols(self.get_chemical_symbols())
    return conf


Structure.to_atoms = __structure_to_atoms


def __repr_function(self):
    s = ['Cell:']
    s += ['{}\n'.format(self.cell)]
    s += ['Element and positions:']
    for symbol, position in zip(self.chemical_symbols, self.positions):
        s += ['{}  [{:}  {:}  {:}]'.format(symbol, *position)]
    return '\n'.join(s)


Structure.__repr__ = __repr_function
