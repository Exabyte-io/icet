from _icet import LatticeSite


def __latticeSite_repr(self):
    return '{} : {}'.format(self.index, self.unitcell_offset)


LatticeSite.__repr__ = __latticeSite_repr
