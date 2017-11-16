from _icetdev import LatticeSite


def __latticeSite_repr(self):
    return '{} : {}'.format(self.index, self.unitcellOffset)

LatticeSite.__repr__ = __latticeSite_repr
