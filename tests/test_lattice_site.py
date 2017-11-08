from icetdev.lattice_site import LatticeSite

''' Test that lattice site is hashable '''

latnbr = LatticeSite(1, [0., 0., 0.])
latnbr_map = {}
latnbr_map[latnbr] = 1
assert latnbr in latnbr_map, 'Failed to hash lattice site'
