'''
This examples demonstrates how to find the lattice sites for a structure from
the list of nearest neighbors
'''

# Start import
from icetdev.tools.geometry import get_scaled_positions
from icetdev.structure import Structure
from icetdev.neighborlist import Neighborlist
from ase import Atom
from ase.build import bulk
import time
import numpy as np
# End import

# Create a prototype Al structure in the form of a 1x1x1 unit cell with 2
# extra Al atoms.
# Start setup
atoms = bulk("Al", "fcc", a=2.0).repeat(1)
atoms.append(Atom("Al", position=[0.5, 0.5, 0.5]))
atoms.append(Atom("Al", position=[0.25, 0.5, 0.5]))
structure = Structure().from_atoms(atoms)
# End setup

# Construct a list of all neighbors within the cutoff (6.0 A).
# Start neighbor
neighbor_cutoff = 6.0
nl = Neighborlist(neighbor_cutoff)
nl.build(structure)
# End neighbor

# Extract the positions and fractional coordinates for all neighbors.
# Start position
pos_neighbors = []
for latNbr in nl.get_neighbors(0):
    pos = structure.get_position(latNbr)
    pos_neighbors.append(pos)
frac_coordinates = get_scaled_positions(np.array(pos_neighbors),
                                        cell=atoms.cell, wrap=False,
                                        pbc=structure.pbc)
# End position

# Check the time required for finding all the lattice sites for the positions
# of the neighbors
# Start lattice_sites
t0 = time.time()
lat_nbrs = structure.find_lattice_sites_by_positions(pos_neighbors)
t1 = time.time()
print("# Found {} LatticeSites in {} ms ".format(len(lat_nbrs),
                                                 round((t0-t1)*1e3, 4)))
# End lattice_sites

# Print the fractional coordinates, positions, lattice sites for all
# neihbors.
# Start results
print("#1: frac. pos. #2: position #3: latt. site")
for fpos, pos, lat_nbr in zip(frac_coordinates, pos_neighbors, lat_nbrs):
    print(fpos, pos, lat_nbr)
# End results
