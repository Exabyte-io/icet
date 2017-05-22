from tests import manybodyNeighborlistTester
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.build import bulk


mbnl_T = manybodyNeighborlistTester.manybodyNeighborlistTester()

atoms  = bulk("Al").repeat(2)

neighbor_cutoff = 5.0
ase_nl = NeighborList(len(atoms)*[neighbor_cutoff/2],skin=1e-8,
                    bothways=True,self_interaction=False)
ase_nl.update(atoms)


index = 0
order = 3
bothways = True


nbrs = mbnl_T.build(ase_nl, index, order, bothways) 

for j in nbrs:    
      print(j[0], len(j[1]))