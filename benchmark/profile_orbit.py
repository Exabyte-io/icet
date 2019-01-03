from benchmark_orbit import (init_cpp_orbit,
                             time_orbit_sites_permutations,
                             time_orbit_translating)
'''
This script is intended to be run in a profiler, e.g running with:

python3 -m cProfile benchmark/profile_orbit.py
'''


orbit_pair, orbit_triplet = init_cpp_orbit(100)
timing = time_orbit_translating(orbit_pair)
timing = time_orbit_sites_permutations(orbit_pair)
