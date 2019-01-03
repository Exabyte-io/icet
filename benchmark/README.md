Benchmarking and profiling
==========================

Notes
-----

* Files beginning with `benchmark_` will print out timings of different
  functions

* Files beginning with `profile_` are similar to the benchmark scripts but are
  designed to be profiled

When making changes to core classes, run the benchmark scripts before and after
the change to make sure nothing gets broken in terms of performance. If
something gets broken then use the profile scripts to pin point the problem.

Read about basic profiling here:
* https://docs.python.org/3/library/profile.html#introduction-to-the-profilers


Example
-------

Profile `LatticeSite` by running

```
python3 -m cProfile profile_lattice_site.py
```

One can easily sort the output by cumulative time with sort:
```
python3 -m cProfile -s cumtime profile_lattice_site.py
```
which results in something like:
```
Timing for sorting: 22.114821 musec
Timing for hash: 18.133105 musec
Timing for index lookup: 13.363365 musec
Timing for offset lookup: 35.801517 musec
         578749 function calls (567144 primitive calls) in 0.628 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    998/1    0.031    0.000    0.629    0.629 {built-in method builtins.exec}
        1    0.000    0.000    0.629    0.629 profile_lattice_site.py:1(<module>)
        5    0.000    0.000    0.612    0.122 __init__.py:2(<module>)
   1194/1    0.018    0.000    0.517    0.517 <frozen importlib._bootstrap>:966(_find_and_load)
   1194/1    0.004    0.000    0.517    0.517 <frozen importlib._bootstrap>:936(_find_and_load_unlocked)
   1301/1    0.001    0.000    0.517    0.517 <frozen importlib._bootstrap>:211(_call_with_frames_removed)
    584/1    0.001    0.000    0.517    0.517 {built-in method builtins.__import__}
    871/1    0.003    0.000    0.517    0.517 <frozen importlib._bootstrap>:651(_load_unlocked)
    715/1    0.002    0.000    0.517    0.517 <frozen importlib._bootstrap_external>:672(exec_module)
       50    0.001    0.000    0.513    0.010 __init__.py:1(<module>)
4305/3001    0.003    0.000    0.245    0.000 <frozen importlib._bootstrap>:997(_handle_fromlist)
        1    0.000    0.000    0.186    0.186 cluster_space.py:1(<module>)
        5    0.000    0.000    0.182    0.036 __init__.py:5(<module>)
        1    0.000    0.000    0.175    0.175 cluster_expansion.py:1(<module>)
        1    0.000    0.000    0.151    0.151 optimizer.py:3(<module>)
...
```


Benchmarks
==========

Orbit
-----

This benchmark typical orbit operations that are done in `icet`. Orbit
translation is done to create a new orbit where all lattice sites have had
their unitcell offsets translated. This is useful to get all lattice sites of a
supercell if your orbit is for the primitive cell.

`orbit_permutation` is a property that returns the permuted equivalent sites so
that they are in an identical order to the representative sites. This is useful
when counting element combinations on an orbit. For example, if we are
interested in how many ABB elements there are compared to BAB then we would
need the permuted sites.

A typical output from `benchmark_orbit.py` is
```
Time for pair orbit translation: 0.00000641 sec
Time for triplet orbit translation: 0.00000788 sec
Time for pair orbit permutations: 0.00008434 sec
Time for triplet orbit permutations: 0.00012472 sec
```
This test was run on a Intel(R) Core(TM) i7-6500U CPU @ 2.50GHz




Profiling
=========

Orbit
-----

Running
```
 python3 -m cProfile -s cumtime profile_orbit.py
 ```
 will result in something like this:
 ```
         576991 function calls (565374 primitive calls) in 0.971 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   1001/1    0.030    0.000    0.972    0.972 {built-in method builtins.exec}
        1    0.000    0.000    0.972    0.972 profile_orbit.py:1(<module>)
       50    0.001    0.000    0.522    0.010 __init__.py:1(<module>)
   1196/1    0.004    0.000    0.516    0.516 <frozen importlib._bootstrap>:966(_find_and_load)
   1196/1    0.004    0.000    0.516    0.516 <frozen importlib._bootstrap>:936(_find_and_load_unlocked)
    874/1    0.003    0.000    0.516    0.516 <frozen importlib._bootstrap>:651(_load_unlocked)
    718/1    0.002    0.000    0.516    0.516 <frozen importlib._bootstrap_external>:672(exec_module)
   1305/1    0.001    0.000    0.516    0.516 <frozen importlib._bootstrap>:211(_call_with_frames_removed)
        1    0.000    0.000    0.516    0.516 benchmark_orbit.py:1(<module>)
   585/11    0.001    0.000    0.511    0.046 {built-in method builtins.__import__}
        5    0.000    0.000    0.509    0.102 __init__.py:2(<module>)
        1    0.447    0.447    0.447    0.447 benchmark_orbit.py:80(time_orbit_sites_permutations)
4313/3001    0.003    0.000    0.245    0.000 <frozen importlib._bootstrap>:997(_handle_fromlist)
...
```
