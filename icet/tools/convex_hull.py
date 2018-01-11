import itertools
import numpy as np
from scipy.spatial import ConvexHull as ConvexHullSciPy
from scipy.spatial.qhull import QhullError
from scipy.interpolate import griddata


class ConvexHull(object):
    '''
    Convex hull of concentration and energies.

    Attributes
    ----------
    dimensions : int
        Number of independent concentrations needed to specify a point in
        concentration space (1 for binaries, 2 for ternaries etc)
    concentrations : NumPy array (N, dimensions)
        Concentration of the N structures on the convex hull.
    energies : NumPy array
        Energy of the N structures on the convex hull.
    structures : list of ints
        Indices of structures that constitute the convex hull (indices are
        defined by the order their concentrations and energies are fed when
        initializing the ConvexHull object).
    '''

    def __init__(self, concentrations, energies):
        '''
        Construct convex hull for (free) energy of mixing. The core function
        the Convex Hull calculator in SciPy, see
         http://docs.scipy.org/doc/scipy-dev/reference/\
            generated/scipy.spatial.ConvexHull.html

        Parameters
        ----------
        concentrations : list of lists of floats
            Concentrations for each structure listed as [[c1, c2], [c1, c2],
            ...]. For binaries, in which case there is only one independent
            concentration, the format [c1, c2, c3, ...] works fine.
        energies : list of floats
            Energy/energy of mixing for each structure.
        '''

        assert len(concentrations) == len(energies)
        # Prepare data in format suitable for SciPy-ConvexHull
        concentrations = np.array(concentrations)
        energies = np.array(energies)
        points = np.column_stack((concentrations, energies))
        self.dimensions = len(points[0]) - 1

        # Construct convex hull
        hull = ConvexHullSciPy(points)

        # Collect convex hull points in handy arrays
        concentrations = []
        energies = []
        for vertex in hull.vertices:
            if self.dimensions == 1:
                concentrations.append(points[vertex][0])
            else:
                concentrations.append(points[vertex][0:-1])
            energies.append(points[vertex][-1])
        concentrations = np.array(concentrations)
        energies = np.array(energies)

        # If there is just one independent concentration, we'd better sort
        # according to it
        if self.dimensions == 1:
            concentrations, energies = (np.array(t) for t in
                                        zip(*sorted(zip(concentrations,
                                                        energies))))

        self.concentrations = concentrations
        self.energies = energies
        self.structures = hull.vertices

        # Remove points that are above "pure element plane"
        self._remove_points_above_tie_plane()

    def _remove_points_above_tie_plane(self, tol=1e-6):
        '''
        Remove all points on the convex hull that correspond to maximum rather
        than minimum energy.

        Parameters
        ----------
        tol : float
            Tolerance for what energy constitutes a lower one.
        '''

        # Identify the "complex concentration hull", i.e. the extremum
        # concentrations. In the simplest case, these should simply be the
        # pure elements.
        if self.dimensions == 1:
            # Then the ConvexHullScipy function doesn't work, so we just pick
            # the indices of the lowest and highest concentrations.
            vertices = []
            vertices.append(np.argmin(self.concentrations))
            vertices.append(np.argmax(self.concentrations))
            vertices = np.array(vertices)
        else:
            concentration_hull = ConvexHullSciPy(self.concentrations)
            vertices = concentration_hull.vertices

        # Remove all points of the convex energy hull that have an energy that
        # is higher than what would be gotten with pure elements at the same
        # concentration. These points are mathematically on the convex hull,
        # but in the physically uninteresting upper part, i.e. they maximize
        # rather than minimize energy.
        to_delete = []
        for i, concentration in enumerate(self.concentrations):
            # The points on the convex concentration hull should always be
            # included, so skip them.
            if i in vertices:
                continue

            # The energy obtained as a linear combination of concentrations on
            # the convex hull is the "z coordinate" of the position on a
            # (hyper)plane in the (number of independent concentrations +
            # 1)-dimensional (N-D) space. This plane is spanned by N points.
            # If there are more vertices on the convex hull, we need to loop
            # over all combinations of N vertices.
            for plane in itertools.combinations(vertices,
                                                min(len(vertices),
                                                    self.dimensions + 1)):
                # Calculate energy that would be gotten with pure elements
                # with ascribed concentration.
                energy_pure = griddata(self.concentrations[np.array(plane)],
                                       self.energies[np.array(plane)],
                                       concentration,
                                       method='linear')

                # Prepare to delete if the energy was lowered. `griddata` gives
                # NaN if the concentration is outside the triangle formed by
                # the three vertices. The result of the below comparison is
                # then False, which is what we want.
                if energy_pure < self.energies[i] - tol:
                    to_delete.append(i)
                    break

        # Finally remove all points
        self.concentrations = np.delete(self.concentrations, to_delete, 0)
        self.energies = np.delete(self.energies, to_delete, 0)
        self.structures = list(np.delete(self.structures, to_delete, 0))

    def get_energy_at_convex_hull(self, target_concentrations):
        '''
        Get energy of convex hull at specified concentrations.

        Parameters
        ----------
        target_concentrations : list of lists
            Concentrations at target points. If there is one independent
            concentration, a list of floats is fine. Otherwise, the
            concentrations are given as a list of lists, such as [[0.1, 0.2],
            [0.3, 0.1], ...]

        Returns
        -------
        NumPy array
            Energies at the specified target_concentrations. If any
            concentration is outside the allowed range, NaN is returned.
        '''
        if self.dimensions > 1:
            assert len(target_concentrations[0]) == self.dimensions

        # Loop over all complexes of N+1 points to make sure that the lowest
        # energy plane is used in the end. This is needed in two dimensions
        # but in higher.
        hull_candidate_energies = []
        for plane in itertools.combinations(range(len(self.energies)),
                                            min(len(self.energies),
                                                self.dimensions + 1)):
            try:
                plane_energies = griddata(self.concentrations[list(plane)],
                                          self.energies[list(plane)],
                                          np.array(target_concentrations),
                                          method='linear')
            except QhullError:
                # If the points lie on a line, the convex hull will fail, but
                # we do not need to care about these "planes" anyway
                continue
            hull_candidate_energies.append(plane_energies)

        # Pick out the lowest energies found
        hull_energies = np.nanmin(hull_candidate_energies, axis=0)
        return hull_energies

    def extract_low_energy_structures(self, concentrations, energies,
                                      energy_tolerance, structures=None):
        '''

        Extract structures that are sufficiently close in energy to the convex
        hull.

        Parameters
        ----------
        concentrations : list of lists
            Concentrations of candidate structures. If there is one
            independent concentration, a list of floats is fine. Otherwise,
            the concentrations are given as a list of lists, such as `[[0.1,
            0.2], [0.3, 0.1], ...]`
        energies : list of floats
            Energies of candidate structures.
        energy_tolerance : float
            Every structure that has an energy within `energy_tolerance` from
            the convex hull at its concentration will be returned.
        structures : list
            List of candidate `ASE Atoms` or other objects corresponding to
            the `concentrations` and `energies`. The same list will be
            returned, but with the objects too far from the convex hull
            removed. If `None`, a list of indices is returned instead.

        Returns
        -------
        list
            The members of structures whose energy is within `energy_tolerance`
            from the convex hull. If `structures` is left empty, a list of
            indices corresponding to the order in `concentrations`/`energies`
            is returned instead.
        '''
        number_of_candidates = len(concentrations)
        assert len(energies) == number_of_candidates
        if structures is None:
            structures = list(range(number_of_candidates))
        assert len(structures) == number_of_candidates

        # Calculate energy at convex hull for specified concentrations
        hull_energies = self.get_energy_at_convex_hull(concentrations)

        # Check which ones were close enough
        close_to_hull_structures = []
        for i in range(number_of_candidates):
            if energies[i] <= hull_energies[i] + energy_tolerance:
                close_to_hull_structures.append(structures[i])

        return close_to_hull_structures
