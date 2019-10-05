import itertools
from collections import OrderedDict


class LabelingGenerator():
    """
    Object used to generate all possible permutations of species on a given
    set of sites. If concentration restrictions are not specified, the
    generation will be a simple itertools.product loop.

    If concentrations are specified, the approach is a bit more elaborate.
    Take as an example a system with four sites, where Pd and Au are allowed
    on two and H and V on the others. `iter_species` will then look something
    like this: `[(0, 1), (0, 1), (2, 3), (2, 3)]`. The algorithm proceeds as
    follows:

    (1) Identify all unique species of `iter_species`, hereafter called
    `site_groups`. In this example they will be `(0, 1)` and `(2, 3)`. These
    unqiue species are saved in `site_groups`.

    (2) For each unique `site_group` and a given cell size, construct all
    (combinations of the species in the site group. If the cell size is 2,
    i.e. a supercell twice as big as the primitive cell, the possible
    (combinations for the `(0, 1)`-`site_group` will be `(0, 0, 0, 0)`, `(0,
    0, 0, 1)`, `(0, 0, 1, 1)` and `(1, 1, 1, 1)`. Order does not matter and
    concentration restrictions are not respected at this point.

    (3) Construct all "products" of the combinations of the unique site
    groups. In our example, the first ones are `[(0, 0, 0, 0), (2, 2, 2,
    2)]`, `[(0, 0, 0, 1), (2, 2, 2, 2)]` etc. Products that do not respect the
    concentration restrictions are discarded at this point.

    (4) For each "product", construct the permutations for each constituent
    `site_group`. For `[(0, 0, 0, 1), (2, 2, 2, 2)]` these would be
    `[(0, 0, 0, 1), (2, 2, 2, 2)]`, `[(0, 0, 1, 0), (2, 2, 2, 2)]`,
    `[(0, 1, 0, 0), (2, 2, 2, 2)]` and `[(1, 0, 0, 0), (2, 2, 2, 2)]`.
    If the second group was not all 2, there would of course be many more
    such permutations.

    (5) Sort the result according to the expectations of the structure
    enumeration (if we call the site group (0, 1) "A" and site group
    (2, 3) "B", the labeling should be AABBAABBAABB etc, i.e., one cell
    at a time). In the first permutation of (4), the resulting labeling
    will thus be `(0, 0, 2, 2, 0, 1, 2, 2)`.

    Parameters
    ----------
    iter_species : list of tuples of ints
        allowed species on each site
    concentrations : dict
        keys represent species (integers), values specify concentration ranges
    tol : float
        tolerance parameter used when comparing concentrations

    Attribtues
    ----------
    site_groups : OrderedDict
        keys are unique iter_species, the values are SiteGroup objects
        corresponding to that iter_species
    """

    def __init__(self, iter_species, concentrations, tol=1e-5):
        self.iter_species = iter_species
        self.concentrations = concentrations
        self.tol = tol

        if self.concentrations:
            self.site_groups = OrderedDict()
            count = 0
            for iter_species_key in iter_species:
                if iter_species_key in self.site_groups.keys():
                    self.site_groups[iter_species_key].multiplicity += 1
                else:
                    self.site_groups[iter_species_key] = SiteGroup(
                        iter_species_key, count)
                    count += 1

    def yield_labelings(self, ncells):
        """
        Yield labelings that comply with the concentration restrictions and,
        with `ncells` primitive cells.

        Parameters
        ----------
        ncells : int
            Size of supercell

        Yields
        ------
        tuple of ints
            Labeling
        """

        if self.concentrations:
            for site_group in self.site_groups.values():
                site_group.compute_all_combinations(ncells)
            for product in self.yield_products(ncells):
                for labeling in self.yield_permutations(product, 0):
                    yield self.sort_labeling(labeling, ncells)
        else:
            for labeling in itertools.product(*self.iter_species * ncells):
                yield labeling

    def yield_products(self, ncells):
        """
        Yield combinations (or rather products in the itertools terminology)
        of decorated site group combinations that comply with the concentration
        restrictions.

        Returns
        -------
        tuple of tuples of ints
            all unique combinations (products) of unordered iter_species
            combinations,
        """
        natoms = len(self.iter_species) * ncells
        combinations = [sg.combinations for sg in self.site_groups.values()]
        for product in itertools.product(*combinations):
            counts = {species: 0 for species in self.concentrations.keys()}
            for species_group in product:
                for species in self.concentrations.keys():
                    counts[species] += species_group.count(species)
            conc_restr_violation = False
            for species, conc_range in self.concentrations.items():
                if counts[species] / natoms < conc_range[0] - self.tol or \
                   counts[species] / natoms > conc_range[1] + self.tol:
                    conc_restr_violation = True
                    break
            if not conc_restr_violation:
                yield product

    def yield_unique_permutations(self, unique_species, permutation,
                                  position):
        """
        Recursively generate all _unique_ permutations of a set of species
        with given multiplicities. The algorithm is inspired by the one given
        at https://stackoverflow.com/a/6285203/6729551

        Parameters
        ----------
        unique_species : dict
            keys represent species,values the corresponding multiplicity
        permutation : list of ints
            permutation in process; should have length equal to the sum of all
            multiplicities
        position : int
            position currently processed; should be the index of the last
            species of `permutation` upon initialization

        Yields
        ------
        tuple with ints
            permutation where each species occurs according to the multiplicity
            specified in `unique_species`
        """
        if position < 0:
            # Finish recursion
            yield tuple(permutation)
        else:
            for species, occurrences in unique_species.items():
                # Add if the multiplicity allows
                if occurrences > 0:
                    permutation[position] = species
                    unique_species[species] -= 1
                    for perm in self.yield_unique_permutations(unique_species,
                                                               permutation,
                                                               position - 1):
                        yield perm
                    unique_species[species] += 1

    def yield_permutations(self, product, position):
        """
        Recursively generate all combinations of unique permutations of the
        tuples in `product`.

        Parameters
        ----------
        product : tuple of tuples of ints
            species allowed for each site group
        position : int
            keeps track of the position where recursion occurs; set to 0 upon
            initialization.

        Yields
        ------
        list of tuples of ints
            unique combinations of unique permutations, ordered by site group
        """
        unique_species = {species: product[position].count(species)
                          for species in set(product[position])}
        natoms = len(product[position])
        for permutation in self.yield_unique_permutations(unique_species,
                                                          [0] * natoms,
                                                          natoms - 1):
            if position == len(product) - 1:
                yield [permutation]
            else:
                for permutation_rest in self.yield_permutations(product,
                                                                position + 1):
                    yield [permutation] + permutation_rest

    def sort_labeling(self, labeling, ncells):
        """
        The elements in labeling are now given in site groups. We want just
        one labeling, ordered by site group, but one primitive cell at a time.
        So if iter_species given upon initialization was
        `[(0, 1), (2), (0, 1)]` and `ncells=2`, the current labeling has two
        tuples, the first corresponding to the `(0, 1)` site group, with 8
        elements in total, and the second corresponding to the `(2)` site group
        and with 2 elements in total. Now, we reorder it so that the elements
        are ordered according to `[(0, 1), (2), (0, 1), (0, 1), (2), (0, 1)]`.

        Parameters
        ----------
        labeling : list of tuple of ints
            As given by yield_permutations
        ncells : int
            Size of supercell

        Returns
        -------
        tuple of ints
            Labeling properly sorted, ready to be given to the enumeration code
        """
        sorted_labeling = [0] * len(self.iter_species * ncells)
        count = 0
        for _ in range(ncells):
            for iter_species in self.iter_species:

                # Find the site group corresponding to the iter species
                site_group = self.site_groups[iter_species]

                # Find the right element by checking (1) where the
                # proper site_group is in the unsorted labeling and
                # (2) which element is next in turn
                sorted_labeling[count] = labeling[
                    site_group.position][site_group.next_to_add]
                count += 1
                site_group.next_to_add += 1

        # Reset site group counters
        for site_group in self.site_groups.values():
            assert site_group.next_to_add == len(labeling[site_group.position])
            site_group.next_to_add = 0
        return tuple(sorted_labeling)


class SiteGroup():
    """
    Keeps track of a group of sites that have the same allowed species.
    That is a site group could correspond to all sites on which species 0 and
    1 are allowed, and the number of such sites is stored in the `multiplicity`
    attribute..

    Parameters
    ----------
    iter_species : tuple of ints
        allowed species on these sites
    position : int
        helps to keep track of when the first group occurred; the first
        site group encountered will have position = 0, the next 1 etc.

    Attributes
    ----------
    multiplicity : int
        multiplicity if the group, i.e., how many sites that have this
        iter_species
    next_to_add : int
        helper attribute that keeps track of which site in this group is the
        next to be added when the elements of a labeling are to be sorted
    """

    def __init__(self, iter_species, position):
        self.iter_species = iter_species
        self.position = position
        self.multiplicity = 1
        self.next_to_add = 0  # Will keep count when reordering elements

    def compute_all_combinations(self, ncells):
        """
        Compute all combinations (without considering order) of the elements
        in the group.

        Parameters
        ----------
        ncells : int
            Size of supercell
        """
        self.combinations = []
        natoms = self.multiplicity * ncells
        for combination in \
                itertools.combinations_with_replacement(self.iter_species,
                                                        natoms):
            self.combinations.append(combination)
