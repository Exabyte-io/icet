from typing import Callable, List, Tuple, Union

import numpy as np
from ase import Atoms
from icet import ClusterSpace, StructureContainer
from icet.input_output.logging_tools import logger
from mchammer.ensembles.canonical_annealing import available_cooling_functions


logger = logger.getChild('training_set_generation')


def _get_fit_matrix(structure_container: StructureContainer,
                    new_inds: np.ndarray,
                    n_base_structures: int) -> np.ndarray:
    """
    Get the current fit matrix.

    Parameters
    ----------
    structure_container
        A structure container.
    new_inds
        The part of the structure container that contains the
        new structures to be added.
    n_base_structures
        Number of structures in the base pool.
    """
    base_inds = np.array(range(n_base_structures), dtype=int)
    inds = np.append(base_inds, n_base_structures + new_inds)
    fit_matrix, _ = structure_container.get_fit_data(inds)
    return fit_matrix


def _do_swap(inds: np.ndarray,
             n_structures_to_add: int,
             n_mcmc_structures: int) -> np.ndarray:
    """
    Update indices to be used as training data.

    Parameters
    ----------
    inds
        The current indicies that are used.
    n_structures_to_add
        Total number of structures to add to the current base structures.
    n_mcmc_structures
        The size of the pool of potential candidate structures.
    """
    # Get index to swap out
    _inds = inds.copy()
    swap_out = np.random.choice(range(inds.size))

    # Get index from the pool that are not currently in inds
    inds_pool = np.array([range(n_mcmc_structures)])
    inds_pool = np.setdiff1d(inds_pool, inds, assume_unique=True)

    # Get index of structure to swap in
    swap_in = np.random.choice(inds_pool)

    # Do the swap
    _inds[swap_out] = swap_in
    return _inds


def structure_selection_annealing(
        cluster_space: ClusterSpace,
        monte_carlo_structures: List[Atoms],
        n_structures_to_add: int,
        n_steps: int,
        base_structures: List[Atoms] = None,
        cooling_start: float = 5,
        cooling_stop: float = 0.001,
        cooling_function: Union[str, Callable] = 'exponential',
        initial_indices: List[int] = None) \
            -> Tuple[List[int], List[float]]:
    """Given a cluster space, a base pool of structures, and a new pool
    of structures, this function uses a Monte Carlo inspired annealing
    method to find a good structure pool for training.

    Returns
    -------
        A tuple comprising the indices of the optimal structures in
        the :attr:`monte_carlo_structures` pool and a list of accepted
        metric values.

    Parameters
    ----------
    cluster_space
        A cluster space defining the lattice to be occupied.
    monte_carlo_structures
        A list of candidate training structures.
    n_structures_to_add
        How many of the structures in the :attr:`monte_carlo_structures`
        pool that should be kept for training.
    n_steps
        Number of steps in the annealing algorithm.
    base_structures
        A list of structures that is already in your training pool;
        can be ``None`` if you do not have any structures yet.
    cooling_start
        Initial value of the :attr:`cooling_function`.
    cooling_stop
        Last value of the :attr:`cooling_function`.
    cooling_function
        Artificial number that rescales the difference between the
        metric value between two iterations.  Available options are
        ``'linear'`` and ``'exponential'``.
    initial_indices
        Picks out the starting structure from the
        :attr:`monte_carlo_structures` pool. Can be used if you want
        to continue from an old run for example.

    Example
    -------

    The following snippet demonstrates the use of this function for
    generating an optimized structure pool. Here, we first set up a
    pool of candidate structures by randomly occupying a FCC supercell
    with Au and Pd::

        >>> from ase.build import bulk
        >>> from icet.tools.structure_generation import occupy_structure_randomly

        >>> prim = bulk('Au', a=4.0)
        >>> cs = ClusterSpace(prim, [6.0], [['Au', 'Pd']])
        >>> structure_pool = []
        >>> for _ in range(500):
        >>>     # Create random supercell.
        >>>     supercell = np.random.randint(1, 4, size=3)
        >>>     structure = prim.repeat(supercell)
        >>>
        >>>     # Randomize concentrations in the supercell
        >>>     n_atoms = len(structure)
        >>>     n_Au = np.random.randint(0, n_atoms)
        >>>     n_Pd = n_atoms - n_Au
        >>>     concentration = {'Au': n_Au / n_atoms, 'Pd': n_Pd / n_atoms}
        >>>
        >>>     # Occupy the structure randomly and store it.
        >>>     occupy_structure_randomly(structure, cs, concentration)
        >>>     structure_pool.append(structure)
        >>> start_inds = [f for f in range(10)]

    Now we can use the :func:`structure_selection_annealing` function to find an
    optimized structure pool::

        >>> inds, cond = structure_selection_annealing(cs,
        >>>                                            structure_pool,
        >>>                                            n_structures_to_add=10,
        >>>                                            n_steps=100)
        >>> training_structures = [structure_pool[ind] for ind in inds]
        >>> print(training_structures)

    """
    if base_structures is None:
        base_structures = []

    # set up cooling function
    if isinstance(cooling_function, str):
        available = sorted(available_cooling_functions.keys())
        if cooling_function not in available:
            raise ValueError(f'Select from the available cooling_functions: {available}')
        _cooling_function = available_cooling_functions[cooling_function]
    elif callable(cooling_function):
        _cooling_function = cooling_function
    else:
        raise TypeError('cooling_function must be either str or a function')

    # set up cluster vectors
    structure_container = StructureContainer(cluster_space)
    for structure in base_structures:
        structure_container.add_structure(structure, properties={'energy': 0})
    for structure in monte_carlo_structures:
        structure_container.add_structure(structure, properties={'energy': 0})

    # get number of structures in monte_carlo_structures
    n_mcmc_structures = len(monte_carlo_structures)
    n_base_structures = len(base_structures)

    # randomly chose starting structure unless user want specific indices
    if not initial_indices:
        inds = np.random.choice(range(len(monte_carlo_structures)), size=n_structures_to_add,
                                replace=False)
    else:
        inds = np.array(initial_indices)

    # get initial fitting_matrix, A in Ax = y
    fit_matrix = _get_fit_matrix(structure_container, inds, n_base_structures)

    # get metric of fitting_matrix
    cond = np.linalg.cond(fit_matrix)
    cond_traj = [cond]
    for n in range(n_steps):
        # get current artificial cooling
        T = _cooling_function(n, cooling_start, cooling_stop, n_steps)

        # do swaps from pool
        new_inds = _do_swap(inds, n_structures_to_add, n_mcmc_structures)
        new_fit_matrix = _get_fit_matrix(structure_container, new_inds, n_base_structures)

        # get new metric
        cond_new = np.linalg.cond(new_fit_matrix)
        if (cond - cond_new) / T > np.log(np.random.uniform()):
            # if accepted update data
            cond = cond_new
            inds = new_inds
            cond_traj.append(cond)

        if n % 100 == 0:
            logger.info(f'step {n:6d}, T {T:8.5f} , current condition number {cond:8.5f}')

    return inds, cond_traj
