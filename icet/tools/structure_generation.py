import random
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from spglib import niggli_reduce as spg_nigg_red

from icet import ClusterSpace
from ..io.logging import logger
from .structure_enumeration import get_symmetry_operations
from .structure_enumeration_support.hermite_normal_form \
    import get_reduced_hnfs
from .structure_enumeration_support.smith_normal_form \
    import get_unique_snfs

logger = logger.getChild('structure_generation')


def generate_sqs_cell(prim: Atoms, target_conc: Dict[str, float],
                      cutoffs: list, maxsize: int, seed: int = 42,
                      T_start: float = 0.5, T_stop: float = 0.001,
                      n_steps: int = 10000) \
                      -> Tuple[Atoms, float]:
    """
    Generate a special quasi-random structure (SQS) using a simulated
    annealing procedure that samples all cell shapes that are
    compatible with the maximum cell size.

    Parameters
    ----------
    prim
        primitive structure
    target_conc
        dictionary with target concentrations, where the keys
        represent chemical species
    cutoffs
        cutoffs for pairs, triplets, etc to consider for correlation
        analysis
    maxsize
        maximum size of supercell in multiples of the primitive structure
    seed
        seed for random number generator
    T_start
        initial temperature for annealing procedure (dimensionless)
    T_final
        final temperature for annealing procedure (dimensionless)
    n_steps
        number of steps for annealing procedure

    Returns
    -------
    structure with lowest score and lowest score


    Examples
    --------
    The following code snippet illustrates the use of this function.
    ```python
    from ase.build import bulk
    from icet.tools import generate_sqs_cell

    prim = bulk('Au')
    target_conc = {'Au': 0.5, 'Ag': 0.5}
    atoms, score = generate_sqs_cell(prim, target_conc,
                                     cutoffs=[8.0], maxsize=12)
    ```

    To see a more verbose output of the annealing procedure turn the
    logging level to 'INFO'.

    ```python
    from icet.io.logging import set_log_config
    set_log_config(level='INFO')
    ```
    """

    logger.debug('Setting up cluster space...')
    cs = ClusterSpace(prim, cutoffs, list(target_conc.keys()))
    logger.debug('Getting target cluster vector...')
    target_cv = _get_sqs_cluster_vector(cs, target_conc)
    logger.debug('Enumerating cell metrices...')
    P_matrices = list(_enumerate_cells(prim, range(maxsize+1)))

    random.seed(seed)
    np.random.seed(seed)

    logger.debug('Starting simulated annealing...')
    logger.info('temperature   new-score     current    minimum')
    min_score, min_structure = 1e4, None
    cur_score = 1e4
    for step in range(n_steps):

        T = T_start - (T_start - T_stop) * np.log(step+1) / np.log(n_steps)

        P = random.choice(P_matrices)
        structure = make_supercell(prim, P)
        elems, conc = list(target_conc.keys()), list(target_conc.values())
        structure.set_chemical_symbols(np.random.choice(
            elems, len(structure), p=conc))

        cv = cs.get_cluster_vector(structure)
        score = np.linalg.norm(cv - target_cv)
        logger.info('{:11.5f}  {:11.5f}  {:11.5f}  {:11.5f}'
                    .format(T, score, cur_score, min_score))
        if cur_score is None \
           or score < cur_score \
           or np.exp((cur_score - score) / T) > np.random.uniform(0, 1, 1):
            cur_score = score
            if min_score is None or score < min_score:
                logger.info('Found new optimal structure with {} sites'.
                            format(len(structure)))
                min_score = score
                min_structure = structure.copy()

    return min_structure, min_score


def _enumerate_cells(atoms: Atoms,
                     sizes: List[int],
                     niggli_reduce: bool = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generates all supercell cell metrices that are possible up to the
    specified sizes based on the primitive structure provided.

    Parameters
    ----------
    atoms
        primitive structure to consider
    sizes
        supercell sizes to consider
    niggli_reduction
        if True perform a Niggli reduction with spglib for each structure;
        the default is ``True`` if ``atoms`` is periodic in all directions,
        ``False`` otherwise.

    Returns
    ------
    the permutation matrix :math:`\boldsymbol{P}` which relates the
    supercell metric :math:`\boldsymbol{h}` to the primitive cell
    metric :math:`\boldsymbol{h}_p` according to
    :math:`\boldsymbol{h_p P} = \boldsymbol{h}`
    """

    # Niggli reduce by default if all directions have
    # periodic boundary conditions
    if niggli_reduce is None:
        niggli_reduce = all(atoms.pbc)

    symmetries = get_symmetry_operations(atoms)

    # Loop over each cell size
    for ncells in sizes:
        if ncells == 0:
            continue

        hnfs = get_reduced_hnfs(ncells, symmetries, atoms.pbc)
        snfs = get_unique_snfs(hnfs)

        for snf in snfs:
            for hnf in snf.hnfs:
                new_cell = np.dot(atoms.cell.T, hnf.H).T
                if niggli_reduce:
                    new_cell = spg_nigg_red(new_cell)
                yield hnf.H


def _get_sqs_cluster_vector(cs: ClusterSpace,
                            target_conc: Dict[str, float]) \
                            -> np.ndarray:
    """
    Returns the cluster vector that corresponds to an ideal random
    distribution.

    Todo
    ----
    * adapt this function for arbitrary concentrations (currently
      limited to 50%) and number of components (currently limited to
      binary systems)
    """
    return [1] + (len(cs) - 1) * [0]
