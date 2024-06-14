import itertools
import numpy as np
from scipy.linalg import null_space


class Constraints:
    """ Class for handling linear constraints with right-hand-side equal to zero.

    Parameters
    ----------
    n_params
        Number of parameters in model.

    Example
    -------
    The following example demonstrates fitting of a cluster expansion under the
    constraint that parameter 2 and parameter 4 should be equal::

        >>> import numpy as np
        >>> from icet.tools import Constraints
        >>> from trainstation import Optimizer

        >>> # Set up random sensing matrix and target "energies"
        >>> n_params = 10
        >>> n_energies = 20
        >>> A = np.random.random((n_energies, n_params))
        >>> y = np.random.random(n_energies)

        >>> # Define constraints
        >>> c = Constraints(n_params=n_params)
        >>> M = np.zeros((1, n_params))
        >>> M[0, 2] = 1
        >>> M[0, 4] = -1
        >>> c.add_constraint(M)

        >>> # Do the actual fit and finally extract parameters
        >>> A_constrained = c.transform(A)
        >>> opt = Optimizer((A_constrained, y), fit_method='ridge')
        >>> opt.train()
        >>> parameters = c.inverse_transform(opt.parameters)

    """

    def __init__(self, n_params: int):
        self.M = np.empty((0, n_params))
        self.constraint_vectors = np.eye(n_params)

    def transform(self, A: np.ndarray) -> np.ndarray:
        """ Transform array to constrained parameter space.

        Parameters
        ----------
        A
            Array to be transformed.
         """
        return A.dot(self.constraint_vectors)

    def inverse_transform(self, A: np.ndarray) -> np.ndarray:
        """ Inverse transform array from constrained parameter space
        to unconstrained space.

        Parameters
        ----------
        A
            Array to be inverse transformed.
        """
        return self.constraint_vectors.dot(A)

    def add_constraint(self, M: np.ndarray) -> None:
        """ Add a constraint matrix and resolve for the constraint space.

        Parameters
        ----------
        M
            Constraint matrix with each constraint as a row. Can (but need not be)
            cluster vectors.
        """
        M = np.array(M)
        self.M = np.vstack((self.M, M))
        self.constraint_vectors = null_space(self.M)


def get_mixing_energy_constraints(cluster_space) -> Constraints:
    """
    A cluster expansion of the *mixing energy* should ideally predict zero energy
    for concentrations 0 and 1. This function constructs a :class:`Constraints`
    object that enforces that condition during training.

    Parameters
    ----------
    cluster_space
        Cluster space corresponding to cluster expansion for which constraints
        should be imposed.

    Example
    -------
    This example demonstrates how to constrain the mixing energy to zero
    at the pure phases in a toy example with random cluster vectors and
    random target energies::

        >>> import numpy as np
        >>> from ase.build import bulk
        >>> from icet import ClusterSpace
        >>> from icet.tools import get_mixing_energy_constraints
        >>> from trainstation import Optimizer

        >>> # Set up cluster space along with random sensing matrix and target "energies"
        >>> prim = bulk('Au')
        >>> cs = ClusterSpace(prim, cutoffs=[6.0, 5.0], chemical_symbols=['Au', 'Ag'])
        >>> n_params = len(cs)
        >>> n_energies = 20
        >>> A = np.random.random((n_energies, n_params))
        >>> y = np.random.random(n_energies)

        >>> # Define constraints
        >>> c = get_mixing_energy_constraints(cs)

        >>> # Do the actual fit and finally extract parameters
        >>> A_constrained = c.transform(A)
        >>> opt = Optimizer((A_constrained, y), fit_method='ridge')
        >>> opt.train()
        >>> parameters = c.inverse_transform(opt.parameters)

    Warning
    -------
    Constraining the energy of one structure is always done at the expense of the
    fit quality of the others. Always expect that your cross-validation scores will
    increase somewhat when using this function.
    """
    M = []

    prim = cluster_space.primitive_structure.copy()
    sublattices = cluster_space.get_sublattices(prim)
    chemical_symbols = [subl.chemical_symbols for subl in sublattices]

    # Loop over all combinations of pure phases
    for symbols in itertools.product(*chemical_symbols):
        structure = prim.copy()
        for subl, symbol in zip(sublattices, symbols):
            for atom_index in subl.indices:
                structure[atom_index].symbol = symbol

        # Add constraint for this pure phase
        M.append(cluster_space.get_cluster_vector(structure))

    c = Constraints(n_params=len(cluster_space))
    c.add_constraint(M)
    return c
