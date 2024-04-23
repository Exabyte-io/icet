from mchammer.free_energy_tools \
        import (_lambda_function_forward, _lambda_function_backward)

from ase import Atoms
from ase.units import kB
from typing import List

from .thermodynamic_base_ensemble import ThermodynamicBaseEnsemble
from ..calculators.base_calculator import BaseCalculator
from .. import DataContainer
from icet.input_output.logging_tools import logger

logger = logger.getChild('thermodynamic_integration_ensemble')


class ThermodynamicIntegrationEnsemble(ThermodynamicBaseEnsemble):
    r"""Instances of this class allow one to find the free energy of the
    system.  To this end, we use the :class:`canonncal ensemble
    <mchammer.ensembles.CanonicalEnsemble>` with a modified
    Hamiltonian,

    .. math::
        H(\lambda) = (1 - \lambda) H_{A} + \lambda H_{B}

    The Hamiltonian is then sampled continuously from :math:`\lambda=0`
    to :math:`\lambda=1`. :math:`H_{B}` is your cluster expansion
    and :math:`H_{A}=0`, is a completely disordered system, with free
    energy given by the ideal mixing entropy.

    The free energy, A, of system B is then given by:

    .. math::
        A_{B} = A_{A} + \int_{0}^{1} \left\langle\frac{\mathrm{d}H(\lambda)}
        {\mathrm{d}\lambda}\right\rangle_{H} \mathrm{d}\lambda

    and since :math:`A_{A}` is known it is easy to compute :math:`A_{B}`

    :math:`\lambda` is parametrized as,

    .. math::
        \lambda(x) = x^5(70x^4 - 315x^3 + 540x^2 - 420x + 126)

    where :math:`x = \mathrm{step} / (\mathrm{n\_steps} - 1)`.

    Parameters
    ----------
    structure
        Atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector.
    calculator
        Calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion.
    temperature
        Temperature  :math:`T` in appropriate units, commonly Kelvin.
    n_lambdas
        Number of :math:`\lambda` values to be sampled between 0 and 1.
    forward
        If this is set to ``True`` the simulation runs from :math:`H_A` to :math:`H_B`,
        otherwise it runs from :math:`H_B` to :math:`H_A`.
        :math:`H_B` is the cluster expansion and :math:`H_A = 0`, is the fully disordered system.
    boltzmann_constant
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e., units that are consistent
        with the underlying cluster expansion
        and the temperature units. Default: eV/K.
    user_tag
        Human-readable tag for ensemble. Default: ``None``.
    random_seed
        Seed for the random number generator used in the Monte Carlo simulation.
    dc_filename
        Name of file the data container associated with the ensemble
        will be written to. If the file exists it will be read, the
        data container will be appended, and the file will be
        updated/overwritten.
    data_container_write_period
        Period in units of seconds at which the data container is
        written to file. Writing periodically to file provides both
        a way to examine the progress of the simulation and to back up
        the data. Default: 600 s.
    ensemble_data_write_interval
        Interval at which data is written to the data container. This
        includes for example the current value of the calculator
        (i.e., usually the energy) as well as ensembles specific fields
        such as temperature or the number of atoms of different species.
        Default: Number of sites in the :attr:`structure`.
    trajectory_write_interval
        Interval at which the current occupation vector of the atomic
        configuration is written to the data container.
        Default: Number of sites in the :attr:`structure`.
    sublattice_probabilities
        Probability for picking a sublattice when doing a random swap.
        This should be as long as the number of sublattices and should
        sum up to 1.


    Example
    -------
    The following snippet illustrate how to carry out a simple thermodynamic
    integration. Here, the parameters of the cluster expansion are set to
    emulate a simple Ising model in order to obtain an
    example that can be run without modification. In practice, one should of
    course use a proper cluster expansion::

        >>> from ase.build import bulk
        >>> from icet import ClusterExpansion, ClusterSpace
        >>> from mchammer.calculators import ClusterExpansionCalculator

        >>> # prepare cluster expansion
        >>> # the setup emulates a second nearest-neighbor (NN) Ising model
        >>> # (zerolet and singlet ECIs are zero; only first and second neighbor
        >>> # pairs are included)
        >>> prim = bulk('Au')
        >>> cs = ClusterSpace(prim, cutoffs=[4.3], chemical_symbols=['Ag', 'Au'])
        >>> ce = ClusterExpansion(cs, [0, 0, 0.1, -0.02])

        >>> # prepare initial configuration
        >>> structure = prim.repeat(3)
        >>> for k in range(5):
        >>>     structure[k].symbol = 'Ag'

        >>> # set up and run MC simulation
        >>> calc = ClusterExpansionCalculator(structure, ce)
        >>> mc = ThermodynamicIntegrationEnsemble(structure=structure, calculator=calc,
        ...                                       temperature=600,
        ...                                       n_steps=100000,
        ...                                       forward=True,
        ...                                       dc_filename='myrun_thermodynamic_integration.dc')
        >>> mc.run()

    """

    def __init__(self,
                 structure: Atoms,
                 calculator: BaseCalculator,
                 temperature: float,
                 n_steps: int,
                 forward: bool,
                 user_tag: str = None,
                 boltzmann_constant: float = kB,
                 random_seed: int = None,
                 dc_filename: str = None,
                 data_container: str = None,
                 data_container_write_period: float = 600,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None,
                 sublattice_probabilities: List[float] = None,
                 ) -> None:

        self._ensemble_parameters = dict(temperature=temperature,
                                         n_steps=n_steps)
        self._last_state = dict()

        super().__init__(
                structure=structure,
                calculator=calculator,
                user_tag=user_tag,
                random_seed=random_seed,
                data_container=data_container,
                dc_filename=dc_filename,
                data_container_class=DataContainer,
                data_container_write_period=data_container_write_period,
                ensemble_data_write_interval=ensemble_data_write_interval,
                trajectory_write_interval=trajectory_write_interval,
                boltzmann_constant=boltzmann_constant)

        if sublattice_probabilities is None:
            self._swap_sublattice_probabilities = \
                self._get_swap_sublattice_probabilities()
        else:
            self._swap_sublattice_probabilities = sublattice_probabilities

        sublattices = []
        for sl in self.sublattices:
            sublattices.append(sl.atomic_numbers)

        # add species count to ensemble parameters
        symbols = set([symbol for sub in calculator.sublattices
                       for symbol in sub.chemical_symbols])
        for symbol in symbols:
            key = 'n_atoms_{}'.format(symbol)
            count = structure.get_chemical_symbols().count(symbol)
            self._ensemble_parameters[key] = count

        self._n_steps = n_steps

        if forward:
            self._lambda_function = _lambda_function_forward
            self._lambda = 0
        else:
            self._lambda_function = _lambda_function_backward
            self._lambda = 1

    @property
    def temperature(self) -> float:
        """ Current temperature. """
        return self._ensemble_parameters['temperature']

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        self._lambda = self._lambda_function(self.n_steps, self.step)
        sublattice_index = self.get_random_sublattice_index(self._swap_sublattice_probabilities)
        swap = self.do_thermodynamic_swap(sublattice_index=sublattice_index,
                                          lambda_val=self._lambda)
        return swap

    def run(self):
        """ Runs the thermodynamic integration. """
        if self.step >= self.n_steps:
            logger.warning('The simulation is already done')
        else:
            super().run(self.n_steps - self.step)

    def _get_ensemble_data(self):
        data = super()._get_ensemble_data()
        data['lambda'] = self._lambda
        return data
