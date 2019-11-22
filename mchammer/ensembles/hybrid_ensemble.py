from typing import Dict, List

import numpy as np
from math import isclose

from ase import Atoms
from ase.units import kB
from ase.data import atomic_numbers, chemical_symbols

from .. import DataContainer
from ..calculators.base_calculator import BaseCalculator
from .thermodynamic_base_ensemble import ThermodynamicBaseEnsemble
from .vcsgc_ensemble import get_phis
from .semi_grand_canonical_ensemble import get_chemical_potentials


class HybridEnsemble(ThermodynamicBaseEnsemble):
    """

    Instances of this class allows one to combine multiple ensembles.
    In particular, a dictionary should be provided for each ensemble,
    which must include the type (`ensemble`) as well as the index of
    the sublattice (`sublattice_index`). In addition, it is possible
    to provide a list of allowed symbols (`allowed_symbols`), which
    must represent a subset of the elements that can occupy the sites
    on the specified sublattice. Note that additional arguments are
    required for the SGC and VCSGC ensembles, namely chemical
    potentials (`chemical_potentials`) for the former and constraint
    parameters (`phis` and `kappa`) for the latter. For more detailed
    information regarding the different ensembles, please see
    :class:`CanonicalEnsemble <mchammer.ensembles.CanonicalEnsemble>`,
    :class:`SemiGrandCanonicalEnsemble
    <mchammer.ensembles.SemiGrandCanonicalEnsemble>`, and
    :class:`VCSGCEnsemble <mchammer.ensembles.VCSGCEnsemble>`.

    This class is particularly useful for effectively sampling complex
    multi-component systems with several active sublattices, in which
    case different ensembles can be defined for each of the latter.
    The fact that it is possible to set the allowed chemical symbols
    means that one can vary the concentrations of a few selected
    species, with the help of a VCSGC or semi-grand canonical
    ensemble, while still allowing swaps between any two sites, using
    a canonical ensemble (see also the below example). It is advisable
    to carefully consider how to define the ensemble probabilities. By
    default the ensembles are weighted by the sizes of the
    corresponding sublattices, which should give suitable
    probabilities in most cases. As is shown in the example below, it
    might be prudent to provide different values if allowed symbols
    are provided as well as for cases where there are several
    ensembles that are active on different sublattices.

    Parameters
    ----------

    structure : :class:`Atoms <ase.Atoms>`
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    calculator : :class:`BaseCalculator <mchammer.calculators.ClusterExpansionCalculator>`
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
    temperature : float
        temperature :math:`T` in appropriate units [commonly Kelvin]
    ensemble_specs: List[Dict]
        A list of dictionaries, which should contain the following items:

            * 'ensemble', which could be either "vcsgc"; "semigrand"
              or "canonical", lowercase and uppercase letters or any
              combination thereof are accepted (required)
            * 'sublattice_index', index for the sublattice of
              interest (required)
            * 'allowed_symbols', list of allowed chemical symbols
              (default: read from ClusterSpace)
            * 'chemical_potentials', a dictionary of chemical
              potentials for each species
              :math:`\\mu_i`; the key denotes the species, the value
              :specifies the chemical potential in units that are
              :consistent with the underlying cluster expansion (only
              :applicable and required for SGC ensembles)
            * 'phis ', dictionary with average constraint parameters
              ':math:`\\phi_i`; the key denotes the species; for a
              N-component sublattice, there should be N - 1
              different `\\phi_i` (referred to as
              :math:`\\bar{\\phi}` in [SadErh12]_; only applicable
              and required for VCSGC ensembles, see also
              :class:`VCSGCEnsemble <mchammer.ensembles.VCSGCEnsemble>`)
            * 'kappa', parameter that constrains the variance of the
              'concentration (referred to as
              :math:`\\bar{\\kappa}` in [SadErh12]_; only applicable
              :and required for VCSGC ensembles)

    probabilities: List[float]
        list of floats with the probabilities for choosing a
        particular ensemble with the same length as ensemble specs.
        If left unspecified the probabilties are weighted by the
        sizes of the associated sublattices
    boltzmann_constant : float
        Boltzmann constant :math:`k_B` in appropriate units, i.e.
        units that are consistent with the underlying cluster
        expansion and the temperature units [default: eV/K]
    user_tag : str
        human-readable tag for ensemble [default: None]
    random_seed : int
        seed for the random number generator used in the Monte Carlo
        simulation
    data_container : str
        name of file the data container associated with the
        ensemble will be written to; if the file
        exists it will be read, the data container will be appended,
        and the file will be updated/overwritten
    data_container_write_period : float
        period in units of seconds at which the data container is
        written to file; writing periodically to file provides both
        a way to examine the progress of the simulation and to
        back up the data [default: np.inf]
    ensemble_data_write_interval : int
        interval at which data is written to the data container;
        this includes for example the current value of the
        calculator (i.e. usually the energy) as well as ensembles
        specific fields such as temperature or the number of atoms
        of different species
    trajectory_write_interval : int
        interval at which the current occupation vector of the
        atomic configuration is written to the data container.

    Example
    -------
    The following snippet illustrates how to carry out a simple Monte Carlo
    simulation using a combination of one canonical and one VCSGC ensemble.
    Specifically, the concentration of one species (Au) is kept constant
    while the others (Ag and Pd) are varied, while swaps are still allowed.
    Here, the parameters of the cluster expansion are set to emulate a simple
    Ising model in order to obtain an example that can be run without
    modification. In practice, one should of course use a proper cluster
    expansion::

        from ase.build import bulk
        from icet import ClusterExpansion, ClusterSpace
        from mchammer.calculators import ClusterExpansionCalculator
        from mchammer.ensembles import HybridEnsemble

        # prepare cluster expansion
        # the setup emulates a second nearest-neighbor (NN) Ising model
        # (zerolet and singlet ECIs are zero; only first and second neighbor
        # pairs are included)
        prim = bulk('Au')
        cs = ClusterSpace(prim, cutoffs=[4.3],
                          chemical_symbols=['Ag', 'Au', 'Pd'])
        ce = ClusterExpansion(
            cs, [0, 0, 0, 0.1, 0.1, 0.1, -0.02, -0.02, -0.02])

        # define structure object
        structure = prim.repeat(3)
        for i, atom in enumerate(structure):
            if i % 2 == 0:
                atom.symbol = 'Ag'
            elif i % 3 == 0:
                atom.symbol = 'Pd'

        # the default probabilities for this case would be [0.5, 0.5], but
        # since the VCSGC ensemble only performs flips on a subset of all
        # sites on the sublattice, namely those originally occupied by Ag
        # and Pd atoms, specific values will be provided
        weights = [len(structure),
                   len([s for s in structure.get_chemical_symbols() if s != 'Au'])]
        norm = sum(weights)
        probabilities = [w / norm for w in weights]

        # set up and run MC simulation
        calc = ClusterExpansionCalculator(structure, ce)
        ensemble_specs = [
            {'ensemble': 'canonical', 'sublattice_index': 0},
            {'ensemble': 'vcsgc', 'sublattice_index': 0,
             'phis': {'Ag': -0.2}, 'kappa': 200,
             'allowed_symbols':['Ag', 'Pd']}]
        mc = HybridEnsemble(structure=structure, calculator=calc,
                            ensemble_specs=ensemble_specs,
                            temperature=600, probabilities=probabilities,
                            data_container='myrun_hybrid.dc')
        mc.run(100)  # carry out 100 trial steps
    """

    def __init__(self,
                 structure: Atoms,
                 calculator: BaseCalculator,
                 temperature: float,
                 ensemble_specs: List[Dict],
                 probabilities: List[float] = None,
                 boltzmann_constant: float = kB,
                 user_tag: str = None,
                 random_seed: int = None,
                 data_container: str = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None) -> None:

        # define available ensembles
        self._ensemble_trial_steps = dict([
            ('canonical', self.do_canonical_swap),
            ('semi-grand', self.do_sgc_flip),
            ('vcsgc', self.do_vcsgc_flip),
            ])

        self._ensemble_parameters = dict(temperature=temperature)

        self._trial_steps_per_ensemble = {"ensemble_{}".format(i): 0 for i in
                                          range(len(ensemble_specs))}

        # process the list of ensembles and parameters
        self._process_ensemble_specs(ensemble_specs)

        super().__init__(
            structure=structure,
            calculator=calculator,
            user_tag=user_tag,
            random_seed=random_seed,
            boltzmann_constant=boltzmann_constant,
            data_container=data_container,
            data_container_class=DataContainer,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

        # postprocess the list of ensembles and parameters
        self._postprocess_ensemble_args()

        # set the probabilities
        self._process_probabilities(probabilities)

    @property
    def temperature(self) -> float:
        """ Current temperature """
        return self._ensemble_parameters['temperature']

    @property
    def probabilities(self) -> float:
        """ Ensemble propabilities """
        return self._probabilities

    @property
    def trial_steps_per_ensemble(self) -> Dict[str, float]:
        """ Number of Monte Carlo trial steps for each ensemble """
        return self._trial_steps_per_ensemble

    def _process_ensemble_specs(
            self, ensemble_specs: List[Dict]):
        """Process the list of ensembles and parameters

        Parameters
        ----------
         ensemble_specs: List[Dict]
            A list of dictionaries, which should contain the following items:
            * 'ensemble', which could be either "vcsgc"; "semigrand" or "canonical", lowercase and
            upercase letters or any combination thereof are accepted
            * 'sublattice_index', index for the sublattice of interest
            * 'allowed_symbols', list of allowed chemical symbols
            * 'chemical_potentials', a dictionary of chemical potentials for each species
            :math:`\\mu_i`; the key denotes the species, the value specifies the chemical potential
            in units that are consistent with the underlying cluster expansion
            * 'phis ', dictionary with average constraint parameters :math:`\\phi_i`; the key
            denotes the species; for a N-component sublattice, there should be N - 1 different
            `\\phi_i`
            * 'kappa', parameter that constrains the variance of the concentration
        """

        ensemble_args = []

        for ind, ensemble_spec in enumerate(ensemble_specs):

            ensemble_arg = {}
            tag = "ensemble_{}".format(ind)
            ensemble_arg['tag'] = tag

            # check the ensemble name
            if 'ensemble' not in ensemble_spec:
                raise ValueError("The dictionary {} lacks the required key"
                                 " 'ensemble'".format(ensemble_spec))
            ensemble = ensemble_spec['ensemble'].lower()
            if ensemble not in self._ensemble_trial_steps.keys():
                msg = ['Ensemble not available']
                msg += ['Please choose one of the following:']
                for key in self._ensemble_trial_steps.keys():
                    msg += [' * ' + key]
                raise ValueError('\n'.join(msg))
            ensemble_arg['ensemble'] = ensemble
            self._ensemble_parameters[tag] = ensemble

            # check that all required keys, and no unknown keys, are present
            keys = ['ensemble', 'sublattice_index', 'allowed_symbols']
            if ensemble == 'semi-grand':
                keys = ['chemical_potentials'] + keys
            elif ensemble == 'vcsgc':
                keys = ['phis', 'kappa'] + keys
            for key in keys[:-1]:
                if key not in ensemble_spec:
                    raise ValueError("The dictionary {} lacks the key '{}', which is required for"
                                     " {} ensembles".format(ensemble_spec, key, ensemble))
            for key in ensemble_spec.keys():
                if key not in keys:
                    raise ValueError("Unknown key '{}', for a {} ensemble,"
                                     " in the dictionary {}".format(key, ensemble, ensemble_spec))

            # record the sublattice index
            ensemble_arg['sublattice_index'] = ensemble_spec['sublattice_index']

            # process chemical potentials
            if 'chemical_potentials' in ensemble_spec:
                chemical_potentials = get_chemical_potentials(ensemble_spec['chemical_potentials'])
                ensemble_arg['chemical_potentials'] = chemical_potentials
                for atnum, chempot in chemical_potentials.items():
                    mu_sym = '{}_mu_{}'.format(tag, chemical_symbols[atnum])
                    self._ensemble_parameters[mu_sym] = chempot

            # process phis
            if 'phis' in ensemble_spec:
                phis = get_phis(ensemble_spec['phis'])
                ensemble_arg['phis'] = phis
                for sym, phi in phis.items():
                    if isinstance(sym, str):
                        chemical_symbol = sym
                    else:
                        chemical_symbol = chemical_symbols[sym]
                    phi_sym = '{}_phi_{}'.format(tag, chemical_symbol)
                    self._ensemble_parameters[phi_sym] = phi

            # process kappa
            if 'kappa' in ensemble_spec:
                ensemble_arg['kappa'] = ensemble_spec['kappa']
                self._ensemble_parameters['{}_kappa'.format(tag)] = ensemble_spec['kappa']

            # record the allowed chemical symbols
            if 'allowed_symbols' in ensemble_spec:
                ensemble_arg['allowed_symbols'] = ensemble_spec['allowed_symbols']

            ensemble_args.append(ensemble_arg)

        self._ensemble_args = ensemble_args

    def _postprocess_ensemble_args(self):
        """Process the list of dictionaries with ensemble specific parameters
        """

        for i in range(len(self._ensemble_args)):

            # check the sublattice index
            self._check_sublattice_index(self._ensemble_args[i]['sublattice_index'])

            # extract the allowed species
            if 'allowed_symbols' in self._ensemble_args[i]:
                self._ensemble_args[i]['allowed_species'] =\
                    self._extract_allowed_species(self._ensemble_args[i]['allowed_symbols'],
                                                  self._ensemble_args[i]['sublattice_index'])
                del self._ensemble_args[i]['allowed_symbols']
            else:
                self._ensemble_args[i]['allowed_species'] = None

            if self._ensemble_args[i]['ensemble'] == 'vcsgc':
                # Check that each sublattice has N - 1 phis
                count_specified_elements = 0
                if self._ensemble_args[i]['allowed_species'] is None:
                    allowed_species =\
                        self.sublattices[self._ensemble_args[i]['sublattice_index']].atomic_numbers
                else:
                    allowed_species = self._ensemble_args[i]['allowed_species']
                for number in allowed_species:
                    if number in self._ensemble_args[i]['phis'].keys():
                        count_specified_elements += 1
                if count_specified_elements != len(allowed_species) - 1:
                    raise ValueError("phis must be set for N - 1 elements on a sublattice with"
                                     " N elements")

    def _check_sublattice_index(self, sublattice_index: int):
        """Check the 'sublattice_index' item in the 'ensemble_spec' dictionary

        Parameters
        ----------
        sublattice_index:
            Specific sublattice to consider provided as as an index or a symbol
        """

        if not isinstance(sublattice_index, int):
            raise TypeError("'sublattice_index' must be an integer, not"
                            " {}".format(type(sublattice_index)))

        # check that the sublattice exists
        if sublattice_index not in range(len(self.sublattices)):
            raise ValueError("There is no sublattice with index {}".format(sublattice_index))

        # check that the sublattice is active
        if len(self.sublattices[sublattice_index].chemical_symbols) == 1:
            raise ValueError("The sublattice {} is inactive".format(sublattice_index))

    def _extract_allowed_species(self, allowed_symbols: List[str], sublattice_index: int
                                 ) -> List[int]:
        """Check and extract the allowed species from the 'allowed_symbols' in the 'ensemble_spec'
        dictionary

        Parameters
        ----------
        allowed_symbols:
            list of allowed chemical symbols
        sublattice_index:
            Index for the relevant sublattice
        """

        if not isinstance(allowed_symbols, list) or not all(
                [isinstance(i, str) for i in allowed_symbols]):
            raise TypeError(
                "'allowed_symbols' must be a List[str], not {}".format(type(allowed_symbols)))
        for symbol in allowed_symbols:
            if symbol not in self.sublattices[sublattice_index].chemical_symbols:
                raise ValueError("The species {} is not allowed on sublattice"
                                 " {}".format(symbol, sublattice_index))

        return [atomic_numbers[s] for s in allowed_symbols]

    def _process_probabilities(self, probabilities: List[float]):
        """Process the list of probabilities

        Parameters
        ----------
        probabilities:
            list of floats with the probabilities for choosing a particular ensemble with the same
            length as self._ensemble_args.
        """

        if probabilities is None:
            # use the sizes of the associated sublattices when calculating the ensemble
            # probabilities
            weights = [len(self.sublattices[ensemble_arg['sublattice_index']].indices) for
                       ensemble_arg in self._ensemble_args]
            norm = sum(weights)
            probabilities = [w / norm for w in weights]
        else:
            if len(probabilities) != len(self._ensemble_args):
                raise ValueError("The number of probabilities must be match the number of"
                                 " ensembles")

            if not isclose(sum(probabilities), 1.0):
                raise ValueError("The sum of all probabilities must be equal to 1")

        self._probabilities = probabilities

    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """

        # randomly pick an ensemble
        ensemble_arg = np.random.choice(self._ensemble_args, p=self._probabilities)

        # count number of trial steps for each ensemble
        self._trial_steps_per_ensemble[ensemble_arg['tag']] += 1

        if ensemble_arg['ensemble'] == 'canonical' and not self.configuration.is_swap_possible(
                ensemble_arg['sublattice_index'], ensemble_arg['allowed_species']):
            return 0
        else:
            arguments = {key: val for key, val in ensemble_arg.items() if key not in
                         ['ensemble', 'tag']}
            return self._ensemble_trial_steps[ensemble_arg['ensemble']](**arguments)

    def _get_ensemble_data(self) -> Dict:
        """
        Returns a dict with the default data of the ensemble. This includes
        atom counts and free energy derivative.
        """
        data = super()._get_ensemble_data()

        ensemble_types = [e['ensemble'] for e in self._ensemble_args]

        # free energy derivative
        if 'vcsgc' in ensemble_types:
            for ensemble_arg in self._ensemble_args:
                if 'vcsgc' == ensemble_arg['ensemble']:
                    data.update(self._get_vcsgc_free_energy_derivatives(
                        ensemble_arg['phis'], ensemble_arg['kappa'],
                        ensemble_arg['sublattice_index']))

        # species counts
        if any([e in ensemble_types for e in ['vcsgc', 'semi-grand']]):
            data.update(self._get_species_counts())

        return data
