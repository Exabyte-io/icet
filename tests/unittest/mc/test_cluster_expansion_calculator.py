import unittest

from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators.cluster_expansion_calculator import \
    ClusterExpansionCalculator
from _icet import _ClusterExpansionCalculator
from icet import Structure


class TestCECalculatorBinary(unittest.TestCase):
    """
    Container for tests of the class functionality.

    Todo
    ----
        * add property test to calculate local contribution when that
          method has been added as intended.

    """

    def __init__(self, *args, **kwargs):
        super(TestCECalculatorBinary, self).__init__(*args, **kwargs)

        self.atoms = bulk('Al', 'fcc', a=4.0)
        self.cutoffs = [5, 5]  # [2.9]
        self.subelements = ['Al', 'Ge']
        self.cs = ClusterSpace(self.atoms, self.cutoffs, self.subelements)
        params_len = self.cs.get_cluster_space_size()
        params = [1.1] * params_len

        self.ce = ClusterExpansion(self.cs, params)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.atoms = bulk('Al', 'fcc', a=4.0).repeat(2)

        self.calculator = ClusterExpansionCalculator(
            self.atoms, self.ce, name='Tests CE calc')

    def test_property_cluster_expansion(self):
        """Tests the cluster expansion property."""
        self.assertIsInstance(
            self.calculator.cluster_expansion, ClusterExpansion)

    def _test_flip_changes(self, msg):
        """Tests differences when flipping."""
        for i in range(len(self.atoms)):
            indices = [i]
            local_diff, total_diff = self._get_energy_diffs_local_and_total(
                indices)
            self.assertAlmostEqual(total_diff, local_diff, msg=msg)

    def _test_swap_changes(self, msg):
        """Tests differences when swapping."""
        for i in range(len(self.atoms)):
            for j in range(len(self.atoms)):
                if j <= i:
                    continue
                indices = [i, j]
                local_diff, total_diff = \
                    self._get_energy_diffs_local_and_total(indices)
                self.assertAlmostEqual(total_diff, local_diff, msg=msg)

    def test_local_contribution_flip(self):
        """Tests potential differences when flipping."""
        # Tests original occupations
        self._test_flip_changes("original occupations")

        # Tests checkerboard-ish
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32

        self._test_flip_changes("Checkerboard")

        # Tests seggregated-ish
        for i in range(len(self.atoms)):
            if i < len(self.atoms)/2:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32
        self._test_flip_changes("Segregated")

    def test_local_contribution_swap(self):
        """Tests correct differences when swapping."""
        # Tests original occupations
        self._test_swap_changes("Original occupations")

        # Tests checkerboard-ish
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32

        self._test_swap_changes("checkerboard")

        # Tests seggregated-ish
        for i in range(len(self.atoms)):
            if i < len(self.atoms)/2:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32
        self._test_swap_changes("segregated")

    def _get_energy_diffs_local_and_total(self, indices):
        """Get energy diffs using local and total."""

        # Original occupations
        original_occupations = self.atoms.numbers.copy()
        # Initial value total energy
        initial_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers())
        # Initial value local energy
        initial_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices, occupations=self.atoms.get_atomic_numbers())
        # Flip indices

        for index in indices:
            if self.atoms[index].number == 13:
                self.atoms[index].number = 32
            elif self.atoms[index].number == 32:
                self.atoms[index].number = 13

        # Calculate new total energy
        new_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers().copy())

        # Calculate new local energy
        new_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices,
            occupations=self.atoms.get_atomic_numbers().copy())

        # difference in energy according to total energy
        total_diff = new_value_total - initial_value_total

        # Difference in energy according to local energy
        local_diff = new_value_local - initial_value_local

        # Reset occupations
        self.atoms.set_atomic_numbers(original_occupations.copy())

        return local_diff, total_diff

    def test_calculate_local_contribution(self):
        """Tests calculate local contribution."""
        indices = [3, 5]
        local_contribution = self.calculator.calculate_local_contribution(
            local_indices=indices, occupations=self.atoms.get_atomic_numbers())
        self.assertIsInstance(local_contribution, float)

        # test local contribution by comparing with differences
        original_occupations = self.atoms.numbers.copy()
        initial_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers())

        self.atoms.set_atomic_numbers(original_occupations.copy())
        initial_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices, occupations=self.atoms.get_atomic_numbers())
        self.atoms.set_atomic_numbers(original_occupations.copy())
        current_occupations = [self.atoms.get_atomic_numbers()[i]
                               for i in indices]
        self.atoms.set_atomic_numbers(original_occupations.copy())
        swapped_elements = []
        for atom in current_occupations:
            if atom == 13:
                swapped_elements.append(32)
            elif atom == 32:
                swapped_elements.append(13)
            else:
                raise Exception(
                    'Found unknown element in atoms object. {}'.format(atom))

        new_occupations = self.atoms.get_atomic_numbers().copy()
        for index, element in zip(indices, swapped_elements):
            new_occupations[index] = element
        self.atoms.set_atomic_numbers(new_occupations.copy())
        new_value_total = self.calculator.calculate_total(
            occupations=new_occupations.copy())
        self.atoms.set_atomic_numbers(new_occupations.copy())
        new_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices, occupations=new_occupations.copy())
        self.atoms.set_atomic_numbers(new_occupations.copy())

        total_diff = new_value_total - initial_value_total
        local_diff = new_value_local - initial_value_local
        self.assertAlmostEqual(total_diff, local_diff)

    def test_get_local_cluster_vector(self):
        """Tests the get local clustervector method."""

        cpp_calc = _ClusterExpansionCalculator(
            self.cs, Structure.from_atoms(self.atoms))

        index = 4
        cpp_calc.get_local_cluster_vector(
            self.atoms.get_atomic_numbers(), index, [])

        self.atoms[index].symbol = 'Ge'

        cpp_calc.get_local_cluster_vector(
            self.atoms.get_atomic_numbers(), index, [])


class TestCECalculatorBinaryHCP(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestCECalculatorBinaryHCP,
              self).__init__(*args, **kwargs)

        self.atoms = bulk('Al', 'hcp', a=4.0, c=3.1)
        self.cutoffs = [6, 6, 6]  # [2.9]
        self.subelements = ['Al', 'Ge']
        self.cs = ClusterSpace(
            self.atoms.copy(), self.cutoffs, self.subelements)
        params_len = self.cs.get_cluster_space_size()
        params = [1.0] * params_len

        self.ce = ClusterExpansion(self.cs, params)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.atoms = bulk('Al', 'hcp', a=4.0, c=3.1).repeat(2)

        self.calculator = ClusterExpansionCalculator(
            self.atoms, self.ce, name='Tests CE calc')

    def _test_flip_changes(self, msg):
        """Tests differences when flipping."""
        for i in range(len(self.atoms)):
            indices = [i]
            local_diff, total_diff = self._get_energy_diffs_local_and_total(
                indices)
            msg += ", indices " + str(indices) + \
                ", len of atoms " + str(len(self.atoms))
            self.assertAlmostEqual(total_diff, local_diff, msg=msg)

    def _test_swap_changes(self, msg):
        """Tests differences when flipping."""
        for i in range(len(self.atoms)):
            for j in range(len(self.atoms)):
                if j <= i:
                    continue
                indices = [i, j]
                local_diff, total_diff = \
                    self._get_energy_diffs_local_and_total(indices)
                msg += ", indices " + \
                    str(indices) + ", len of atoms " + str(len(self.atoms))
                self.assertAlmostEqual(total_diff, local_diff, msg=msg)

    def test_local_contribution_flip(self):
        """Tests potential differences when flipping."""
        # Tests original occupations
        self._test_flip_changes("original occupations")

        # Tests checkerboard-ish
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32

        self._test_flip_changes("Checkerboard")

        # Tests segregated-ish
        for i in range(len(self.atoms)):
            if i < len(self.atoms)/2:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32
        self._test_flip_changes("Segregated")

    def test_local_contribution_swap(self):
        """Tests correct differences when swapping."""
        # Tests original occupations
        self._test_swap_changes("Original occupations")

        # Tests checkerboard-ish
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32

        self._test_swap_changes("checkerboard")

        # Tests segregated-ish
        for i in range(len(self.atoms)):
            if i < len(self.atoms)/2:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32
        self._test_swap_changes("segregated")

    def _get_energy_diffs_local_and_total(self, indices):
        """Gets energy diffs using local and total."""

        # Original occupations
        original_occupations = self.atoms.numbers.copy()
        # Initial value total energy
        initial_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers())
        # Initial value local energy
        initial_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices, occupations=self.atoms.get_atomic_numbers())
        # Flip indices

        for index in indices:
            if self.atoms[index].number == 13:
                self.atoms[index].number = 32
            elif self.atoms[index].number == 32:
                self.atoms[index].number = 13
        # Calculate new total energy
        new_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers().copy())

        # Calculate new local energy
        new_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices,
            occupations=self.atoms.get_atomic_numbers().copy())

        # difference in energy according to total energy
        total_diff = new_value_total - initial_value_total

        # Difference in energy according to local energy
        local_diff = new_value_local - initial_value_local

        # Reset occupations
        self.atoms.set_atomic_numbers(original_occupations.copy())

        return local_diff, total_diff


class TestCECalculatorBinaryBCC(unittest.TestCase):
    """
    Container for tests of the class functionality.

    Todo
    ----
        * add property test to calculate local contribution when that
          method has been added as intended.

    """

    def __init__(self, *args, **kwargs):
        super(TestCECalculatorBinaryBCC,
              self).__init__(*args, **kwargs)

        self.atoms = bulk('Al', 'bcc', a=4.0)
        self.cutoffs = [6, 6, 6]
        self.subelements = ['Al', 'Ge']
        self.cs = ClusterSpace(self.atoms, self.cutoffs, self.subelements)
        params_len = self.cs.get_cluster_space_size()
        params = [1.1] * params_len

        self.ce = ClusterExpansion(self.cs, params)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.atoms = bulk('Al', 'bcc', a=4.0).repeat(2)
        self.calculator = ClusterExpansionCalculator(
            self.atoms, self.ce, name='Tests CE calc')

    def _test_flip_changes(self, msg):
        """Tests differences when flipping."""
        for i in range(len(self.atoms)):
            indices = [i]
            local_diff, total_diff = self._get_energy_diffs_local_and_total(
                indices)
            self.assertAlmostEqual(total_diff, local_diff, msg=msg)

    def _test_swap_changes(self, msg):
        """Tests differences when flipping."""
        for i in range(len(self.atoms)):
            for j in range(len(self.atoms)):
                if j <= i:
                    continue
                indices = [i, j]
                local_diff, total_diff = \
                    self._get_energy_diffs_local_and_total(indices)
                self.assertAlmostEqual(total_diff, local_diff, msg=msg)

    def test_local_contribution_flip(self):
        """Tests potential differences when flipping."""

        # Tests original occupations
        self._test_flip_changes("original occupations")

        # Tests checkerboard-ish
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32

        self._test_flip_changes("Checkerboard")

        # Tests segregated-ish
        for i in range(len(self.atoms)):
            if i < len(self.atoms)/2:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32
        self._test_flip_changes("Segregated")

    def test_local_contribution_swap(self):
        """Tests correct differences when swapping."""
        # Tests original occupations
        self._test_swap_changes("Original occupations")

        # Tests checkerboard-ish
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32

        self._test_swap_changes("checkerboard")

        # Tests segregated-ish
        for i in range(len(self.atoms)):
            if i < len(self.atoms)/2:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32
        self._test_swap_changes("segregated")

    def _get_energy_diffs_local_and_total(self, indices):
        """Gets energy diffs using local and total."""

        # Original occupations
        original_occupations = self.atoms.numbers.copy()
        # Initial value total energy
        initial_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers())
        # Initial value local energy
        initial_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices, occupations=self.atoms.get_atomic_numbers())
        # Flip indices
        for index in indices:
            if self.atoms[index].number == 13:
                self.atoms[index].number = 32
            elif self.atoms[index].number == 32:
                self.atoms[index].number = 13
        # Calculate new total energy
        new_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers().copy())

        # Calculate new local energy
        new_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices,
            occupations=self.atoms.get_atomic_numbers().copy())

        # difference in energy according to total energy
        total_diff = new_value_total - initial_value_total

        # Difference in energy according to local energy
        local_diff = new_value_local - initial_value_local

        # Reset occupations
        self.atoms.set_atomic_numbers(original_occupations.copy())

        return local_diff, total_diff


class TestCECalculatorTernaryBCC(unittest.TestCase):
    """
    Container for tests of the class functionality.

    Todo
    ----
        * add property test to calculate local contribution when that
          method has been added as intended.

    """

    def __init__(self, *args, **kwargs):
        super(TestCECalculatorTernaryBCC,
              self).__init__(*args, **kwargs)

        self.atoms = bulk('Al', 'bcc', a=4.0)
        self.cutoffs = [6, 6, 6]
        self.subelements = ['Al', 'Ge', 'H']
        self.cs = ClusterSpace(self.atoms, self.cutoffs, self.subelements)
        params_len = self.cs.get_cluster_space_size()
        params = [1.0] * params_len
        self.ce = ClusterExpansion(self.cs, params)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.atoms = bulk('Al', 'bcc', a=4.0).repeat(2)
        self.calculator = ClusterExpansionCalculator(
            self.atoms, self.ce, name='Tests CE calc')

    def _test_flip_changes(self, msg):
        """Tests differences when flipping."""
        for i in range(len(self.atoms)):
            indices = [i]
            local_diff, total_diff = self._get_energy_diffs_local_and_total(
                indices)
            self.assertAlmostEqual(total_diff, local_diff, msg=msg)

    def _test_swap_changes(self, msg):
        """Tests differences when swapping."""
        for i in range(len(self.atoms)):
            for j in range(len(self.atoms)):
                if j <= i:
                    continue
                indices = [i, j]
                local_diff, total_diff = \
                    self._get_energy_diffs_local_and_total(indices)
                msg1 = "[{}, {}]".format(i, j)
                self.assertAlmostEqual(total_diff, local_diff, msg=msg1)

    def test_local_contribution_flip(self):
        """Tests potential differences when flipping."""

        # Tests original occupations
        self._test_flip_changes("original occupations")

        # Tests checkerboard-ish
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32

        self._test_flip_changes("Checkerboard")

        # Tests segregated-ish
        for i in range(len(self.atoms)):
            if i < len(self.atoms)/2:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32
        self._test_flip_changes("Segregated")

    def test_local_contribution_swap(self):
        """Tests correct differences when swapping."""
        # Tests original occupations
        self._test_swap_changes("Original occupations")

        # Tests checkerboard-ish
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32

        self._test_swap_changes("checkerboard")

        # Tests segregated-ish
        for i in range(len(self.atoms)):
            if i < len(self.atoms)/2:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32
        self._test_swap_changes("segregated")

    def _get_energy_diffs_local_and_total(self, indices):
        """Gets energy diffs using local and total."""

        # Original occupations
        original_occupations = self.atoms.numbers.copy()
        # Initial value total energy
        initial_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers())
        # Initial value local energy
        initial_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices, occupations=self.atoms.get_atomic_numbers())
        # Flip indices

        for index in indices:
            if self.atoms[index].number == 13:
                self.atoms[index].number = 32
            elif self.atoms[index].number == 32:
                self.atoms[index].number = 13
        # Calculate new total energy
        new_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers().copy())

        # Calculate new local energy
        new_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices,
            occupations=self.atoms.get_atomic_numbers().copy())

        # Difference in energy according to total energy
        total_diff = new_value_total - initial_value_total

        # Difference in energy according to local energy
        local_diff = new_value_local - initial_value_local

        self.atoms.set_atomic_numbers(original_occupations.copy())

        return local_diff, total_diff


class TestCECalculatorTernaryHCP(unittest.TestCase):
    """
    Container for tests of the class functionality.

    Todo
    ----
        * add property test to calculate local contribution when that
          method has been added as intended.

    """

    def __init__(self, *args, **kwargs):
        super(TestCECalculatorTernaryHCP,
              self).__init__(*args, **kwargs)

        self.atoms = bulk('Al', 'hcp', a=4.0, c=3.1)
        self.cutoffs = [6, 6, 6]
        self.subelements = ['Al', 'Ge', 'H']
        self.cs = ClusterSpace(self.atoms, self.cutoffs, self.subelements)
        params_len = self.cs.get_cluster_space_size()
        params = [1.0] * params_len
        self.ce = ClusterExpansion(self.cs, params)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.atoms = bulk('Al', 'hcp', a=4.0, c=3.1).repeat(2)
        self.calculator = ClusterExpansionCalculator(
            self.atoms, self.ce, name='Tests CE calc')

    def _test_flip_changes(self, msg):
        """Tests differences when flipping."""
        for i in range(len(self.atoms)):
            indices = [i]
            local_diff, total_diff = self._get_energy_diffs_local_and_total(
                indices)
            self.assertAlmostEqual(total_diff, local_diff, msg=msg)

    def _test_swap_changes(self, msg):
        """Tests differences when swapping."""
        for i in range(len(self.atoms)):
            for j in range(len(self.atoms)):
                if j <= i:
                    continue
                indices = [i, j]
                local_diff, total_diff = \
                    self._get_energy_diffs_local_and_total(indices)
                msg1 = "[{}, {}]".format(i, j)
                self.assertAlmostEqual(total_diff, local_diff, msg=msg1)

    def test_local_contribution_flip(self):
        """Tests potential differences when flipping."""
        # Tests original occupations
        self._test_flip_changes("original occupations")

        # Tests checkerboard-ish
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32

        self._test_flip_changes("Checkerboard")

        # Tests segregated-ish
        for i in range(len(self.atoms)):
            if i < len(self.atoms)/2:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32
        self._test_flip_changes("Segregated")

    def test_local_contribution_swap(self):
        """Tests correct differences when swapping."""
        # Tests original occupations
        self._test_swap_changes("Original occupations")

        # Tests checkerboard-ish
        for i in range(len(self.atoms)):
            if i % 2 == 0:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32

        self._test_swap_changes("checkerboard")

        # Tests segregated-ish
        for i in range(len(self.atoms)):
            if i < len(self.atoms)/2:
                self.atoms[i].number = 13
            else:
                self.atoms[i].number = 32
        self._test_swap_changes("segregated")

    def _get_energy_diffs_local_and_total(self, indices):
        """Gets energy diffs using local and total."""

        # Original occupations
        original_occupations = self.atoms.numbers.copy()
        # Initial value total energy
        initial_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers())
        # Initial value local energy
        initial_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices, occupations=self.atoms.get_atomic_numbers())

        # Flip indices
        for index in indices:
            if self.atoms[index].number == 13:
                self.atoms[index].number = 32
            elif self.atoms[index].number == 32:
                self.atoms[index].number = 13
        # Calculate new total energy
        new_value_total = self.calculator.calculate_total(
            occupations=self.atoms.get_atomic_numbers().copy())

        # Calculate new local energy
        new_value_local = self.calculator.calculate_local_contribution(
            local_indices=indices,
            occupations=self.atoms.get_atomic_numbers().copy())

        # Difference in energy according to total energy
        total_diff = new_value_total - initial_value_total

        # Difference in energy according to local energy
        local_diff = new_value_local - initial_value_local

        self.atoms.set_atomic_numbers(original_occupations.copy())

        return local_diff, total_diff


if __name__ == '__main__':
    unittest.main()
