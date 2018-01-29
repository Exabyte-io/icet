import numpy as np
from icet.fitting import Optimizer


def test_sizes(train_frac=None, test_frac=None):
    if train_frac is None:
        train_frac = 1.0 - test_frac
    if test_frac is None:
        test_frac = 1.0 - train_frac

    for train_size in [train_frac, int(train_frac*N)]:
        for test_size in [test_frac, int(test_frac*N)]:
            opt = Optimizer(
                (A, y), training_size=train_size, test_size=test_size)
            assert abs(opt.training_fraction - train_frac) < tol
            assert abs(opt.test_fraction - test_frac) < tol


def test_sets(train_frac=None, test_frac=None):

    # setup training and test set given the fractions
    test_set, training_set = None, None
    if train_frac is not None:
        training_set = np.random.choice(
            np.arange(N), int(N * train_frac), replace=False)
    if test_frac is not None:
        test_set = np.random.choice(
            np.arange(N), int(N * test_frac), replace=False)

    opt = Optimizer((A, y), training_set=training_set, test_set=test_set)
    # test training set
    if training_set is not None:
        assert np.all(opt.training_set == training_set)
        assert abs(opt.training_fraction - train_frac) < tol
    # test training set
    if test_set is not None:
        assert np.all(opt.test_set == test_set)
        assert abs(opt.test_fraction - test_frac) < tol


# setup dummy Ax=y problem
# ------------------------
N, M = 5000, 50
A = np.random.normal(0, 1.0, (N, M))
x_true = np.random.normal(0.0, 10.0, M)
y = np.dot(A, x_true) + np.random.normal(0.0, 0.2, N)

tol = 1.0 / N


# Try training_size and test_size
# --------------------------------

# Only specify train_frac
train_frac, test_frac = 0.8, None
test_sizes(train_frac=train_frac, test_frac=test_frac)

# Only specify test_frac
train_frac, test_frac = None, 0.4
test_sizes(train_frac=train_frac, test_frac=test_frac)

# Specify both fracs
train_frac, test_frac = 0.5, 0.3
test_sizes(train_frac=train_frac, test_frac=test_frac)


# Try training_set and test_set
# -----------------------------

# Only specify training_set
train_frac, test_frac = 0.8, None
test_sets(train_frac, test_frac)

# Only specify test_set
train_frac, test_frac = None, 0.4
test_sets(train_frac, test_frac)

# Specify both training and test set
train_frac, test_frac = 0.5, 0.3
test_sets(train_frac, test_frac)
