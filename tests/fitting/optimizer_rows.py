import numpy as np
from icet.fitting import Optimizer

# setup Ax=y problem
N, M = 5000, 50
A = np.random.normal(0, 1.0, (N, M))
x_true = np.random.normal(0.0, 10.0, M)
y = np.dot(A, x_true) + np.random.normal(0.0, 0.2, N)

tol = 1.0 / N


# Only specify train_frac
train_frac = 0.8
opt = Optimizer((A, y), train_fraction=train_frac)
assert abs(opt.train_fraction - train_frac) < tol
assert abs(opt.test_fraction - 1.0 + train_frac) < tol

# Only specify test_frac
test_frac = 0.4
opt = Optimizer((A, y), train_fraction=None, test_fraction=test_frac)
assert abs(opt.test_fraction - test_frac) < tol
assert abs(opt.train_fraction - 1.0 + test_frac) < tol

# Specify both fracs
train_frac = 0.5
test_frac = 0.4
opt = Optimizer((A, y), train_fraction=train_frac, test_fraction=test_frac)
assert abs(opt.train_fraction - train_frac) < tol
assert abs(opt.test_fraction - test_frac) < tol


# Only specify training_set
train_frac = 0.3
training_set = np.random.choice(np.arange(N),
                                int(N * train_frac), replace=False)
opt = Optimizer((A, y), training_set=training_set)
assert np.all(opt.training_set == training_set)
assert abs(opt.train_fraction - train_frac) < tol
assert abs(opt.test_fraction - 1.0 + train_frac) < tol

# Only specify test_set
test_frac = 0.1
test_set = np.random.choice(np.arange(N), int(N * test_frac), replace=False)
opt = Optimizer((A, y), test_set=test_set)
assert np.all(opt.test_set == test_set)
assert abs(opt.test_fraction - test_frac) < tol
assert abs(opt.train_fraction - 1.0 + test_frac) < tol


# Specify both rows
train_frac = 0.6
test_frac = 0.2
training_set = np.random.choice(np.arange(N),
                                int(N * train_frac), replace=False)
test_set = np.random.choice(np.arange(N), int(N * test_frac), replace=False)
opt = Optimizer((A, y), training_set=training_set, test_set=test_set)
assert np.all(opt.training_set == training_set)
assert np.all(opt.test_set == test_set)
opt.train()
