import numpy as np
from icet.fitting import Optimizer, available_fit_methods

np.random.seed(42)

# setup Ax=y problem
N, M = 200, 50
A = np.random.normal(0, 1.0, (N, M))
x_true = np.random.normal(0.0, 10.0, M)
y = np.dot(A, x_true) + np.random.normal(0.0, 0.2, N)

# test all fit_methods
for fit_method in available_fit_methods:
    opt = Optimizer((A, y), fit_method=fit_method, train_fraction=0.8)
    opt.train()
    assert (np.linalg.norm(x_true - opt.parameters)) < 0.2, fit_method
