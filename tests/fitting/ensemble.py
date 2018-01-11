'''
Simple test for using ensemble optimizer and if correctly can identify outliers
'''

import numpy as np
from icet.fitting import EnsembleOptimizer


def find_outliers(data, threshold=7.0):
    ''' Outlier detection for 1D data '''
    median = np.median(data)
    diff = np.sqrt((data - median)**2)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    return np.where(modified_z_score > threshold)[0]


np.random.seed(42)

# setup Ax=y problem with sparse solution
N, M, N_outlier = 400, 100, 10
A = np.random.normal(0, 1.0, (N, M))
x_true = np.random.normal(0.0, 10.0, M)

# generate outlier rows
outliers = np.sort(np.random.choice(np.arange(N), N_outlier, replace=False))
noise = np.random.normal(0.0, 1, N)
noise[outliers] += (np.random.choice([-1, 1], N_outlier) *
                    np.random.normal(20, 1, N_outlier))

y = np.dot(A, x_true) + noise

# run ensemble optimizer
eopt = EnsembleOptimizer((A, y), ensemble_size=100)
eopt.train()
assert np.linalg.norm(eopt.parameters - x_true) < 5.0

# find outliters
E = eopt.get_errors()
E_ave = np.mean(np.abs(E), axis=1)
predicted_outliers = find_outliers(E_ave)

assert np.all(predicted_outliers == outliers)
