import numpy as np
from icet.fitting import CrossValidationEstimator
from sklearn.model_selection import KFold

np.random.seed(42)

# setup Ax=y problem
N, M = 200, 50
A = np.random.normal(0, 1.0, (N, M))
x_true = np.random.normal(0.0, 10.0, M)
y = np.dot(A, x_true) + np.random.normal(0.0, 0.2, N)

# test CrossValidationEstimator
cve = CrossValidationEstimator((A, y),
                               validation_method='k-fold',
                               number_of_splits=10)
cve.train()

# Manual k-fold
kf = KFold(n_splits=10)
cv_folds = []
target, predicted = [], []
for train_rows, test_rows in kf.split(A):
    A_train, y_train = A[train_rows, :], y[train_rows]
    A_test, y_test = A[test_rows, :], y[test_rows]
    parameters = np.linalg.lstsq(A_train, y_train)[0]
    cv_folds.append(np.sqrt(np.mean((np.dot(A_test, parameters) - y_test)**2)))
    target.extend(y_test)
    predicted.extend(np.dot(A_test, parameters))

cv_folds = np.array(cv_folds)
final_parameters = np.linalg.lstsq(A, y)[0]

# assert results is the same
assert np.allclose(cve.rmse_validation_splits, cv_folds)
assert np.allclose(cve.parameters, final_parameters)

assert np.allclose(cve.validation_scatter_data.target, target)
assert np.allclose(cve.validation_scatter_data.predicted, predicted)
