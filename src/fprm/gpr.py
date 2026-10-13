"""Gaussian process regression used as the FPRM mapping F (paper Eq. 15-18).

The paper's code is MATLAB (fitrgp). fitrgp defaults are: squared-exponential
kernel, constant basis (mean) function, kernel scale initialised to the mean of
the predictor standard deviations (= the paper's theta), noise sigma initialised
to std(y)/sqrt(2), hyper-parameters fitted by maximising the marginal
likelihood, and a subset-of-data approximation when n > 2000.
We mirror that with scikit-learn:
  * hyper-parameters are fitted on a random subset (n_opt unique points) to keep
    cost low, with one optimizer restart and a noise floor of 1e-4 (relative),
  * the final exact GP uses at most max_train training points (subset of data).
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


class GPRMap:
    def __init__(self, max_train=2000, n_opt=600, optimize=True, seed=0):
        self.max_train, self.n_opt, self.optimize, self.seed = max_train, n_opt, optimize, seed

    def _kernel(self, X, y):
        theta = float(np.mean(np.std(X, axis=0))) or 1.0
        s2 = float(np.var(y)) / 2 or 1e-2
        return (ConstantKernel(1.0, (1e-3, 1e5)) * RBF(theta, (1e-2, 1e3))
                + WhiteKernel(s2 / max(np.var(y), 1e-12), (1e-4, 1e1)))

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        kernel = self._kernel(X, y)
        if self.optimize:
            # identical rows (e.g. night-time zeros in PV data) make the marginal likelihood
            # favour a zero-noise, tiny-length-scale interpolant; optimise on unique rows
            _, uniq = np.unique(np.round(np.column_stack([X, y]), 6), axis=0, return_index=True)
            sub = rng.choice(uniq, min(len(uniq), self.n_opt), replace=False)
            gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1,
                                          random_state=self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(X[sub], y[sub])
            kernel = gp.kernel_
        idx = rng.choice(n, min(n, self.max_train), replace=False)
        self.gp = GaussianProcessRegressor(kernel, optimizer=None, normalize_y=True)
        self.gp.fit(X[idx], y[idx])
        return self

    def predict(self, X):
        return self.gp.predict(X)
