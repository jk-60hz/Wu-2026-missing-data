"""Baseline imputers. Every function maps (X with NaN, ctx) -> filled X.

ctx keys: cs (clear-sky GHI), zenith, tod (time-of-day slot index), day_idx
(local day number), complete (list of fully observed columns), ref (designated
reference column).
"""
import warnings

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LinearRegression

from ..fprm.model import interp_fill, make_learner
from ..solar import ClearSkyIndex


def _cols_missing(X):
    return [j for j in range(X.shape[1]) if np.isnan(X[:, j]).any()]


def mean_impute(X, ctx):
    Y = X.copy()
    for j in _cols_missing(X):
        Y[np.isnan(X[:, j]), j] = np.nanmean(X[:, j])
    return Y


def linear_interp(X, ctx):
    Y = X.copy()
    for j in _cols_missing(X):
        Y[:, j] = interp_fill(X[:, j])
    return Y


def diurnal_profile(X, ctx, window_days=7):
    """Mean of the observed values in the same time-of-day slot within +-window_days."""
    Y = X.copy()
    tod, day = ctx["tod"], ctx["day_idx"]
    for j in _cols_missing(X):
        miss = np.flatnonzero(np.isnan(X[:, j]))
        for t in miss:
            sel = (tod == tod[t]) & (np.abs(day - day[t]) <= window_days) & ~np.isnan(X[:, j])
            Y[t, j] = X[sel, j].mean() if sel.any() else np.nanmean(X[tod == tod[t], j])
    return np.nan_to_num(Y)


def csi_interp(X, ctx):
    """Linear interpolation of the clear-sky index over daytime points, back-transformed."""
    t = ClearSkyIndex(ctx["cs"], ctx["zenith"]).fit(X)
    K = t.transform(X)
    ok = t.ok
    for j in _cols_missing(X):
        k = K[ok, j]
        K[ok, j] = interp_fill(k)
    F = t.inverse(np.nan_to_num(K))
    Y = X.copy()
    Y[np.isnan(X)] = F[np.isnan(X)]  # observed values are left untouched
    return Y


def _best_ref(X, j, complete):
    obs = ~np.isnan(X[:, j])
    c = [abs(np.corrcoef(X[obs, i], X[obs, j])[0, 1]) for i in complete]
    return complete[int(np.nanargmax(c))]


def neighbor_ratio(X, ctx, window_days=3):
    """target = ratio * best-correlated complete series, ratio from observed daytime
    energy within +-window_days (the classic 'neighbouring plant scaling')."""
    Y = X.copy()
    day, ok = ctx["day_idx"], ctx["zenith"] < 85
    for j in _cols_missing(X):
        cand = [c for c in ctx["complete"] if c != j]
        if not cand:
            Y[:, j] = csi_interp(X, ctx)[:, j]
            continue
        i = _best_ref(X, j, cand)
        obs = ~np.isnan(X[:, j]) & ok
        glob = X[obs, j].sum() / max(X[obs, i].sum(), 1e-9)
        for t in np.flatnonzero(np.isnan(X[:, j])):
            sel = obs & (np.abs(day - day[t]) <= window_days)
            den = X[sel, i].sum()
            r = X[sel, j].sum() / den if den > 0 else glob
            Y[t, j] = r * X[t, i]
    return Y


def _fallback(Y, X, ctx):
    """Rows the model could not fill (no complete predictor): clear-sky-index interpolation."""
    if np.isnan(Y).any():
        F = csi_interp(X, ctx)
        Y[np.isnan(Y)] = F[np.isnan(Y)]
    return Y


def direct_regression(X, ctx, learner="gpr", seed=0):
    """Regress each incomplete column on the *current* values of all complete
    columns (no delay embedding) -- the paper's 'direct fitting' baselines."""
    Y = X.copy()
    for j in _cols_missing(X):
        cand = [c for c in ctx["complete"] if c != j]
        if not cand:
            continue
        miss = np.isnan(X[:, j])
        F = X[:, cand]
        mu, sd = F.mean(0), F.std(0) + 1e-9
        F = (F - mu) / sd
        ym, ys = np.nanmean(X[:, j]), np.nanstd(X[:, j]) + 1e-9
        if learner == "ols":
            m = LinearRegression().fit(F[~miss], X[~miss, j])
            Y[miss, j] = m.predict(F[miss])
            continue
        m = make_learner(learner, seed).fit(F[~miss], (X[~miss, j] - ym) / ys)
        Y[miss, j] = m.predict(F[miss]) * ys + ym
    return _fallback(Y, X, ctx)


def mice(X, ctx, seed=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp = IterativeImputer(BayesianRidge(), max_iter=10, random_state=seed)
        return imp.fit_transform(X)


def knn_imputer(X, ctx):
    return KNNImputer(n_neighbors=10, weights="distance").fit_transform(X)
