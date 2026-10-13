"""Full-Partial Reconstruction Mapping (FPRM), Wu et al., Nat. Commun. 2026.

For every incomplete series x_j and a fully observed reference x_i:
  1. full reconstruction   X_i(t) = (x_i(t), x_i(t+tau), ..., x_i(t+(E-1)tau))   (Eq. 6)
  2. partial reconstruction X_j(t), first column = x_j(t)                     (Eq. 9)
  3. learn F with x_j(t) = F(X_i(t)) on time points where x_j(t) is observed   (Eq. 13)
  4. impute x_j(k) = F(X_i(k)) for the missing k                               (Eq. 14)
F is a Gaussian process by default (Eq. 15-18); SVR / MLP can be swapped in
(the paper's SVM* / NN* variants).

Extensions that are NOT in the paper are flagged as such:
  * ``refs='multi'``: concatenate the delay vectors of several complete series
  * ``direction='backward'``: causal embedding (x_i(t), x_i(t-tau), ...)
  * rows whose delay vector cannot be formed (reference itself missing) fall
    back to linear interpolation of the target (or are left NaN for the caller
    with ``fallback=None``), and are counted in ``self.info``.
"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from .embedding import delay_embed, edge_fill
from .gpr import GPRMap
from .selection import tau_ami, e_fnn


def make_learner(name, seed=0, max_train=2000):
    if name == "gpr":
        return GPRMap(max_train=max_train, seed=seed)
    if name == "svr":
        return _Sub(SVR(C=1.0, epsilon=0.05, gamma="scale"), max_train, seed)
    if name == "mlp":
        return _Sub(MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=800, early_stopping=True, random_state=seed), 20000, seed)
    if name == "knn":
        return _Sub(KNeighborsRegressor(10, weights="distance"), 100000, seed)
    if name == "rf":
        return _Sub(RandomForestRegressor(100, min_samples_leaf=3, n_jobs=1, random_state=seed), 20000, seed)
    raise ValueError(name)


class _Sub:
    """Wrap an sklearn regressor and train on a random subset of at most n rows."""

    def __init__(self, est, n, seed):
        self.est, self.n, self.seed = est, n, seed

    def fit(self, X, y):
        if len(y) > self.n:
            idx = np.random.default_rng(self.seed).choice(len(y), self.n, replace=False)
            X, y = X[idx], y[idx]
        self.est.fit(X, y)
        return self

    def predict(self, X):
        return self.est.predict(X)


def interp_fill(y):
    y = np.asarray(y, float).copy()
    obs = ~np.isnan(y)
    if obs.sum() == 0:
        return np.zeros_like(y)
    t = np.arange(len(y))
    y[~obs] = np.interp(t[~obs], t[obs], y[obs])
    return y


class FPRM:
    def __init__(self, E=None, tau=None, learner="gpr", refs="first", direction="forward",
                 select="cv", E_grid=(1, 2, 3, 4, 5, 6, 8), tau_grid=(1, 2, 3, 4, 6),
                 max_train=2000, fallback="linear", seed=0):
        self.E, self.tau, self.learner, self.refs = E, tau, learner, refs
        self.direction, self.select = direction, select
        self.E_grid, self.tau_grid = E_grid, tau_grid
        self.max_train, self.seed = max_train, seed
        self.fallback = fallback  # "linear" or None (leave NaN for the caller)
        self.info = {}

    # ---------------------------------------------------------------- helpers
    def _features(self, Z, ref_cols, E, tau):
        feats = []
        for i in ref_cols:
            x = Z[:, i]
            if not np.isnan(x).any():
                x = edge_fill(x)  # only pads beyond the series ends
            else:
                x = x.copy()
            emb = delay_embed(x, E, tau, self.direction)
            # pad the last (E-1)tau rows (outside the series) with the edge value
            if not np.isnan(Z[:, i]).any():
                emb = np.where(np.isnan(emb), x[-1] if self.direction == "forward" else x[0], emb)
            feats.append(emb)
        return np.hstack(feats)

    def _choose_refs(self, Z, j, complete):
        if self.refs == "first":
            return [complete[0]]
        if self.refs == "multi":
            return list(complete)
        if self.refs == "best":  # highest |corr| on observed rows of the target
            obs = ~np.isnan(Z[:, j])
            c = [abs(np.corrcoef(Z[obs, i], Z[obs, j])[0, 1]) for i in complete]
            return [complete[int(np.nanargmax(c))]]
        if isinstance(self.refs, (list, tuple)):
            return list(self.refs)
        raise ValueError(self.refs)

    def _select(self, Z, j, ref_cols):
        if self.E is not None and self.tau is not None:
            return self.E, self.tau
        if self.select == "ami_fnn":
            x = Z[:, ref_cols[0]]
            x = x[~np.isnan(x)]
            tau = self.tau or tau_ami(x)
            return (self.E or e_fnn(x, tau)), tau
        # cross-validation on observed target values only
        rng = np.random.default_rng(self.seed + 7)
        y = Z[:, j]
        obs = np.flatnonzero(~np.isnan(y))
        obs = rng.choice(obs, min(len(obs), 600), replace=False)
        va, tr = obs[: len(obs) // 3], obs[len(obs) // 3:]
        best, best_s = (1, 1), np.inf
        for E in ([self.E] if self.E else self.E_grid):
            for tau in ([self.tau] if self.tau else (self.tau_grid if E > 1 else (1,))):
                F = self._features(Z, ref_cols, E, tau)
                ok_tr = tr[np.isfinite(F[tr]).all(1)]
                ok_va = va[np.isfinite(F[va]).all(1)]
                if len(ok_tr) < 20 or len(ok_va) < 10:
                    continue
                m = make_learner(self.learner, self.seed, 400).fit(F[ok_tr], y[ok_tr])
                s = np.sqrt(np.mean((m.predict(F[ok_va]) - y[ok_va]) ** 2))
                if s < best_s - 1e-9:
                    best, best_s = (E, tau), s
        return best

    # ------------------------------------------------------------------- main
    def fit_impute(self, X, complete=None):
        """X: (T, n) array with NaN for missing entries. Returns a filled copy."""
        X = np.asarray(X, float)
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        sd[~(sd > 0)] = 1.0
        Z = (X - mu) / sd  # z-score with observed statistics only
        if complete is None:
            complete = [c for c in range(X.shape[1]) if not np.isnan(X[:, c]).any()]
        if not complete:  # no complete series: use the least-missing one as reference
            complete = [int(np.argmin(np.isnan(X).sum(0)))]
        out = Z.copy()
        for j in range(X.shape[1]):
            miss = np.isnan(Z[:, j])
            if not miss.any():
                continue
            cand = [c for c in complete if c != j]
            if not cand:
                out[:, j] = interp_fill(Z[:, j]) if self.fallback else Z[:, j]
                continue
            ref_cols = self._choose_refs(Z, j, cand)
            E, tau = self._select(Z, j, ref_cols)
            F = self._features(Z, ref_cols, E, tau)
            ok = np.isfinite(F).all(1)
            tr = ~miss & ok
            te = miss & ok
            col = Z[:, j].copy()
            if tr.sum() >= 5 and te.any():
                m = make_learner(self.learner, self.seed, self.max_train).fit(F[tr], Z[tr, j])
                col[te] = m.predict(F[te])
            fb = np.isnan(col)
            if fb.any() and self.fallback:
                col[fb] = interp_fill(col)[fb]
            out[:, j] = col
            self.info[j] = dict(E=E, tau=tau, refs=ref_cols, n_train=int(tr.sum()),
                                n_pred=int(te.sum()), n_fallback=int(fb.sum()))
        return out * sd + mu
