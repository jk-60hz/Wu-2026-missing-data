"""Embedding-parameter (E, tau) selection.

The paper's Supplementary Fig. S1 (parameter selection) is not available to us,
so two observed-data-only selectors are provided:
  * ``cv``      : grid search, scoring a held-out part of the *observed* target
                  values (default; never touches ground truth of missing entries)
  * ``ami_fnn`` : tau = first local minimum of auto mutual information,
                  E = false-nearest-neighbour criterion (Kennel et al. 1992)
"""
import numpy as np
from scipy.spatial import cKDTree


def ami(x, max_lag=30, bins=16):
    x = np.asarray(x, float)
    out = []
    for L in range(1, max_lag + 1):
        a, b = x[:-L], x[L:]
        h, _, _ = np.histogram2d(a, b, bins=bins)
        p = h / h.sum()
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        out.append(float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum()))
    return np.array(out)


def tau_ami(x, max_lag=30):
    m = ami(x, max_lag)
    for i in range(1, len(m) - 1):
        if m[i] < m[i - 1] and m[i] <= m[i + 1]:
            return i + 1
    # no local minimum: first lag where AMI drops below 1/e of AMI(1)
    below = np.flatnonzero(m < m[0] / np.e)
    return int(below[0] + 1) if len(below) else 1


def e_fnn(x, tau, max_E=10, rtol=15.0, frac=0.05):
    x = np.asarray(x, float)
    for E in range(1, max_E + 1):
        n = len(x) - E * tau
        if n < 20:
            return E
        Y = np.column_stack([x[k * tau: k * tau + n] for k in range(E)])
        nxt = x[E * tau: E * tau + n]
        d, i = cKDTree(Y).query(Y, k=2)
        d, i = d[:, 1], i[:, 1]
        ok = d > 0
        ratio = np.abs(nxt[ok] - nxt[i[ok]]) / d[ok]
        if np.mean(ratio > rtol) < frac:
            return E
    return max_E
