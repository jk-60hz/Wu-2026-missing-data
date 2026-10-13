"""Delay-coordinate embedding helpers (paper Eq. 6 and 9)."""
import numpy as np


def lag_offsets(E, tau, direction="forward"):
    """Time offsets of the E embedding coordinates.

    forward  : (0, tau, ..., (E-1)tau)        -- the paper's X(t) = (x(t), x(t+tau), ...)
    backward : (0, -tau, ..., -(E-1)tau)      -- causal variant (usable online)
    centered : symmetric around 0
    """
    k = np.arange(E) * tau
    if direction == "forward":
        return k
    if direction == "backward":
        return -k
    if direction == "centered":
        return k - (E - 1) * tau // 2
    raise ValueError(direction)


def delay_embed(x, E, tau, direction="forward"):
    """Return a (T, E) matrix whose row t is the delay vector at time t.

    Coordinates that fall outside [0, T) are NaN (the paper simply drops the
    last (E-1)tau rows, m = l - (E-1)tau; we keep T rows so every time index has
    a row and let the caller decide how to handle incomplete rows).
    Missing values in x (NaN) propagate to every row that uses them -- this is
    exactly the "partial reconstruction" of Eq. 9.
    """
    x = np.asarray(x, float)
    T = len(x)
    out = np.full((T, E), np.nan)
    for c, off in enumerate(lag_offsets(E, tau, direction)):
        src = np.arange(T) + off
        ok = (src >= 0) & (src < T)
        out[ok, c] = x[src[ok]]
    return out


def edge_fill(x):
    """Fill leading/trailing NaN with the nearest valid value (used only to give
    the last (E-1)tau rows of a *complete* series a usable delay vector)."""
    x = np.asarray(x, float).copy()
    idx = np.flatnonzero(~np.isnan(x))
    if len(idx) == 0:
        return x
    x[: idx[0]] = x[idx[0]]
    x[idx[-1] + 1:] = x[idx[-1]]
    return x
