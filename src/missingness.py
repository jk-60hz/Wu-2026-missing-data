"""Missing-data scenarios S1-S7 (S8 = reference sweep, handled by the benchmark).

All generators return a boolean mask M of shape (T, n): True = value removed.
`targets` are the columns allowed to lose data; the rest stay complete.
Each mask is fully determined by (scenario, params, seed).
"""
import numpy as np

STEPS_PER_HOUR = {"15min": 4, "1h": 1}


def _blocks(T, L, frac, rng, avoid=None):
    """Random non-overlapping blocks of length L covering ~frac of T."""
    m = np.zeros(T, bool)
    target = int(round(frac * T))
    tries = 0
    while m.sum() < target and tries < 10000:
        s = rng.integers(0, T - L)
        if not m[max(0, s - 1): s + L + 1].any():
            m[s: s + L] = True
        tries += 1
    return m


def s1_mcar(T, n, targets, rng, r):
    M = np.zeros((T, n), bool)
    for j in targets:
        M[rng.choice(T, int(round(r * T)), replace=False), j] = True
    return M


def s2_blocks(T, n, targets, rng, hours, frac, sph):
    M = np.zeros((T, n), bool)
    for j in targets:
        M[:, j] = _blocks(T, max(1, int(hours * sph)), frac, rng)
    return M


def s3_days(T, n, targets, rng, frac, local_day):
    """Whole local calendar days missing (24 h, including the full daylight period)."""
    M = np.zeros((T, n), bool)
    days = np.unique(local_day)
    for j in targets:
        drop = rng.choice(days, max(1, int(round(frac * len(days)))), replace=False)
        M[np.isin(local_day, drop), j] = True
    return M


def s4_periodic(T, n, targets, rng, period):
    """Logger keeps only every `period`-th sample (random phase per series)."""
    M = np.zeros((T, n), bool)
    t = np.arange(T)
    for j in targets:
        M[:, j] = (t + rng.integers(0, period)) % period != 0
    return M


def s5_threshold(X, targets, day, q, side):
    """MNAR: values above the (1-q) daytime quantile (sensor saturation/clipping, side='upper')
    or below the q daytime quantile (low-light cut-off, side='lower') are lost."""
    M = np.zeros(X.shape, bool)
    for j in targets:
        v = X[day, j]
        if side == "upper":
            M[:, j] = day & (X[:, j] >= np.quantile(v, 1 - q))
        else:
            M[:, j] = day & (X[:, j] <= np.quantile(v, q))
    return M


def s7_simultaneous(T, n, rng, hours, frac, sph):
    """Region-wide outage: the same blocks are missing in ALL series (incl. the reference)."""
    m = _blocks(T, max(1, int(hours * sph)), frac, rng)
    return np.repeat(m[:, None], n, axis=1)


# --------------------------------------------------------------------------- registry
SCENARIOS = {
    # id: list of (param label, kwargs)
    "S1_mcar": [(f"r={r}", dict(r=r)) for r in (0.1, 0.3, 0.5, 0.7, 0.9)],
    "S2_block": [(f"L={h}h", dict(hours=h, frac=0.2)) for h in (1, 6, 24, 72, 168)],
    "S3_day": [(f"days={f}", dict(frac=f)) for f in (0.05, 0.1, 0.2)],
    "S4_periodic": [(f"P={p}", dict(period=p)) for p in (2, 4, 8)],
    "S5_threshold": [(f"{s}{q}", dict(q=q, side=s)) for s in ("upper", "lower") for q in (0.05, 0.2)],
    "S6_oneref": [(f"r={r}", dict(r=r)) for r in (0.5, 0.9)],
    "S7_simul": [(f"L={h}h", dict(hours=h, frac=0.1)) for h in (6, 24)],
}


def make_mask(scenario, kw, X, ref, day, local_day, sph, seed):
    """Build the mask for one test set.

    S1-S5: half of the non-reference columns (random, >= 1) are incomplete,
           the others (and the reference) stay complete.
    S6   : only the reference column is complete, every other column is MCAR(r).
    S7   : identical blocks missing in every column.
    """
    T, n = X.shape
    rng = np.random.default_rng(seed)
    others = [c for c in range(n) if c != ref]
    if scenario == "S6_oneref":
        return s1_mcar(T, n, others, rng, **kw), others
    if scenario == "S7_simul":
        return s7_simultaneous(T, n, rng, sph=sph, **kw), list(range(n))
    k = max(1, len(others) // 2)
    targets = sorted(rng.choice(others, k, replace=False).tolist())
    if scenario == "S1_mcar":
        M = s1_mcar(T, n, targets, rng, **kw)
    elif scenario == "S2_block":
        M = s2_blocks(T, n, targets, rng, sph=sph, **kw)
    elif scenario == "S3_day":
        M = s3_days(T, n, targets, rng, local_day=local_day, **kw)
    elif scenario == "S4_periodic":
        M = s4_periodic(T, n, targets, rng, **kw)
    elif scenario == "S5_threshold":
        M = s5_threshold(X, targets, day, **kw)
    else:
        raise ValueError(scenario)
    return M, targets
