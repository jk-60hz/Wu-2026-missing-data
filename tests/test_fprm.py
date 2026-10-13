import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fprm import FPRM, delay_embed  # noqa: E402
from src.metrics import paper_metrics  # noqa: E402
from src.missingness import SCENARIOS, make_mask  # noqa: E402
from src.synthetic import ecology  # noqa: E402


def test_delay_embed_forward_backward():
    x = np.arange(10.0)
    F = delay_embed(x, 3, 2)
    assert F[0].tolist() == [0, 2, 4]
    assert np.isnan(F[-1, 1:]).all()
    B = delay_embed(x, 3, 2, "backward")
    assert B[9].tolist() == [9, 7, 5]
    assert np.isnan(B[0, 1:]).all()


def test_missing_values_propagate_into_partial_reconstruction():
    x = np.arange(10.0)
    x[4] = np.nan
    F = delay_embed(x, 2, 1)
    assert np.isnan(F[3, 1]) and np.isnan(F[4, 0])


def test_observed_values_untouched_and_no_nan():
    S = ecology()
    X = S.copy()
    rng = np.random.default_rng(0)
    m = rng.random(len(S)) < 0.5
    X[m, 1] = np.nan
    Y = FPRM(E=3, tau=1).fit_impute(X, complete=[0])
    assert not np.isnan(Y).any()
    np.testing.assert_allclose(Y[~m, 1], S[~m, 1])
    np.testing.assert_allclose(Y[:, 0], S[:, 0])


def test_no_leakage_of_hidden_values():
    """Changing the ground truth at missing positions must not change the output."""
    S = ecology()
    m = np.random.default_rng(1).random(len(S)) < 0.6
    out = []
    for fill in (0.0, 123.0):
        X = S.copy()
        X[m, 2] = fill   # hidden truth replaced ...
        X[m, 2] = np.nan  # ... then removed as in the benchmark
        out.append(FPRM(seed=0).fit_impute(X, complete=[0]))
    np.testing.assert_allclose(out[0], out[1])


def test_reproduces_paper_ecology_r05():
    S = ecology()
    X = S.copy()
    rng = np.random.default_rng(2)
    masks = {}
    for j in (1, 2):
        masks[j] = rng.random(len(S)) < 0.5
        X[masks[j], j] = np.nan
    Y = FPRM(seed=0).fit_impute(X, complete=[0])
    Y1 = FPRM(E=1, tau=1, seed=0).fit_impute(X, complete=[0])
    for j in (1, 2):
        m = paper_metrics(S[masks[j], j], Y[masks[j], j])
        m1 = paper_metrics(S[masks[j], j], Y1[masks[j], j])
        assert m["rho"] > 0.95 and m["nrmse"] < 0.3          # paper Fig. 3d,e: rho > 0.8, RMSE < 0.4
        assert m1["nrmse"] > m["nrmse"] + 0.3                 # delay embedding is what makes it work


@pytest.mark.parametrize("sc", list(SCENARIOS))
def test_masks_deterministic_and_well_formed(sc):
    T, n = 24 * 40, 4
    X = np.abs(np.random.default_rng(0).normal(size=(T, n)))
    day = (np.arange(T) % 24 >= 7) & (np.arange(T) % 24 <= 17)
    local_day = np.arange(T) // 24
    for param, kw in SCENARIOS[sc]:
        M1, t1 = make_mask(sc, kw, X, 0, day, local_day, 1, 5)
        M2, t2 = make_mask(sc, kw, X, 0, day, local_day, 1, 5)
        assert (M1 == M2).all() and t1 == t2
        untouched = [c for c in range(n) if c not in t1]
        assert not M1[:, untouched].any()
        if sc != "S7_simul":
            assert not M1[:, 0].any()  # the reference stays complete
        else:
            assert (M1 == M1[:, :1]).all()
