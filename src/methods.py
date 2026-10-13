"""Registry of imputation methods evaluated in the benchmark."""
import numpy as np

from . import baselines as B
from .fprm import FPRM
from .solar import ClearSkyIndex

# shared (E, tau) grid; kept modest to bound the cost of observed-data CV
E_GRID = (1, 2, 3, 4, 6, 8)
TAU_GRID = (1, 2, 4)


def _in_space(fn, X, ctx, space):
    if space == "raw":
        return fn(X)
    t = ClearSkyIndex(ctx["cs"], ctx["zenith"]).fit(X)
    return t.inverse(fn(t.transform(X)))


def fprm(X, ctx, learner="gpr", refs="best", space="raw", direction="forward", E=None, tau=None, seed=0):
    if refs == "designated":
        refs = [ctx["ref"]] if ctx["ref"] in ctx["complete"] else "best"
    model = FPRM(E=E, tau=tau, learner=learner, refs=refs, direction=direction,
                 E_grid=E_GRID, tau_grid=TAU_GRID, fallback=None, seed=seed)
    Y = _in_space(lambda Z: model.fit_impute(Z, complete=ctx["complete"] or None), X, ctx, space)
    if np.isnan(Y).any():
        # rows with no usable reference (e.g. S7 simultaneous outage): solar fallback
        Y = B.csi_interp(Y, ctx)
    ctx["_info"] = model.info  # picked up by the benchmark runner (E, tau, refs)
    return Y, model.info


def _wrap_fprm(**kw):
    def f(X, ctx, seed=0):
        Y, info = fprm(X, ctx, seed=seed, **kw)
        return Y
    f.fprm = True
    return f


METHODS = {
    # simple / practitioner baselines
    "Mean": lambda X, c, seed=0: B.mean_impute(X, c),
    "Linear": lambda X, c, seed=0: B.linear_interp(X, c),
    "Diurnal": lambda X, c, seed=0: B.diurnal_profile(X, c),
    "CSI-Linear": lambda X, c, seed=0: B.csi_interp(X, c),
    "NbrRatio": lambda X, c, seed=0: B.neighbor_ratio(X, c),
    "NbrOLS": lambda X, c, seed=0: B.direct_regression(X, c, "ols"),
    # direct ML regressions on all complete series (no embedding)
    "kNN": lambda X, c, seed=0: B.direct_regression(X, c, "knn", seed),
    "GPR": lambda X, c, seed=0: B.direct_regression(X, c, "gpr", seed),
    "SVR": lambda X, c, seed=0: B.direct_regression(X, c, "svr", seed),
    "RF": lambda X, c, seed=0: B.direct_regression(X, c, "rf", seed),
    "MLP": lambda X, c, seed=0: B.direct_regression(X, c, "mlp", seed),
    "MICE": lambda X, c, seed=0: B.mice(X, c, seed),
    "KNNImputer": lambda X, c, seed=0: B.knn_imputer(X, c),
    # FPRM and ablations
    "FPRM": _wrap_fprm(),                                    # paper: GPR, single (best) reference
    "FPRM-ref": _wrap_fprm(refs="designated"),               # single designated reference (GHI / site)
    "FPRM-E1": _wrap_fprm(E=1, tau=1),                       # same, no embedding
    "FPRM-SVR": _wrap_fprm(learner="svr"),                   # paper's SVM*
    "FPRM-MLP": _wrap_fprm(learner="mlp"),                   # paper's NN*
    "FPRM-multi": _wrap_fprm(refs="multi"),                  # extension: all complete refs
    "FPRM-back": _wrap_fprm(direction="backward"),           # extension: causal embedding
    "FPRM-csi": _wrap_fprm(space="csi"),                     # clear-sky-index space
    "FPRM-E1-csi": _wrap_fprm(E=1, tau=1, space="csi"),
}


def postprocess(Y, ctx):
    """Physical post-processing applied identically to every method."""
    Y = np.clip(np.nan_to_num(Y, nan=0.0), 0, None)
    Y[ctx["zenith"] >= 90] = 0.0
    return Y
