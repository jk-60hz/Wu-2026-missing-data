"""Reproduce the paper's low-dimensional experiments (Fig. 3d,e / 3j,k).

x is complete, y and z each lose a fraction r of their values at random.
Writes results/synthetic/reproduction.csv and prints the summary table.

    python scripts/reproduce_synthetic.py --reps 20 --workers 4
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import concurrent.futures as cf  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fprm import FPRM  # noqa: E402
from src.metrics import paper_metrics  # noqa: E402
from src.synthetic import ecology, lorenz  # noqa: E402

SYSTEMS = {"ecology": ecology, "lorenz": lorenz,
           # same number of points but 3x coarser sampling (dt = 0.06 instead of 0.02)
           "lorenz_coarse": lambda: lorenz(n=1600, t_span=(10, 106))}
METHODS = {"FPRM-GPR": dict(learner="gpr"), "GPR(E=1)": dict(learner="gpr", E=1, tau=1),
           "FPRM-SVR": dict(learner="svr"), "FPRM-MLP": dict(learner="mlp")}


def one(job):
    name, r, rep, select = job
    S = SYSTEMS[name]()
    rng = np.random.default_rng(1000 * rep + int(r * 100))
    X = S.copy()
    masks = {}
    for j in (1, 2):
        m = np.zeros(len(S), bool)
        m[rng.choice(len(S), int(round(r * len(S))), replace=False)] = True
        X[m, j] = np.nan
        masks[j] = m
    rows = []
    for method, kw in METHODS.items():
        model = FPRM(select=select, seed=rep, tau_grid=(1, 2, 3, 4, 6, 8), **kw)
        Y = model.fit_impute(X, complete=[0])
        for j, var in ((1, "y"), (2, "z")):
            mt = paper_metrics(S[masks[j], j], Y[masks[j], j])
            rows.append(dict(system=name, r=r, rep=rep, method=method, var=var,
                             E=model.info[j]["E"], tau=model.info[j]["tau"], **mt))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--rates", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--systems", default="ecology,lorenz,lorenz_coarse")
    p.add_argument("--select", default="cv")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    jobs = [(s, float(r), k, a.select) for s in a.systems.split(",") for r in a.rates.split(",")
            for k in range(a.reps)]
    rows = []
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(one, jobs)):
            rows += res
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "reproduction.csv", index=False)
    print(df.groupby(["system", "method", "var", "r"])[["rho", "nrmse", "E", "tau"]].mean().round(3).to_string())
