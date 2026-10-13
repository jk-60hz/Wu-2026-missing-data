"""Generate (deterministic) missing-data test sets and summarise them.

    python scripts/make_testsets.py --datasets maui_15min,maui_1h,stpete_1h --seeds 5

Masks -> data/testsets/<dataset>/<scenario>__<param>__s<seed>.npz
Summary -> results/testsets_summary.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets import load  # noqa: E402
from src.missingness import SCENARIOS, make_mask  # noqa: E402

TS = ROOT / "data" / "testsets"


def seed_of(dataset, scenario, param, seed):
    # stable, human-independent seed per test set
    return 10_000 * seed + sum(map(ord, dataset + scenario + param))


def get_testset(dataset, scenario, param, kw, seed, df=None, ctx=None, ref=None):
    """ref: override the designated reference column (S8 reference sweep)."""
    tag = "" if ref is None else f"__ref{ref}"
    path = TS / dataset / f"{scenario}__{param}__s{seed}{tag}.npz"
    if path.exists():
        z = np.load(path)
        return z["mask"], z["targets"].tolist()
    if df is None:
        df, _, ctx = load(dataset)
    X = df.to_numpy()
    day = ctx["zenith"] < 85
    M, targets = make_mask(scenario, kw, X, ctx["ref"] if ref is None else ref, day, ctx["day_idx"], ctx["sph"],
                           seed_of(dataset, scenario, param, seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mask=M, targets=np.array(targets))
    return M, targets


def runs_of(gaps):
    d = np.diff(np.r_[0, gaps.astype(int), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="maui_15min,maui_1h,stpete_1h")
    p.add_argument("--seeds", type=int, default=5)
    a = p.parse_args()
    rows = []
    for ds in a.datasets.split(","):
        df, meta, ctx = load(ds)
        day = ctx["zenith"] < 85
        for sc, params in SCENARIOS.items():
            for param, kw in params:
                for s in range(a.seeds):
                    M, targets = get_testset(ds, sc, param, kw, s, df, ctx)
                    tm = M[:, targets]
                    runs = np.concatenate([runs_of(tm[:, k]) for k in range(tm.shape[1])])
                    rows.append(dict(dataset=ds, scenario=sc, param=param, seed=s,
                                     n_targets=len(targets),
                                     miss_frac=tm.mean(), miss_frac_day=tm[day].mean(),
                                     gap_mean_h=runs.mean() / ctx["sph"] if len(runs) else 0,
                                     gap_max_h=runs.max() / ctx["sph"] if len(runs) else 0))
        print(ds, "done")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    s = pd.DataFrame(rows)
    s.to_csv(out / "testsets_summary.csv", index=False)
    print(s.groupby(["dataset", "scenario", "param"])[["n_targets", "miss_frac", "miss_frac_day", "gap_mean_h", "gap_max_h"]]
          .mean().round(3).to_string())
