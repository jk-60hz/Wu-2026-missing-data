"""Run all methods on all test sets of the given datasets.

    python scripts/run_benchmark.py --config configs/main.yaml

Writes results/<run>/metrics.csv (tidy: one row per testset x method x variable)
and results/<run>/examples/*.npz (predictions for seed 0, for figures).
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import concurrent.futures as cf  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_testsets import get_testset  # noqa: E402
from src.datasets import load  # noqa: E402
from src.methods import METHODS, postprocess  # noqa: E402
from src.metrics import solar_metrics  # noqa: E402
from src.missingness import SCENARIOS  # noqa: E402

_CACHE = {}


def _ds(name):
    if name not in _CACHE:
        _CACHE[name] = load(name)
    return _CACHE[name]


def run_one(job):
    ds, sc, param, kw, seed, methods, out_dir, ref_override = job
    df, meta, ctx0 = _ds(ds)
    truth = df.to_numpy()
    M, targets = get_testset(ds, sc, param, kw, seed, df, ctx0, ref_override)
    X = truth.copy()
    X[M] = np.nan
    ctx = dict(ctx0)
    if ref_override is not None:
        ctx["ref"] = ref_override
    ctx["complete"] = [c for c in range(X.shape[1]) if not M[:, c].any()]
    day = ctx["zenith"] < 85
    rows, preds = [], {}
    for name in methods:
        t0 = time.time()
        try:
            ctx.pop("_info", None)
            Y = postprocess(METHODS[name](X.copy(), ctx, seed=seed), ctx)
        except Exception:
            traceback.print_exc()
            continue
        dt = time.time() - t0
        info = ctx.pop("_info", {}) or {}
        preds[name] = Y[:, targets].astype(np.float32)
        for j in targets:
            ev = M[:, j] & day
            if ev.sum() < 3:
                continue
            m = solar_metrics(truth[ev, j], Y[ev, j], ctx["capacity"][j])
            alltime = M[:, j]
            m_all = solar_metrics(truth[alltime, j], Y[alltime, j], ctx["capacity"][j])
            inf = info.get(j, {})
            rows.append(dict(dataset=ds, scenario=sc, param=param, seed=seed, method=name,
                             var=ctx["columns"][j], ref=ctx["columns"][ctx["ref"]] if ref_override is not None else "",
                             seconds=dt / len(targets), rho_all=m_all["rho"], nrmse_all=m_all["nrmse"],
                             E=inf.get("E"), tau=inf.get("tau"),
                             fprm_refs=",".join(ctx["columns"][r] for r in inf.get("refs", [])),
                             n_fallback=inf.get("n_fallback"), **m))
    if seed == 0 and out_dir is not None:
        ex = Path(out_dir) / "examples"
        ex.mkdir(parents=True, exist_ok=True)
        tag = f"{ds}__{sc}__{param}" + (f"__ref{ref_override}" if ref_override is not None else "")
        np.savez_compressed(ex / f"{tag}.npz", targets=np.array(targets), mask=M[:, targets],
                            truth=truth[:, targets], **{k.replace("-", "_"): v for k, v in preds.items()})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = ROOT / "results" / cfg["run"]
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ds in cfg["datasets"]:
        for sc in cfg["scenarios"]:
            for param, kw in SCENARIOS[sc]:
                if cfg.get("params", {}).get(sc) and param not in cfg["params"][sc]:
                    continue
                for s in range(cfg["seeds"]):
                    jobs.append((ds, sc, param, kw, s, cfg["methods"], str(out), None))
        # S8: reference sweep -- every column in turn is the single designated reference
        if cfg.get("ref_sweep"):
            rs = cfg["ref_sweep"]
            df, _, _ = _ds(ds)
            for r in range(df.shape[1]):
                for param, kw in SCENARIOS[rs["scenario"]]:
                    if param not in rs["params"]:
                        continue
                    for s in range(rs["seeds"]):
                        jobs.append((ds, rs["scenario"], param, kw, s, rs["methods"], str(out), r))
    print(len(jobs), "jobs", flush=True)
    csv = out / "metrics.csv"
    first = not csv.exists()
    t0 = time.time()
    with cf.ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for k, f in enumerate(cf.as_completed(futs)):
            rows = f.result()
            if rows:
                pd.DataFrame(rows).to_csv(csv, mode="a", header=first, index=False)
                first = False
            j = futs[f]
            print(f"[{k + 1}/{len(jobs)}] {j[0]} {j[1]} {j[2]} s{j[4]} ref={j[7]} ({time.time() - t0:.0f}s)", flush=True)
