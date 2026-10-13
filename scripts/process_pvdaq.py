"""Resample PVDAQ per-day CSV files (one folder per system) to 15-min mean AC power.

    python scripts/process_pvdaq.py stpete   # column rtw (15-s samples)
    python scripts/process_pvdaq.py nola     # column w_avg (~1-min samples)

A 15-min bin is kept only if it holds >= 50% of the typical number of samples.
Writes data/processed/<name>_15min_all.parquet (all systems, NaN where no data).
"""
import concurrent.futures as cf
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
CFG = {"stpete": ("pvdaq_stpete", "rtw"), "nola": ("pvdaq_nola", "w_avg")}


def one_system(args):
    d, prefix = args
    parts = []
    for f in sorted(d.glob("*.csv")):
        x = pd.read_csv(f, parse_dates=["measured_on"])
        col = [c for c in x.columns if c.startswith(prefix)]
        if not col or x.empty:
            continue
        parts.append(x.set_index("measured_on")[col[0]])
    if not parts:
        return d.name, None
    s = pd.concat(parts)
    s = s[~s.index.duplicated()].sort_index()
    g = s.resample("15min")
    m, n = g.mean(), g.count()
    typical = n[n > 0].median()
    return d.name, m.where(n >= 0.5 * typical)


if __name__ == "__main__":
    name = sys.argv[1]
    raw, prefix = CFG[name]
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = [(d, prefix) for d in (ROOT / "data" / "raw" / raw).iterdir() if d.is_dir()]
    with cf.ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(one_system, dirs))
    df = pd.DataFrame({k: v for k, v in res.items() if v is not None}).asfreq("15min").sort_index(axis=1)
    df.to_parquet(OUT / f"{name}_15min_all.parquet")
    print(df.notna().resample("QS").mean().round(2).to_string())
