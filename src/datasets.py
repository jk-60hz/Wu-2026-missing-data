"""Load processed datasets and the solar context used by methods and metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .solar import clearsky

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# designated reference column per dataset ("complete" series for FPRM-ref / S6)
REFERENCE = {"maui_15min": "ghi", "maui_1h": "ghi", "stpete_1h": "s1314", "nola_1h": "h1263"}


def load(name):
    meta = json.loads((PROC / "datasets.json").read_text())[name]
    df = pd.read_parquet(PROC / f"{name}.parquet")
    cs, zen = clearsky(df.index, meta["lat"], meta["lon"])
    local = df.index + pd.Timedelta(hours=meta["lon"] / 15)  # local solar time
    step_min = int((df.index[1] - df.index[0]).total_seconds() // 60)
    tod = (local.hour * 60 + local.minute).to_numpy() // step_min
    day_idx = ((local.normalize() - local.normalize()[0]).days).to_numpy()
    ref_name = REFERENCE.get(name) or meta.get("reference") or df.columns[0]
    ctx = dict(cs=cs, zenith=zen, tod=tod, day_idx=day_idx, ref=list(df.columns).index(ref_name),
               sph=60 // step_min, columns=list(df.columns),
               capacity=np.array([meta["capacity"][c] for c in df.columns]))
    return df, meta, ctx
