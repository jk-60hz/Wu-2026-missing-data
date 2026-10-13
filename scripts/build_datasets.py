"""Build analysis-ready, fully observed multi-variable PV datasets.

Every dataset is a window where all selected columns are observed (after
night-time gaps are set to 0), so it can serve as ground truth.
Output: data/processed/<name>.parquet  (index = naive UTC interval start)
        data/processed/datasets.json   (metadata: lat/lon, freq, capacity, columns)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.solar import clearsky  # noqa: E402

RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
META = {}


def finish(name, df, lat, lon, desc, max_gap_frac=0.0):
    cs, zen = clearsky(df.index, lat, lon)
    night = zen >= 90
    df = df.copy()
    for c in df.columns:
        if c == "ghi":
            df.loc[night, c] = df.loc[night, c].fillna(0)
        else:
            df.loc[night, c] = 0.0  # inverters are switched off / not logging at night
        df[c] = df[c].clip(lower=0)
    # tiny gaps (<= 2 steps) are linearly bridged; anything larger disqualifies the column
    df = df.interpolate(limit=2, limit_area="inside")
    bad = [c for c in df.columns if df[c].isna().mean() > max_gap_frac]
    if bad:
        print(name, "dropping incomplete columns", bad)
        df = df.drop(columns=bad)
    df = df.dropna()
    cap = {c: float(np.percentile(df[c], 99.9)) for c in df.columns}
    df.to_parquet(OUT / f"{name}.parquet")
    META[name] = dict(lat=lat, lon=lon, freq=pd.infer_freq(df.index) or str(df.index[1] - df.index[0]),
                      start=str(df.index[0]), end=str(df.index[-1]), n=len(df),
                      columns=list(df.columns), capacity=cap, description=desc)
    print(name, df.shape, df.index[0], df.index[-1])


def maui():
    R = RAW / "pvdaq_2105"
    inv = {}
    for i in range(1, 12):
        d = pd.read_csv(R / f"2105_inv{i:02d}_data.csv", parse_dates=["measured_on"]).set_index("measured_on")
        c = [c for c in d.columns if "ac_output_(kwh)" in c][0]
        s = d[c][~d.index.duplicated()]
        inv[f"inv{i:02d}"] = s.resample("15min").mean()
    inv = pd.DataFrame(inv)
    # clock offsets found by maximising correlation with pvlib clear-sky GHI (see data/README.md)
    inv.index = inv.index + pd.Timedelta(minutes=615)
    g = pd.read_csv(R / "2105_irradiance_data.csv", parse_dates=["measured_on"]).set_index("measured_on").iloc[:, 0]
    g = g[~g.index.duplicated()]
    g.index = g.index - pd.Timedelta(minutes=195)
    df = pd.concat([g.rename("ghi"), inv], axis=1, sort=True).asfreq("15min")
    # inv06 is stuck at a constant during daylight (dead channel); inv05 shows
    # multi-day partial-output faults -> neither is usable as ground truth
    df = df.drop(columns=["inv05", "inv06"])
    w = df["2022-01-10":"2022-03-10"]  # 60 days where all 11 inverters log (checked by coverage scan)
    finish("maui_15min", w, 20.884, -156.3405, "PVDAQ 2105 Maui Ocean Center: GHI + 11 string inverters, 15-min", 0.0)
    h = df["2021-11-01":"2022-04-30"].resample("1h").mean()  # hourly, 6 months
    finish("maui_1h", h, 20.884, -156.3405, "PVDAQ 2105 Maui Ocean Center: GHI + 11 string inverters, hourly", 0.0)


def stpete():
    df = pd.read_parquet(OUT / "stpete_15min_all.parquet")
    cols = ["1314", "1318", "1325", "1326"]  # 1321 has a clock shift, 1329 is corrupt (see README)
    # timestamps are local standard time (EST, UTC-5)
    w = df.loc["2014-12-01":"2015-04-30", cols].copy()
    w.index = w.index + pd.Timedelta(hours=5)
    w.columns = [f"s{c}" for c in cols]
    finish("stpete_1h", w.resample("1h").mean(), 27.77, -82.69,
           "PVDAQ St. Petersburg Parks production meters (4 sites, ~10 km apart), hourly", 0.0)


def nola():
    df = pd.read_parquet(OUT / "nola_15min_all.parquet")
    # 10 residential systems with ~100% coverage in the window, median pairwise corr > 0.92,
    # mixed orientations (peak hour 10-13 local); timestamps are local standard time (CST, UTC-6)
    cols = ["1263", "1244", "1250", "1252", "1254", "1255", "1257", "1264", "1265", "1271"]
    w = df.loc["2013-10-01":"2014-02-28", cols].copy()
    w.index = w.index + pd.Timedelta(hours=6)
    w.columns = [f"h{c}" for c in cols]
    finish("nola_1h", w.resample("1h").mean(), 29.97, -90.08,
           "PVDAQ New Orleans residential rooftop systems (10 houses, <15 km apart), hourly", 0.0)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["maui", "stpete", "nola"]
    mp = OUT / "datasets.json"
    if mp.exists():
        META.update(json.loads(mp.read_text()))
    for w in which:
        globals()[w]()
    mp.write_text(json.dumps(META, indent=1))
