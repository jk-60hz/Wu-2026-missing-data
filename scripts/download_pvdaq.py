"""Download NREL PVDAQ data from the public OEDI S3 bucket (no credentials needed).

Usage:
    python scripts/download_pvdaq.py maui      # system 2105 (Maui Ocean Center, 11 inverters + GHI)
    python scripts/download_pvdaq.py stpete    # St. Petersburg (FL) Parks production meters
    python scripts/download_pvdaq.py nola      # New Orleans (LA) residential systems
"""
import concurrent.futures as cf
import re
import sys
import urllib.request
from pathlib import Path

BUCKET = "https://oedi-data-lake.s3.amazonaws.com"
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"

# St. Petersburg Parks CDP "-Production Meter" systems (AC power/energy channels)
STPETE_IDS = [1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323,
              1324, 1325, 1326, 1328, 1329, 1330, 1331]
# New Orleans residential systems (gross production, not the "Net Meter" twins)
NOLA_IDS = list(range(1244, 1276))
MAUI_FILES = ["environment_1", "environment_2", "irradiance", "meter"] + [f"inv{i:02d}" for i in range(1, 12)]


def get(url, retries=4):
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                raise


def list_keys(prefix):
    keys, marker = [], ""
    while True:
        xml = get(f"{BUCKET}/?prefix={prefix}&marker={marker}").decode()
        batch = re.findall(r"<Key>([^<]+)</Key>", xml)
        keys += batch
        if "<IsTruncated>true</IsTruncated>" not in xml or not batch:
            return keys
        marker = batch[-1]


def fetch(key, out):
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(get(f"{BUCKET}/{key}"))


def maui():
    out = ROOT / "pvdaq_2105"
    base = "pvdaq/2023-solar-data-prize/2105_OEDI"
    jobs = [(f"{base}/data/2105_{f}_data.csv", out / f"2105_{f}_data.csv") for f in MAUI_FILES]
    jobs.append((f"{base}/metadata/2105_system_metadata.json", out / "2105_system_metadata.json"))
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))


def stpete(ids=STPETE_IDS, name="pvdaq_stpete"):
    out = ROOT / name
    fetch("pvdaq/csv/systems.csv", out / "systems.csv")
    jobs = []
    for sid in ids:
        for k in list_keys(f"pvdaq/csv/pvdata/system_id={sid}/"):
            if k.endswith(".csv"):
                jobs.append((k, out / str(sid) / k.rsplit("/", 1)[1]))
        try:
            fetch(f"pvdaq/csv/system_metadata/{sid}_system_metadata.json", out / f"{sid}_system_metadata.json")
        except Exception:
            pass  # metadata file not published for every system
    print(f"{len(jobs)} daily files")
    with cf.ThreadPoolExecutor(32) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))


if __name__ == "__main__":
    {"maui": maui, "stpete": stpete,
     "nola": lambda: stpete(NOLA_IDS, "pvdaq_nola")}[sys.argv[1]]()
