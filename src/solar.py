"""Solar-geometry helpers: clear-sky reference, day/night masks, clear-sky index."""
import numpy as np
import pandas as pd
import pvlib


def clearsky(index_utc, lat, lon, alt=0.0):
    """Clear-sky GHI (W/m2, Ineichen) and solar zenith for a naive-UTC DatetimeIndex.

    Values are computed at the middle of each interval (labels are interval start)."""
    step = index_utc[1] - index_utc[0]
    mid = (index_utc + step / 2).tz_localize("UTC")
    loc = pvlib.location.Location(lat, lon, altitude=alt)
    cs = loc.get_clearsky(mid)["ghi"].to_numpy()
    zen = loc.get_solarposition(mid)["apparent_zenith"].to_numpy()
    return cs, zen


def day_mask(zenith, max_zenith=85.0):
    return np.asarray(zenith) < max_zenith


class ClearSkyIndex:
    """Transform each column to a clear-sky index k = P / (s_c * GHI_cs).

    s_c is a per-column scale fitted on observed values (95th pct ratio on
    near-noon points), so k ~ 1 under clear sky for every series. At night (and
    very low sun) k is set to 0 and the back-transform returns 0.
    """

    def __init__(self, cs, zenith, max_zenith=85.0):
        self.cs = np.asarray(cs, float)
        self.ok = (np.asarray(zenith) < max_zenith) & (self.cs > 20)

    def fit(self, X):
        X = np.asarray(X, float)
        self.scale = np.ones(X.shape[1])
        hi = self.ok & (self.cs > np.nanpercentile(self.cs[self.ok], 60))
        for c in range(X.shape[1]):
            r = X[hi, c] / self.cs[hi]
            r = r[np.isfinite(r)]
            if len(r):
                self.scale[c] = max(np.percentile(r, 95), 1e-9)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        K = np.zeros_like(X)
        den = self.cs[:, None] * self.scale[None, :]
        K[self.ok] = X[self.ok] / den[self.ok]
        K[np.isnan(X)] = np.nan
        return K

    def inverse(self, K):
        X = np.zeros_like(K)
        den = self.cs[:, None] * self.scale[None, :]
        X[self.ok] = K[self.ok] * den[self.ok]
        return X
