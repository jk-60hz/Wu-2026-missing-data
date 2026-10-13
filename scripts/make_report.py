"""Aggregate benchmark CSVs into tables and figures used by REPORT.md.

    python scripts/make_report.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES, FIG, TAB = ROOT / "results", ROOT / "results" / "figures", ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# reference categorical palette (dataviz skill), fixed order, light mode
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 5, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

DS = ["stpete_1h", "nola_1h", "maui_1h", "maui_15min"]
DS_LABEL = {"stpete_1h": "St. Pete (4 sites, 1h)", "nola_1h": "New Orleans (10 houses, 1h)",
            "maui_1h": "Maui (GHI + 8 inv., 1h)", "maui_15min": "Maui (GHI + 7 inv., 15min)"}
FPRM_FAMILY = [m for m in ["FPRM", "FPRM-E1", "FPRM-multi", "FPRM-csi", "FPRM-ref", "FPRM-SVR",
                           "FPRM-MLP", "FPRM-back", "FPRM-E1-csi"]]
BASELINES = ["Mean", "Linear", "Diurnal", "CSI-Linear", "NbrRatio", "NbrOLS", "kNN", "GPR", "SVR", "RF",
             "MLP", "MICE", "KNNImputer"]
PARAM_ORDER = {
    "S1_mcar": ["r=0.1", "r=0.3", "r=0.5", "r=0.7", "r=0.9"],
    "S2_block": ["L=1h", "L=6h", "L=24h", "L=72h", "L=168h"],
    "S3_day": ["days=0.05", "days=0.1", "days=0.2"],
    "S4_periodic": ["P=2", "P=4", "P=8"],
    "S5_threshold": ["upper0.05", "upper0.2", "lower0.05", "lower0.2"],
    "S6_oneref": ["r=0.5", "r=0.9"],
    "S7_simul": ["L=6h", "L=24h"],
}


def agg(d, keys):
    g = d.groupby(keys)
    out = g[["nrmse", "rho", "rmse_cap", "mae_cap", "bias_cap", "energy_err", "seconds"]].mean()
    out["nrmse_sd"] = g["nrmse"].std()
    out["n"] = g.size()
    return out.reset_index()


def synthetic():
    p = RES / "synthetic" / "reproduction.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    t = d.groupby(["system", "method", "var", "r"])[["rho", "nrmse", "E", "tau"]].agg(["mean", "std"]).round(3)
    t.to_csv(TAB / "synthetic.csv")
    systems = list(d.system.unique())
    fig, ax = plt.subplots(2, len(systems), figsize=(3.2 * len(systems), 4.6), sharex=True)
    ax = np.atleast_2d(ax).reshape(2, -1)
    meths = ["FPRM-GPR", "FPRM-SVR", "FPRM-MLP", "GPR(E=1)"]
    for k, s in enumerate(systems):
        for i, m in enumerate(meths):
            for var, ls in (("y", "-"), ("z", "--")):
                q = d[(d.system == s) & (d.method == m) & (d["var"] == var)].groupby("r")
                ax[0, k].plot(q.rho.mean().index, q.rho.mean(), ls, color=C[i], marker="o",
                              label=f"{m} ({var})" if k == 0 else None)
                ax[1, k].plot(q.nrmse.mean().index, q.nrmse.mean(), ls, color=C[i], marker="o")
        ax[0, k].set_title(s, color=INK)
        ax[1, k].set_xlabel("missing ratio r")
    ax[0, 0].set_ylabel("Pearson ρ")
    ax[1, 0].set_ylabel("normalised RMSE")
    fig.legend(loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "synthetic_reproduction.png", dpi=150)
    plt.close(fig)


def main_tables(d):
    a = agg(d, ["dataset", "scenario", "param", "method"])
    a.to_csv(TAB / "main_by_param.csv", index=False)
    # rank of each method within each (testset, variable) by rmse_cap
    d = d.copy()
    d["rank"] = d.groupby(["dataset", "scenario", "param", "seed", "var"])["rmse_cap"].rank()
    r = d.groupby(["scenario", "method"])["rank"].mean().unstack(0).round(2)
    r.to_csv(TAB / "mean_rank_by_scenario.csv")
    # pivot: rows (scenario,param), columns method, value daytime rmse %cap, averaged over datasets
    pv = a.pivot_table(index=["scenario", "param"], columns="method", values="rmse_cap").round(2)
    pv.to_csv(TAB / "rmse_cap_pivot.csv")
    # FPRM vs best non-FPRM baseline per (dataset, scenario, param): paired Wilcoxon over seed x var
    rows = []
    for (ds, sc, pa), g in d.groupby(["dataset", "scenario", "param"]):
        base = g[g.method.isin(BASELINES)].groupby("method").rmse_cap.mean()
        if base.empty or "FPRM" not in set(g.method):
            continue
        best = base.idxmin()
        f = g[g.method == "FPRM"].set_index(["seed", "var"]).rmse_cap
        b = g[g.method == best].set_index(["seed", "var"]).rmse_cap
        e1 = g[g.method == "FPRM-E1"].set_index(["seed", "var"]).rmse_cap
        j = pd.concat([f, b, e1], axis=1, keys=["f", "b", "e1"]).dropna()
        try:
            p_best = wilcoxon(j.f, j.b).pvalue if (j.f != j.b).any() else 1.0
            p_e1 = wilcoxon(j.f, j.e1).pvalue if (j.f != j.e1).any() else 1.0
        except ValueError:
            p_best = p_e1 = np.nan
        rows.append(dict(dataset=ds, scenario=sc, param=pa, fprm=j.f.mean(), best_baseline=best,
                         best=j.b.mean(), fprm_e1=j.e1.mean(), rel_vs_best=(j.f.mean() - j.b.mean()) / j.b.mean() * 100,
                         p_vs_best=p_best, rel_vs_e1=(j.f.mean() - j.e1.mean()) / j.e1.mean() * 100,
                         p_vs_e1=p_e1, n_pairs=len(j)))
    w = pd.DataFrame(rows)
    w.to_csv(TAB / "fprm_vs_best_baseline.csv", index=False)
    return a, w


def curves(a, scenario, xlabel, fname, methods):
    params = PARAM_ORDER[scenario]
    fig, ax = plt.subplots(1, len(DS), figsize=(3.1 * len(DS), 3.0), sharey=False)
    for k, ds in enumerate(DS):
        for i, m in enumerate(methods):
            q = a[(a.dataset == ds) & (a.scenario == scenario) & (a.method == m)].set_index("param").reindex(params)
            ax[k].plot(range(len(params)), q.rmse_cap, color=C[i], marker="o", label=m if k == 0 else None,
                       linewidth=2.6 if m == "FPRM" else 1.6)
        ax[k].set_xticks(range(len(params)))
        ax[k].set_xticklabels([p.split("=")[-1] for p in params])
        ax[k].set_title(DS_LABEL[ds], color=INK, fontsize=9)
        ax[k].set_xlabel(xlabel)
    ax[0].set_ylabel("daytime RMSE (% of capacity)")
    fig.legend(loc="lower center", ncol=len(methods), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def skill_heatmap(w):
    """Diverging heatmap: FPRM vs best baseline (% RMSE change; blue = FPRM better)."""
    rows = [(sc, p) for sc, ps in PARAM_ORDER.items() for p in ps]
    M = np.full((len(rows), len(DS)), np.nan)
    for i, (sc, p) in enumerate(rows):
        for k, ds in enumerate(DS):
            q = w[(w.dataset == ds) & (w.scenario == sc) & (w.param == p)]
            if len(q):
                M[i, k] = q.rel_vs_best.iloc[0]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
    lim = np.nanmax(np.abs(np.clip(M, -60, 60))) if np.isfinite(M).any() else 1
    fig, ax = plt.subplots(figsize=(6.4, 9))
    im = ax.imshow(np.clip(M, -lim, lim), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.grid(False)
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):
            if np.isfinite(M[i, k]):
                q = w[(w.dataset == DS[k]) & (w.scenario == rows[i][0]) & (w.param == rows[i][1])]
                sig = "*" if q.p_vs_best.iloc[0] < 0.05 else ""
                ax.text(k, i, f"{M[i, k]:+.0f}%{sig}\n{q.best_baseline.iloc[0]}", ha="center", va="center",
                        fontsize=6.5, color=INK)
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels([DS_LABEL[d].replace(" (", "\n(") for d in DS], fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{sc.split('_')[0]} {p}" for sc, p in rows], fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.5)
    cb.set_label("FPRM RMSE vs best baseline (%)\n(negative = FPRM better)")
    ax.set_title("FPRM vs best non-FPRM method per test set\n(* paired Wilcoxon p<0.05; label = best baseline)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fprm_vs_best_heatmap.png", dpi=150)
    plt.close(fig)


def example_plot(ds, tag, methods, fname, title):
    p = RES / "main" / "examples" / f"{ds}__{tag}.npz"
    if not p.exists():
        return
    from src.datasets import load
    df, meta, ctx = load(ds)
    z = np.load(p)
    k = 0
    miss = z["mask"][:, k]
    truth = z["truth"][:, k]
    # pick three days with missing daytime values: clearest, cloudiest, most variable (by truth)
    day_id = ctx["day_idx"]
    cand = [dd for dd in np.unique(day_id) if (miss & (day_id == dd) & (ctx["zenith"] < 85)).sum() >= 3]
    if not cand:
        return
    cs = ctx["cs"]
    ratio = {dd: truth[day_id == dd].sum() / max(cs[day_id == dd].sum(), 1) for dd in cand}
    var = {dd: np.abs(np.diff(truth[day_id == dd])).sum() / max(truth[day_id == dd].sum(), 1e-9) for dd in cand}
    picks = [max(ratio, key=ratio.get), min(ratio, key=ratio.get), max(var, key=var.get)]
    names = ["clearest day", "cloudiest day", "most variable day"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for a_, dd, nm in zip(ax, picks, names):
        sel = np.flatnonzero((day_id == dd) & (ctx["zenith"] < 92))
        x = np.arange(len(sel)) / ctx["sph"]
        a_.plot(x, truth[sel], color=INK, linewidth=1.2, label="ground truth")
        a_.scatter(x[miss[sel]], truth[sel][miss[sel]], s=10, color=INK2, zorder=3, label="removed values")
        for i, m in enumerate(methods):
            key = m.replace("-", "_")
            if key in z:
                y = z[key][:, k][sel].astype(float)
                y[~miss[sel]] = np.nan
                a_.plot(x, y, color=C[i], marker="o", markersize=3, linewidth=1.4, label=m)
        a_.set_title(f"{nm} ({df.index[sel[0]].date()})", fontsize=9, color=INK)
        a_.set_xlabel("hours after sunrise")
    ax[0].set_ylabel(f"{ctx['columns'][int(z['targets'][k])]} AC output")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def ablation():
    p = RES / "ablation" / "metrics.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    main = d[d.ref.isna()]
    a = agg(main, ["dataset", "scenario", "param", "method"])
    a.to_csv(TAB / "ablation_by_param.csv", index=False)
    pv = a.pivot_table(index=["scenario", "param"], columns="method", values="rmse_cap").round(2)
    pv.to_csv(TAB / "ablation_pivot.csv")
    sw = d[d.ref.notna()]
    if len(sw):
        s = agg(sw, ["dataset", "ref", "method"])
        s.to_csv(TAB / "refsweep.csv", index=False)


if __name__ == "__main__":
    synthetic()
    p = RES / "main" / "metrics.csv"
    if p.exists():
        d = pd.read_csv(p)
        a, w = main_tables(d)
        cm = ["FPRM", "FPRM-E1", "FPRM-csi", "NbrRatio", "NbrOLS", "CSI-Linear"]
        curves(a, "S1_mcar", "missing ratio r", "s1_mcar_curves.png", cm)
        curves(a, "S2_block", "gap length (hours)", "s2_block_curves.png", cm)
        curves(a, "S6_oneref", "missing ratio r (only 1 complete series)", "s6_oneref_curves.png", cm)
        skill_heatmap(w)
        for ds in DS:
            example_plot(ds, "S2_block__L=24h", ["FPRM", "FPRM-csi", "NbrRatio", "CSI-Linear"],
                         f"example_{ds}_S2_24h.png", f"{DS_LABEL[ds]} — S2 24-h gaps, seed 0")
            example_plot(ds, "S6_oneref__r=0.9", ["FPRM", "FPRM-csi", "NbrRatio", "CSI-Linear"],
                         f"example_{ds}_S6_r09.png", f"{DS_LABEL[ds]} — S6 one complete series, r=0.9, seed 0")
    ablation()
    print("ok")
