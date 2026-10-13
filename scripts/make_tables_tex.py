"""Generate the LaTeX tables of the technical report directly from results/.

    python scripts/make_tables_tex.py      # -> paper_tex/tables/{en,ko}/*.tex

Every number in the paper's tables is produced here (no hand copying).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "results" / "tables"
OUT = ROOT / "paper_tex" / "tables"

SC = ["S1_mcar", "S2_block", "S3_day", "S4_periodic", "S5_threshold", "S6_oneref", "S7_simul"]
SC_ID = {s: s.split("_")[0] for s in SC}
ORDER = ["FPRM", "FPRM-multi", "FPRM-csi", "FPRM-E1", "RF", "SVR", "MLP", "kNN", "GPR", "NbrOLS",
         "NbrRatio", "MICE", "KNNImputer", "CSI-Linear", "Diurnal", "Linear", "Mean"]
DS = ["stpete_1h", "nola_1h", "maui_1h", "maui_15min"]
L = {
    "en": dict(method="Method", all="All", dataset="Dataset", vars="Vars", res="Res.", period="Period", steps="Steps",
               ref="Ref.", comp="Composition", scen="Scenario", params="Parameters", tgt="Incomplete series",
               meaning="Real-world analogue", cells="Cells", med="Median rel.", wl="Win / sig.\\ win / sig.\\ loss",
               vsE1="vs.\\ FPRM-E1", system="System", var="Var.", paper="Paper FPRM", ours="FPRM-GPR (ours)",
               svr="FPRM-SVR", e1="GPR ($E{=}1$)", variant="Variant", desc="Description", rmse="RMSE (\\%cap)",
               sec="Time (s)", best="Best baseline", range_="range (mean)", gain="Embedding gain",
               group="Group", members="Methods"),
    "ko": dict(method="기법", all="전체", dataset="데이터셋", vars="변수", res="해상도", period="기간", steps="길이",
               ref="참조", comp="구성", scen="시나리오", params="파라미터", tgt="결측 변수", meaning="대표하는 현실 상황",
               cells="셀 수", med="중앙 상대오차", wl="승 / 유의승 / 유의패", vsE1="FPRM-E1 대비", system="시스템",
               var="변수", paper="논문 FPRM", ours="FPRM-GPR (본 구현)", svr="FPRM-SVR", e1="GPR ($E{=}1$)",
               variant="변형", desc="설명", rmse="RMSE (\\%cap)", sec="시간 (s)", best="최강 베이스라인",
               range_="범위 (평균)", gain="임베딩 이득", group="그룹", members="기법"),
}


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


def bold_min(df, fmt="{:.1f}", axis=0):
    """Format a numeric frame, bolding the minimum of each column (axis=0) or row (axis=1)."""
    out = df.copy().astype(object)
    mins = df.min(axis=axis)
    for i in df.index:
        for c in df.columns:
            v = df.loc[i, c]
            if pd.isna(v):
                out.loc[i, c] = "--"
                continue
            m = mins[c] if axis == 0 else mins[i]
            s = fmt.format(v)
            out.loc[i, c] = f"\\textbf{{{s}}}" if np.isclose(v, m) else s
    return out


def tabular(df, colspec, header, midrules=()):
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for k, (i, row) in enumerate(df.iterrows()):
        if k in midrules:
            lines.append("\\midrule")
        lab = i if not isinstance(i, tuple) else " & ".join(map(str, i))
        lines.append(f"{lab} & " + " & ".join(map(str, row.values)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write(lang, name, text):
    p = OUT / lang / f"{name}.tex"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def main_metrics():
    d = pd.read_csv(RES / "main" / "metrics.csv")
    d["rank"] = d.groupby(["dataset", "scenario", "param", "seed", "var"])["rmse_cap"].rank()
    return d


def t_datasets(lang):
    meta = json.loads((ROOT / "data" / "processed" / "datasets.json").read_text())
    comp = {"en": {"stpete_1h": "4 commercial roofs, $\\approx$10\\,km", "nola_1h": "10 residential roofs, $<$15\\,km",
                   "maui_1h": "GHI + 8 string inverters", "maui_15min": "GHI + 7 string inverters"},
            "ko": {"stpete_1h": "공원 건물 4곳, 약 10\\,km", "nola_1h": "주택 지붕 10곳, 15\\,km 이내",
                   "maui_1h": "GHI + 스트링 인버터 8개", "maui_15min": "GHI + 스트링 인버터 7개"}}[lang]
    ref = {"stpete_1h": "s1314", "nola_1h": "h1263", "maui_1h": "GHI", "maui_15min": "GHI"}
    rows = []
    for ds in DS:
        m = meta[ds]
        rows.append([f"\\texttt{{{esc(ds)}}}", comp[ds], len(m["columns"]), "15\\,min" if "15" in ds else "1\\,h",
                     f"{m['start'][:10]} -- {m['end'][:10]}", f"{m['n']:,}", ref[ds]])
    t = L[lang]
    df = pd.DataFrame(rows).set_index(0)
    write(lang, "datasets", tabular(df, "llrllrl", [t["dataset"], t["comp"], t["vars"], t["res"], t["period"],
                                                    t["steps"], t["ref"]]))


def t_scenarios(lang):
    en = [
        ("S1", "MCAR points", "$r \\in \\{0.1,0.3,0.5,0.7,0.9\\}$", "half of non-ref.", "packet loss"),
        ("S2", "contiguous blocks", "$L \\in \\{1,6,24,72,168\\}$\\,h, 20\\,\\%", "half of non-ref.", "communication/logger outage"),
        ("S3", "whole days", "5, 10, 20\\,\\% of days", "half of non-ref.", "day not collected"),
        ("S4", "periodic logging", "keep 1 of $P \\in \\{2,4,8\\}$", "half of non-ref.", "low-rate polling"),
        ("S5", "threshold (MNAR)", "upper/lower 5, 20\\,\\% (daytime)", "half of non-ref.", "saturation, clipping, low-light cut-off"),
        ("S6", "one complete series", "$r \\in \\{0.5,0.9\\}$", "all but ref.", "most neighbours degraded"),
        ("S7", "simultaneous blocks", "$L \\in \\{6,24\\}$\\,h, 10\\,\\%", "all incl.\\ ref.", "regional outage"),
        ("S8", "reference sweep", "S6, $r{=}0.5$, every ref.", "all but ref.", "choice of reference"),
    ]
    ko = [
        ("S1", "MCAR 무작위 점", "$r \\in \\{0.1,0.3,0.5,0.7,0.9\\}$", "비참조의 절반", "패킷 손실"),
        ("S2", "연속 블록", "$L \\in \\{1,6,24,72,168\\}$\\,h, 20\\,\\%", "비참조의 절반", "통신·로거 장애"),
        ("S3", "하루 전체", "날짜의 5, 10, 20\\,\\%", "비참조의 절반", "일 단위 미수집"),
        ("S4", "주기 로깅", "$P \\in \\{2,4,8\\}$ 중 1개 저장", "비참조의 절반", "저주기 폴링"),
        ("S5", "임계값 (MNAR)", "주간 상/하위 5, 20\\,\\%", "비참조의 절반", "포화·클리핑, 저조도 컷오프"),
        ("S6", "완전 시계열 1개", "$r \\in \\{0.5,0.9\\}$", "참조 외 전부", "이웃 대부분 열화"),
        ("S7", "동시 블록", "$L \\in \\{6,24\\}$\\,h, 10\\,\\%", "참조 포함 전부", "지역 단위 정전"),
        ("S8", "참조 스윕", "S6, $r{=}0.5$, 모든 참조", "참조 외 전부", "참조 선택"),
    ]
    t = L[lang]
    df = pd.DataFrame(en if lang == "en" else ko).set_index(0)
    write(lang, "scenarios", tabular(df, "lllll", ["ID", t["scen"], t["params"], t["tgt"], t["meaning"]]))


def t_methods(lang):
    rows = {"en": [
        ("Simple", "Mean, Linear (interpolation), Diurnal ($\\pm$7-day same-slot mean)"),
        ("PV practice", "CSI-Linear (clear-sky-index interpolation), NbrRatio (best neighbour $\\times$ $\\pm$3-day energy ratio), NbrOLS (OLS on all complete series)"),
        ("Direct ML", "kNN, GPR, SVR, RF, MLP --- current values of all complete series, no embedding"),
        ("Multivariate", "MICE (IterativeImputer), KNNImputer"),
        ("FPRM family", "FPRM (paper: GPR, single best reference), FPRM-E1 (no embedding), FPRM-multi$^\\dagger$ (all complete references), FPRM-csi$^\\dagger$ (clear-sky-index space)"),
        ("Ablation only", "FPRM-SVR, FPRM-MLP (paper's SVM$^*$/NN$^*$), FPRM-back$^\\dagger$ (causal embedding), FPRM-ref (designated reference), FPRM-E1-csi"),
    ], "ko": [
        ("단순", "Mean, Linear(선형 보간), Diurnal($\\pm$7일 같은 시각 평균)"),
        ("태양광 실무", "CSI-Linear(clear-sky index 보간), NbrRatio(최고 상관 이웃 $\\times$ $\\pm$3일 에너지 비율), NbrOLS(완전 시계열 전체에 대한 OLS)"),
        ("직접 ML 회귀", "kNN, GPR, SVR, RF, MLP --- 완전 시계열 전체의 현재값 입력, 임베딩 없음"),
        ("다변량 대치", "MICE(IterativeImputer), KNNImputer"),
        ("FPRM 계열", "FPRM(논문: GPR, 최적 참조 1개), FPRM-E1(임베딩 제거), FPRM-multi$^\\dagger$(완전 참조 전체), FPRM-csi$^\\dagger$(clear-sky index 공간)"),
        ("Ablation 전용", "FPRM-SVR, FPRM-MLP(논문의 SVM$^*$/NN$^*$), FPRM-back$^\\dagger$(과거 방향 임베딩), FPRM-ref(지정 참조), FPRM-E1-csi"),
    ]}[lang]
    t = L[lang]
    df = pd.DataFrame(rows).set_index(0)
    write(lang, "methods", tabular(df, "lp{0.74\\linewidth}", [t["group"], t["members"]]))


def t_synthetic(lang):
    d = pd.read_csv(RES / "synthetic" / "reproduction.csv")
    d = d[d.r == 0.9]
    paper = {("ecology", "y"): (0.99, 0.11), ("ecology", "z"): (0.99, 0.13),
             ("lorenz", "y"): (0.90, 0.48), ("lorenz", "z"): (0.81, 0.68)}
    name = {"ecology": "3D ecology", "lorenz": "Lorenz ($\\Delta t{=}0.02$)", "lorenz_coarse": "Lorenz ($\\Delta t{=}0.06$)"}
    rows = []
    for s in ["ecology", "lorenz", "lorenz_coarse"]:
        for v in ["y", "z"]:
            g = d[(d.system == s) & (d["var"] == v)].groupby("method")[["rho", "nrmse"]].mean()
            pp = paper.get((s, v))
            f = lambda m: f"{g.loc[m, 'rho']:.3f} / {g.loc[m, 'nrmse']:.3f}"
            rows.append([name[s] if v == "y" else "", f"${v}$",
                         f"{pp[0]:.2f} / {pp[1]:.2f}" if pp else "--", f("FPRM-GPR"), f("FPRM-SVR"), f("GPR(E=1)")])
    t = L[lang]
    df = pd.DataFrame(rows).set_index(0)
    write(lang, "synthetic", tabular(df, "llcccc", [t["system"], t["var"], t["paper"], t["ours"], t["svr"], t["e1"]],
                                     midrules=(2, 4)))


def t_rank_median(lang, d):
    t = L[lang]
    r = d.pivot_table(index="method", columns="scenario", values="rank")[SC]
    r[t["all"]] = d.groupby("method")["rank"].mean()
    r = r.loc[ORDER]
    r.columns = [SC_ID.get(c, c) for c in r.columns]
    write(lang, "rank", tabular(bold_min(r), "l" + "r" * r.shape[1], [t["method"]] + list(r.columns), midrules=(4, 9, 13)))
    m = d.pivot_table(index="method", columns="scenario", values="rmse_cap", aggfunc="median")[SC].loc[ORDER]
    m.columns = [SC_ID[c] for c in m.columns]
    write(lang, "median_rmse", tabular(bold_min(m), "l" + "r" * m.shape[1], [t["method"]] + list(m.columns),
                                       midrules=(4, 9, 13)))
    e = d.pivot_table(index="method", columns="scenario", values="energy_err", aggfunc="median")[SC].loc[ORDER]
    e.columns = [SC_ID[c] for c in e.columns]
    write(lang, "energy", tabular(bold_min(e), "l" + "r" * e.shape[1], [t["method"]] + list(e.columns),
                                  midrules=(4, 9, 13)))
    # per-dataset mean rank (appendix)
    pr = d.pivot_table(index="method", columns="dataset", values="rank")[DS].loc[ORDER]
    pr.columns = [f"\\texttt{{{esc(c)}}}" for c in pr.columns]
    write(lang, "rank_by_dataset", tabular(bold_min(pr), "lrrrr", [t["method"]] + list(pr.columns), midrules=(4, 9, 13)))


def t_family(lang):
    f = pd.read_csv(TAB / "family_summary.csv")
    t = L[lang]
    rows = []
    for sc in SC:
        a = f[(f.method == "FPRM") & (f.scenario == sc)].iloc[0]
        b = f[(f.method == "FPRM-multi") & (f.scenario == sc)].iloc[0]
        rows.append([SC_ID[sc], int(a.cells), f"{a.median_rel_vs_best:+.1f}\\,\\%",
                     f"{int(a.wins)} / {int(a.sig_wins)} / {int(a.sig_losses)}",
                     f"{b.median_rel_vs_best:+.1f}\\,\\%", f"{int(b.wins)} / {int(b.sig_wins)} / {int(b.sig_losses)}",
                     f"{a.median_rel_vs_e1:+.1f}\\,\\%"])
    df = pd.DataFrame(rows).set_index(0)
    hdr = [t["scen"], t["cells"], "\\multicolumn{2}{c}{FPRM}", "\\multicolumn{2}{c}{FPRM-multi}", "FPRM " + t["vsE1"]]
    body = tabular(df, "lrrcrcr", hdr)
    sub = f" & & {t['med']} & {t['wl']} & {t['med']} & {t['wl']} & \\\\\n\\cmidrule(lr){{3-4}}\\cmidrule(lr){{5-6}}\n"
    body = body.replace("\\\\\n\\midrule\n", "\\\\\n" + sub + "\\midrule\n", 1)
    write(lang, "family", body)


def t_s6(lang, d):
    t = L[lang]
    base = ["Mean", "Linear", "Diurnal", "CSI-Linear", "NbrRatio", "NbrOLS", "kNN", "GPR", "SVR", "RF", "MLP",
            "MICE", "KNNImputer"]
    cols = ["FPRM", "FPRM-E1", "NbrRatio", "RF", "GPR", "MICE"]
    g = d[d.scenario == "S6_oneref"].groupby(["dataset", "param", "method"]).rmse_cap.mean().unstack()
    rows = []
    for ds in DS:
        for p in ["r=0.5", "r=0.9"]:
            row = g.loc[(ds, p)]
            bb = row[base].idxmin()
            rows.append([f"\\texttt{{{esc(ds)}}}" if p == "r=0.5" else "", p.replace("r=", "$r{=}") + "$"]
                        + list(row[cols].values) + [f"{bb} ({row[bb]:.2f})"])
    df = pd.DataFrame(rows).astype(object)
    num = df.iloc[:, 2:8].astype(float)
    fm = bold_min(num, "{:.2f}", axis=1)
    for k, c in enumerate(df.columns[2:8]):
        df[c] = fm.iloc[:, k].values
    df = df.set_index(0)
    write(lang, "s6", tabular(df, "ll" + "r" * 6 + "l", [t["dataset"], "$r$"] + cols + [t["best"]],
                              midrules=(2, 4, 6)))


def t_ablation(lang):
    a = pd.read_csv(TAB / "ablation_by_param.csv")
    g = a.groupby("method")[["rmse_cap", "seconds"]].mean()
    desc = {"en": {"FPRM": "GPR, forward, best single reference", "FPRM-SVR": "SVR mapping (paper's SVM$^*$)",
                   "FPRM-MLP": "MLP mapping (paper's NN$^*$)", "FPRM-back": "backward (causal) embedding$^\\dagger$",
                   "FPRM-csi": "clear-sky-index space$^\\dagger$", "FPRM-E1": "no embedding ($E{=}1$)",
                   "FPRM-E1-csi": "no embedding, clear-sky-index space", "FPRM-ref": "designated reference (Maui: GHI)"},
            "ko": {"FPRM": "GPR, forward, 최적 단일 참조", "FPRM-SVR": "SVR 매핑(논문 SVM$^*$)",
                   "FPRM-MLP": "MLP 매핑(논문 NN$^*$)", "FPRM-back": "과거 방향(인과) 임베딩$^\\dagger$",
                   "FPRM-csi": "clear-sky index 공간$^\\dagger$", "FPRM-E1": "임베딩 제거($E{=}1$)",
                   "FPRM-E1-csi": "임베딩 제거 + clear-sky index", "FPRM-ref": "지정 참조 고정(Maui: GHI)"}}[lang]
    order = ["FPRM", "FPRM-SVR", "FPRM-MLP", "FPRM-back", "FPRM-csi", "FPRM-E1", "FPRM-E1-csi", "FPRM-ref"]
    g = g.loc[order]
    f = bold_min(g[["rmse_cap"]], "{:.2f}")
    s = bold_min(g[["seconds"]], "{:.2f}")
    df = pd.DataFrame({"d": [desc[m] for m in order], "r": f.rmse_cap.values, "s": s.seconds.values}, index=order)
    t = L[lang]
    write(lang, "ablation", tabular(df, "llrr", [t["variant"], t["desc"], t["rmse"], t["sec"]]))
    # per-dataset effect of the designated (GHI) reference
    q = a[a.method.isin(["FPRM", "FPRM-ref"]) & a.scenario.isin(["S1_mcar", "S2_block"])]
    q = q.pivot_table(index="dataset", columns="method", values="rmse_cap").loc[DS]
    q.index = [f"\\texttt{{{esc(i)}}}" for i in q.index]
    write(lang, "ablation_ref", tabular(q.map(lambda v: f"{v:.2f}"), "lrr", [t["dataset"], "FPRM", "FPRM-ref"]))


def t_refsweep(lang):
    s = pd.read_csv(TAB / "refsweep.csv")
    p = s.pivot_table(index=["dataset", "ref"], columns="method", values="rmse_cap")
    p["gain"] = (p["FPRM-ref"] - p["FPRM-E1"]) / p["FPRM-E1"] * 100
    rows = []
    for ds in DS:
        q = p.loc[ds]
        f = lambda c: f"{q[c].min():.1f}--{q[c].max():.1f} ({q[c].mean():.1f})"
        rows.append([f"\\texttt{{{esc(ds)}}}", len(q), f("FPRM-ref"), f("FPRM-E1"), f("NbrRatio"),
                     f"${q.gain.mean():.1f}$ (${q.gain.min():.0f}$, ${q.gain.max():.0f}$)",
                     f"{(q['FPRM-ref'] < q['NbrRatio']).sum()}/{len(q)}"])
    t = L[lang]
    df = pd.DataFrame(rows).set_index(0)
    hdr = [t["dataset"], "\\#" + t["ref"], "FPRM-ref", "FPRM-E1", "NbrRatio", t["gain"] + " (\\%)",
           "FPRM-ref $<$ NbrRatio"]
    write(lang, "refsweep", tabular(df, "lrllllc", hdr))


def t_full(lang):
    a = pd.read_csv(TAB / "main_by_param.csv")
    pv = a.pivot_table(index=["scenario", "param"], columns="method", values="rmse_cap")[ORDER]
    order = [(sc, p) for sc in SC for p in
             {"S1_mcar": ["r=0.1", "r=0.3", "r=0.5", "r=0.7", "r=0.9"],
              "S2_block": ["L=1h", "L=6h", "L=24h", "L=72h", "L=168h"],
              "S3_day": ["days=0.05", "days=0.1", "days=0.2"], "S4_periodic": ["P=2", "P=4", "P=8"],
              "S5_threshold": ["upper0.05", "upper0.2", "lower0.05", "lower0.2"],
              "S6_oneref": ["r=0.5", "r=0.9"], "S7_simul": ["L=6h", "L=24h"]}[sc]]
    pv = pv.loc[order]
    # MICE diverges in S4 (values ~1e45): cap display
    disp = pv.copy()
    f = bold_min(disp.clip(upper=999), axis=1)
    f = f.where(~(pv > 999), "div.")
    f.index = [f"{SC_ID[s]} & {esc(p)}" for s, p in pv.index]
    hdr = [L[lang]["scen"], L[lang]["params"]] + [f"\\rotatebox{{90}}{{{m}}}" for m in ORDER]
    breaks = tuple(int(x) for x in np.cumsum([5, 5, 3, 3, 4, 2]))
    write(lang, "full", tabular(f, "ll" + "r" * len(ORDER), hdr, midrules=breaks))


if __name__ == "__main__":
    d = main_metrics()
    for lang in ("en", "ko"):
        t_datasets(lang)
        t_scenarios(lang)
        t_methods(lang)
        t_synthetic(lang)
        t_rank_median(lang, d)
        t_family(lang)
        t_s6(lang, d)
        t_ablation(lang)
        t_refsweep(lang)
        t_full(lang)
    print("tables written to", OUT)
