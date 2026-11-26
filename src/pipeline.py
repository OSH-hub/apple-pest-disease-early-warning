"""Unified early warning of first threshold exceedance for apple pests and diseases.

Unit      a survey of a county in a year at which the index has stayed below its threshold
          in every survey of the season so far.
Target    the index is at or above the threshold at the survey h rounds later
          (h = 1, about 15 days; h = 2, about 30 days).
Design    rolling origin: each test year is predicted by a model fitted to earlier years only.

Usage: python src/pipeline.py --project <project root> --out <output dir>
Writes tables/A*.csv and results/predictions_h*.csv under the output directory.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 20261126
N_BOOT = 2000
MIN_HISTORY = 3
LAST_YEAR = 2025
ITEMS = {
    "갈색무늬병(피해잎율)": "MB", "점무늬낙엽병(피해잎율)": "AB", "복숭아순나방(트랩마리수)": "OFM_trap",
    "복숭아순나방(피해신초율)": "OFM_shoot", "복숭아순나방(피해과율)": "OFM_fruit", "사과응애(100잎당마리수)": "ERM",
}
INDICES = list(ITEMS.values())
HELD_OUT = {"사과혹진딧물(640잎당마리수)": "LCA"}     # index kept out of model development, used as an independent test
LABEL = {"MB": "Marssonina blotch (damaged leaves, %)", "AB": "Alternaria blotch (damaged leaves, %)",
         "OFM_trap": "Oriental fruit moth (trap catch)", "OFM_shoot": "Oriental fruit moth (damaged shoots, %)",
         "OFM_fruit": "Oriental fruit moth (damaged fruit, %)", "ERM": "European red mite (per 100 leaves)"}
LEAF_DISEASE_THRESHOLD = 1.0      # % damaged leaves, for MB and AB
REFERENCE_YEARS = (2015, 2017)    # thresholds of the other indices: median positive value in these years
GROUPS = {
    "state": ["cur", "prev", "rise", "season_max", "n_surveys"],
    "season": ["doy_t", "gap", "clim"],
    "calendar": ["doy_t", "gap"],
    "history": ["hist_freq", "last_year_max"],
    "neighbours": ["nat_cur", "prov_cur", "nat_exceeded"],
    "weather": ["t15", "h15", "r15", "rd15", "t30", "h30", "r30", "rd30"],
}
CORE = ["state", "season", "history", "neighbours"]


# ---------------------------------------------------------------- data
def load_panel(root: Path) -> tuple[pd.DataFrame, dict]:
    a = pd.read_csv(root / "data/ncpms_api/apple_surveillance_sigungu.csv", dtype=str)
    flow = {"rows_all_indices": len(a)}
    a = a[a.dbyhsNm.isin({**ITEMS, **HELD_OUT})].copy()
    a["item"] = a.dbyhsNm.map({**ITEMS, **HELD_OUT})
    a["year"] = a.examinYear.astype(int)
    a["v"] = pd.to_numeric(a.inqireValue, errors="coerce")
    a["date"] = pd.to_datetime(a.inputStdrDatetm, format="%Y%m%d")
    a = a[a.year <= LAST_YEAR]
    flow["rows_six_indices"] = a.groupby("item").size().to_dict()
    a = a[a.v.notna()]
    flow["rows_numeric"] = a.groupby("item").size().to_dict()
    w = a.pivot_table(index=["sigunguNm", "sidoCode", "year", "date"], columns="item", values="v", aggfunc="first").reset_index()
    w["doy"] = w.date.dt.dayofyear
    return w.sort_values(["sigunguNm", "year", "date"]).reset_index(drop=True), flow


def thresholds(w: pd.DataFrame) -> dict[str, float]:
    ref = w[w.year.between(*REFERENCE_YEARS)]
    tau = {"MB": LEAF_DISEASE_THRESHOLD, "AB": LEAF_DISEASE_THRESHOLD}
    for it in INDICES[2:] + list(HELD_OUT.values()):
        tau[it] = float(ref.loc[ref[it] > 0, it].median())
    return tau


def load_weather(root: Path) -> pd.DataFrame:
    wx = pd.read_csv(root / "data/pairing/pairing_daily.csv", usecols=["sigunguNm", "model", "date", "temp_avg", "hum", "rain"])
    wx = wx[wx.model == "AMB"].copy()
    wx["date"] = pd.to_datetime(wx.date)
    for c in ("temp_avg", "hum", "rain"):
        wx[c] = pd.to_numeric(wx[c], errors="coerce")
    wx = wx.sort_values(["sigunguNm", "date"])
    g = wx.groupby("sigunguNm")
    for win in (15, 30):
        mp = int(win * 0.8)
        wx[f"t{win}"] = g.temp_avg.transform(lambda s: s.rolling(win, min_periods=mp).mean())
        wx[f"h{win}"] = g.hum.transform(lambda s: s.rolling(win, min_periods=mp).mean())
        wx[f"r{win}"] = g.rain.transform(lambda s: np.log1p(s.fillna(0).rolling(win, min_periods=mp).sum()))
        wx[f"rd{win}"] = g.rain.transform(lambda s: (s.fillna(0) >= 1).rolling(win, min_periods=mp).mean())
    return wx[["sigunguNm", "date"] + GROUPS["weather"]].rename(columns={"sigunguNm": "county"})


# ---------------------------------------------------------------- units and features
def build_units(w: pd.DataFrame, item: str, h: int, at_risk: str = "first") -> pd.DataFrame:
    """At-risk surveys of one index. Values in ``w`` are already divided by the threshold."""
    rows = []
    for (county, year), s in w.dropna(subset=[item]).groupby(["sigunguNm", "year"]):
        s = s.reset_index(drop=True)
        v = s[item].to_numpy()
        for i in range(len(s) - h):
            gap = (s.date[i + h] - s.date[i]).days
            if not 7 * h <= gap <= 25 * h:
                continue
            below = v[: i + 1].max() < 1.0 if at_risk == "first" else v[i] < 1.0
            if not below:
                continue
            prev = np.log1p(v[i - 1]) if i else 0.0
            rows.append({"index": item, "county": county, "prov": s.sidoCode[i], "year": year, "date": s.date[i],
                         "doy": s.doy[i], "doy_t": s.doy[i + h], "gap": gap, "cur": np.log1p(v[i]), "prev": prev,
                         "rise": np.log1p(v[i]) - prev, "season_max": np.log1p(v[: i + 1].max()), "n_surveys": i + 1,
                         "y": float(v[i + h] >= 1.0), "mid": float(h == 2 and v[i + 1] >= 1.0)})
    return pd.DataFrame(rows)


def add_neighbours(d: pd.DataFrame, w: pd.DataFrame, item: str) -> None:
    """State of the other counties in surveys up to the current date of the same year."""
    by_year = {y: g for y, g in w.dropna(subset=[item]).groupby("year")}
    nat_cur, prov_cur, nat_exc = [], [], []
    for r in d.itertuples():
        g = by_year[r.year]
        g = g[(g.sigunguNm != r.county) & (g.date <= r.date)]
        recent = g[g.date >= r.date - pd.Timedelta(days=10)]
        nat_cur.append(np.log1p(recent[item]).mean() if len(recent) else np.nan)
        pr = recent[recent.sidoCode == r.prov]
        prov_cur.append(np.log1p(pr[item]).mean() if len(pr) else np.nan)
        seen = g.groupby("sigunguNm")[item].max()
        nat_exc.append((seen >= 1.0).mean() if len(seen) else np.nan)
    d["nat_cur"], d["prov_cur"], d["nat_exceeded"] = nat_cur, prov_cur, nat_exc


def add_history(d: pd.DataFrame, w: pd.DataFrame, item: str) -> None:
    peak = w.dropna(subset=[item]).groupby(["sigunguNm", "year"])[item].max()
    freq, last = [], []
    for r in d.itertuples():
        p = peak.get(r.county)
        p = p[p.index < r.year] if p is not None else pd.Series(dtype=float)
        freq.append((p.ge(1.0).sum() + 1) / (len(p) + 2))
        last.append(np.log1p(p.get(r.year - 1, np.nan)))
    d["hist_freq"], d["last_year_max"] = freq, last


def climatology(d: pd.DataFrame, fit: np.ndarray, apply: np.ndarray) -> np.ndarray:
    """Seasonal hazard of first exceedance (15-day bins of the target date) times a shrunk county ratio."""
    b = (d.doy_t // 15).to_numpy()
    yv = d.y.to_numpy()
    n = np.bincount(b[fit], minlength=b.max() + 1).astype(float)
    e = np.bincount(b[fit], weights=yv[fit], minlength=b.max() + 1)
    overall = yv[fit].mean() if fit.any() else 0.0
    base = ((e + 2 * overall) / (n + 2))[b]
    county = d.county.to_numpy()
    obs = pd.Series(yv[fit]).groupby(county[fit]).sum()
    exp = pd.Series(base[fit]).groupby(county[fit]).sum()
    ratio = ((obs + 2) / (exp + 2)).reindex(np.unique(county)).fillna(1.0)
    return np.clip(base[apply] * pd.Series(county[apply]).map(ratio).to_numpy(), 1e-4, 1 - 1e-4)


def make_dataset(w: pd.DataFrame, h: int, at_risk: str = "first", weather: pd.DataFrame | None = None) -> pd.DataFrame:
    parts = []
    for it in INDICES:
        d = build_units(w, it, h, at_risk)
        add_neighbours(d, w, it)
        add_history(d, w, it)
        parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    D["clim"] = np.nan
    for it in INDICES:
        D[f"is_{it}"] = (D["index"] == it).astype(float)
    if weather is not None:
        D = D.merge(weather, on=["county", "date"], how="left")
    return D


# ---------------------------------------------------------------- models
def fit_predict(d: pd.DataFrame, cols: list[str], tr: np.ndarray, te: np.ndarray, kind: str = "logit") -> np.ndarray:
    x = d[cols]
    x = x.fillna(x[tr].median()).fillna(0.0)
    if kind == "gb":
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20, random_state=SEED)
        return m.fit(x[tr], d.y[tr]).predict_proba(x[te])[:, 1]
    sc = StandardScaler().fit(x[tr])
    m = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(x[tr]), d.y[tr])
    return m.predict_proba(sc.transform(x[te]))[:, 1]


def feature_cols(groups: list[str]) -> list[str]:
    return [c for g in groups for c in GROUPS[g]]


def fill_climatology(D: pd.DataFrame, y: int) -> None:
    """Climatology of every index for test year ``y`` and, out of year, for its training years."""
    for it in INDICES:
        k = (D["index"] == it).to_numpy()
        d = D[k].reset_index(drop=True)
        tr, te = (d.year < y).to_numpy(), (d.year == y).to_numpy()
        clim = np.full(len(d), np.nan)
        if d.y[tr].sum() >= 10 and te.any():
            clim[te] = climatology(d, tr, te)
            for yy in np.unique(d.year[tr]):
                a = tr & (d.year == yy).to_numpy()
                clim[a] = climatology(d, tr & ~a, a)
        D.loc[k, "clim"] = clim


def rolling(D: pd.DataFrame, variants: dict[str, dict]) -> dict[str, np.ndarray]:
    """Rolling-origin predictions. ``variants`` maps a name to {groups, kind, scope, dummies, exclude_index}."""
    years = np.sort(D.year.unique())
    P = {k: np.full(len(D), np.nan) for k in ["Persistence", "Climatology", *variants]}
    dummies = [f"is_{it}" for it in INDICES]
    for y in years[MIN_HISTORY:]:
        fill_climatology(D, y)
        valid = D.clim.notna().to_numpy()
        tr_all, te_all = valid & (D.year < y).to_numpy(), valid & (D.year == y).to_numpy()
        P["Persistence"][te_all] = D.cur[te_all]
        P["Climatology"][te_all] = D.clim[te_all]
        for name, v in variants.items():
            cols = feature_cols(v["groups"])
            if v.get("scope", "pooled") == "separate":
                for it in INDICES:
                    k = (D["index"] == it).to_numpy()
                    if (tr_all & k).any() and (te_all & k).any():
                        P[name][te_all & k] = fit_predict(D, cols, tr_all & k, te_all & k, v.get("kind", "logit"))
            elif v.get("scope") == "transfer":      # each index predicted by a model that never saw it
                for it in INDICES:
                    k = (D["index"] == it).to_numpy()
                    if (te_all & k).any():
                        P[name][te_all & k] = fit_predict(D, cols, tr_all & ~k, te_all & k, v.get("kind", "logit"))
            else:
                use = cols + (dummies if v.get("dummies", True) else [])
                P[name][te_all] = fit_predict(D, use, tr_all, te_all, v.get("kind", "logit"))
    return P


def held_out_index(D: pd.DataFrame, w: pd.DataFrame, item: str, h: int, tag: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """An index that took no part in model development, predicted by the model fitted to the six indices."""
    E = build_units(w, item, h)
    add_neighbours(E, w, item)
    add_history(E, w, item)
    E["clim"] = np.nan
    variants = {"Unified (seasonal record of the index)": feature_cols(CORE),
                "Unified (no record of the index)": feature_cols(["state", "calendar", "neighbours"])}
    P = {k: np.full(len(E), np.nan) for k in ["Persistence", "Climatology", *variants]}
    for y in np.sort(D.year.unique())[MIN_HISTORY:]:
        fill_climatology(D, y)
        tr = D.clim.notna().to_numpy() & (D.year < y).to_numpy()
        e_tr, e_te = (E.year < y).to_numpy(), (E.year == y).to_numpy()
        if not e_te.any() or E.y[e_tr].sum() < 10:
            continue
        E.loc[e_te, "clim"] = climatology(E, e_tr, e_te)
        both = pd.concat([D.loc[tr, list(E.columns)], E[e_te]], ignore_index=True)
        a = np.arange(len(both)) < tr.sum()
        P["Persistence"][e_te], P["Climatology"][e_te] = E.cur[e_te], E.clim[e_te]
        for name, cols in variants.items():
            P[name][e_te] = fit_predict(both, cols, a, ~a)
    ok = ~np.isnan(P["Persistence"])
    y, years = E.y[ok].to_numpy(), E.year[ok].to_numpy()
    W = boot_weights(years, np.random.default_rng(SEED))
    draws = {k: np.array([wauc(p[ok], y, wt) for wt in W]) for k, p in P.items()}
    rows, diffs = [], []
    for k, p in P.items():
        lo, hi = interval(draws[k])
        pod, pofd = matched(p[ok], y, 0.2)
        rows.append({**tag, "index": item, "forecast": k, "units": int(ok.sum()), "events": int(y.sum()), "test_years": len(np.unique(years)),
                     "AUC": roc_auc_score(y, p[ok]), "AUC_low": lo, "AUC_high": hi, "POD_top20": pod, "POFD_top20": pofd})
        for b in ("Persistence", "Climatology"):
            if k not in ("Persistence", "Climatology"):
                lo, hi = interval(draws[k] - draws[b])
                diffs.append({**tag, "index": item, "forecast": k, "minus": b, "dAUC": rows[-1]["AUC"] - roc_auc_score(y, P[b][ok]), "low": lo, "high": hi})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


# ---------------------------------------------------------------- scores
def boot_weights(years: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uy = np.unique(years)
    draws = rng.integers(0, len(uy), size=(N_BOOT, len(uy)))
    counts = np.stack([np.bincount(dr, minlength=len(uy)) for dr in draws])
    return counts[:, np.searchsorted(uy, years)]


def wauc(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
        return np.nan
    return roc_auc_score(y, p, sample_weight=w)


def interval(x: np.ndarray) -> tuple[float, float]:
    return tuple(np.nanpercentile(x, [2.5, 97.5]))


def warn_share(p: np.ndarray, rate: float) -> np.ndarray:
    """Warning for the highest ``rate`` of the forecasts; units tied at the cut share the remaining warnings."""
    k = max(1, int(round(rate * len(p))))
    cut = np.sort(p)[::-1][k - 1]
    above, tie = p > cut, p == cut
    out = above.astype(float)
    out[tie] = (k - above.sum()) / tie.sum()
    return out


def matched(p: np.ndarray, y: np.ndarray, rate: float, years: np.ndarray | None = None) -> tuple[float, float]:
    """Detection and false-alarm rate; with ``years`` the warnings are allotted within each year."""
    if years is None:
        warn = warn_share(p, rate)
    else:
        warn = np.zeros(len(p))
        for yy in np.unique(years):
            a = years == yy
            warn[a] = warn_share(p[a], rate)
    return float(warn[y == 1].mean()), float(warn[y == 0].mean())


def relative_value(pod: float, pofd: float, s: float, a: np.ndarray) -> np.ndarray:
    return (np.minimum(a, s) - pofd * a * (1 - s) + pod * s * (1 - a) - s) / (np.minimum(a, s) - s * a)


def score_table(D: pd.DataFrame, P: dict[str, np.ndarray], tag: dict, reference: str = "Pooled") -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = ~np.isnan(P[reference])
    for p in P.values():
        ok &= ~np.isnan(p)
    y, years, idx = D.y[ok].to_numpy(), D.year[ok].to_numpy(), D["index"][ok].to_numpy()
    W = boot_weights(years, np.random.default_rng(SEED))
    masks = {it: idx == it for it in INDICES}
    rows, draws = [], {}
    for name, p in P.items():
        p = p[ok]
        per = {it: roc_auc_score(y[k], p[k]) for it, k in masks.items()}
        bd = np.array([[wauc(p[k], y[k], wt[k]) for k in masks.values()] for wt in W])
        draws[name] = bd
        for j, it in enumerate(INDICES):
            k = masks[it]
            lo, hi = interval(bd[:, j])
            row = {**tag, "forecast": name, "index": it, "units": int(k.sum()), "events": int(y[k].sum()), "AUC": per[it],
                   "AUC_low": lo, "AUC_high": hi}
            row["POD_top20"], row["POFD_top20"] = matched(p[k], y[k], 0.2)
            row["POD_top20_within_year"], row["POFD_top20_within_year"] = matched(p[k], y[k], 0.2, years[k])
            if name != "Persistence":
                row["Brier"] = float(np.mean((p[k] - y[k]) ** 2))
            rows.append(row)
        lo, hi = interval(np.nanmean(bd, axis=1))
        rows.append({**tag, "forecast": name, "index": "MEAN", "units": int(ok.sum()), "events": int(y.sum()),
                     "AUC": float(np.mean(list(per.values()))), "AUC_low": lo, "AUC_high": hi})
    out = pd.DataFrame(rows)
    brier_c = out[out.forecast == "Climatology"].set_index("index").Brier
    out["Brier_skill_vs_climatology"] = [1 - r.Brier / brier_c.get(r["index"], np.nan) if pd.notna(r.get("Brier")) else np.nan
                                         for _, r in out.iterrows()]
    diffs = []
    names = list(P)
    for a in names:
        for b in names:
            if a == b:
                continue
            dd = draws[a] - draws[b]
            point = out[(out.forecast == a)].set_index("index").AUC - out[(out.forecast == b)].set_index("index").AUC
            lo, hi = interval(np.nanmean(dd, axis=1))
            diffs.append({**tag, "forecast": a, "minus": b, "index": "MEAN", "dAUC": point["MEAN"], "low": lo, "high": hi})
            for j, it in enumerate(INDICES):
                lo, hi = interval(dd[:, j])
                diffs.append({**tag, "forecast": a, "minus": b, "index": it, "dAUC": point[it], "low": lo, "high": hi})
    return out, pd.DataFrame(diffs)


def yearly_table(D: pd.DataFrame, P: dict[str, np.ndarray], tag: dict) -> pd.DataFrame:
    ok = ~np.isnan(P["Pooled"])
    rows = []
    for yy in np.unique(D.year[ok]):
        for name, p in P.items():
            aucs = []
            for it in INDICES:
                k = ok & (D.year == yy).to_numpy() & (D["index"] == it).to_numpy()
                yk = D.y[k].to_numpy()
                if 0 < yk.sum() < k.sum():
                    aucs.append(roc_auc_score(yk, p[k]))
            rows.append({**tag, "year": int(yy), "forecast": name, "mean_AUC": float(np.mean(aucs)), "indices_scored": len(aucs)})
    return pd.DataFrame(rows)


def reliability_table(D: pd.DataFrame, P: dict[str, np.ndarray], tag: dict, bins: int = 8) -> pd.DataFrame:
    rows = []
    ok = ~np.isnan(P["Pooled"])
    for name in ("Climatology", "Pooled"):
        p, y = P[name][ok], D.y[ok].to_numpy()
        order = np.argsort(p, kind="stable")
        for b, part in enumerate(np.array_split(order, bins), 1):
            rows.append({**tag, "forecast": name, "bin": b, "units": len(part), "mean_predicted": float(p[part].mean()),
                         "observed_frequency": float(y[part].mean())})
    return pd.DataFrame(rows)


def value_table(D: pd.DataFrame, P: dict[str, np.ndarray], tag: dict) -> pd.DataFrame:
    """Relative economic value by cost-loss ratio, averaged over the indices.

    Probability forecasts act when the probability exceeds the cost-loss ratio; persistence
    is represented by its best threshold at each ratio."""
    alphas = np.arange(0.01, 0.61, 0.01)
    ok = ~np.isnan(P["Pooled"])
    rows = []
    for it in INDICES:
        k = ok & (D["index"] == it).to_numpy()
        y = D.y[k].to_numpy()
        s = y.mean()
        for name in ("Climatology", "Pooled", "Persistence"):
            p = P[name][k]
            if name == "Persistence":
                best = np.full(len(alphas), -np.inf)
                for thr in np.unique(p):
                    warn = p >= thr
                    best = np.maximum(best, relative_value(warn[y == 1].mean(), warn[y == 0].mean(), s, alphas))
                v = best
            else:
                v = np.array([relative_value((p > a)[y == 1].mean(), (p > a)[y == 0].mean(), s, a) for a in alphas])
            rows += [{**tag, "index": it, "forecast": name, "cost_loss_ratio": round(float(a), 2), "value": float(x)}
                     for a, x in zip(alphas, v)]
    return pd.DataFrame(rows)


def coefficient_table(D: pd.DataFrame, tag: dict) -> pd.DataFrame:
    """Standardised coefficients of the pooled model fitted to all years (for interpretation only)."""
    for it in INDICES:                      # climatology of every year from the other years
        k = (D["index"] == it).to_numpy()
        d = D[k].reset_index(drop=True)
        clim = np.full(len(d), np.nan)
        for yy in np.unique(d.year):
            a = (d.year == yy).to_numpy()
            clim[a] = climatology(d, ~a, a)
        D.loc[k, "clim"] = clim
    cols = feature_cols(CORE) + [f"is_{it}" for it in INDICES]
    k = D.clim.notna().to_numpy()
    x = D.loc[k, cols]
    x = x.fillna(x.median()).fillna(0.0)
    sc = StandardScaler().fit(x)
    m = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(x), D.y[k])
    return pd.DataFrame({**tag, "feature": cols, "coefficient": m.coef_[0]})


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tables, results = args.out / "tables", args.out / "results"
    tables.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    raw, flow = load_panel(args.project)
    tau = thresholds(raw)
    weather = load_weather(args.project)
    LABEL["LCA"] = "Apple leaf-curling aphid (per 640 leaves)"
    pd.DataFrame([{"index": it, "label": LABEL[it], "threshold": tau[it]} for it in INDICES + ["LCA"]]).to_csv(tables / "A00_thresholds.csv", index=False)

    # descriptive tables on the raw scale
    desc = []
    for it in INDICES:
        s = raw.dropna(subset=[it])
        peak = s.groupby(["sigunguNm", "year"])[it].max()
        desc.append({"index": it, "label": LABEL[it], "rows_all": flow["rows_six_indices"][it], "rows_numeric": flow["rows_numeric"][it],
                     "counties": s.sigunguNm.nunique(), "years": s.year.nunique(), "county_years": len(peak),
                     "share_positive": float((s[it] > 0).mean()), "threshold": tau[it],
                     "share_at_or_above_threshold": float((s[it] >= tau[it]).mean()),
                     "county_years_exceeding": int((peak >= tau[it]).sum())})
    pd.DataFrame(desc).to_csv(tables / "A01_data_summary.csv", index=False)
    season = []
    for it in INDICES:
        s = raw.dropna(subset=[it]).copy()
        s["half_month"] = (s.doy // 15) * 15
        g = s.groupby("half_month")[it]
        season += [{"index": it, "doy": int(d), "surveys": int(n), "mean": float(m), "share_at_or_above_threshold": float(e)}
                   for d, n, m, e in zip(g.size().index, g.size(), g.mean(), g.apply(lambda x: (x >= tau[it]).mean()))]
    pd.DataFrame(season).to_csv(tables / "A02_seasonal_pattern.csv", index=False)
    surveys = raw[["sigunguNm", "sidoCode", "year", "date"] + INDICES].copy()
    for it in INDICES:
        surveys[it] = surveys[it] / tau[it]
    surveys.to_csv(results / "surveys_relative_to_threshold.csv", index=False)

    main_variants = {
        "Separate": {"groups": CORE, "scope": "separate"},
        "Pooled": {"groups": CORE},
        "Pooled (gradient boosting)": {"groups": CORE, "kind": "gb"},
        "Transfer (index unseen)": {"groups": CORE, "scope": "transfer", "dummies": False},
    }
    ablation = {
        "State": {"groups": ["state"]},
        "State + season": {"groups": ["state", "season"]},
        "State + season + history": {"groups": ["state", "season", "history"]},
        "Pooled": {"groups": CORE},
        "Pooled + weather": {"groups": CORE + ["weather"]},
        "Pooled without state": {"groups": ["season", "history", "neighbours"]},
        "Pooled without season": {"groups": ["state", "history", "neighbours"]},
        "Pooled without history": {"groups": ["state", "season", "neighbours"]},
        "Pooled without neighbours": {"groups": ["state", "season", "history"]},
    }
    no_record = ["state", "calendar", "neighbours"]     # predictors that need no earlier record of the index
    new_index = {
        "Pooled": {"groups": CORE},
        "Transfer (index unseen)": main_variants["Transfer (index unseen)"],
        "Pooled (no record of the index)": {"groups": no_record},
        "Transfer (no record of the index)": {"groups": no_record, "scope": "transfer", "dummies": False},
    }
    S, Dd, Y, Rl, V, C, AB, ABd, SE, SEd, NI, NId, HO, HOd = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for h in (1, 2):
        w = raw.copy()
        for it in INDICES:
            w[it] = w[it] / tau[it]
        D = make_dataset(w, h, "first", weather)
        tag = {"horizon": h}
        w["LCA"] = raw["LCA"] / tau["LCA"]
        s, dd = held_out_index(D, w, "LCA", h, tag)
        HO.append(s); HOd.append(dd)
        print(s.round(3).to_string(), flush=True); print(dd.round(3).to_string(), flush=True)
        P = rolling(D, main_variants)
        s, dd = score_table(D, P, tag)
        S.append(s); Dd.append(dd); Y.append(yearly_table(D, P, tag)); Rl.append(reliability_table(D, P, tag))
        V.append(value_table(D, P, tag)); C.append(coefficient_table(D.copy(), tag))
        out = D[["index", "county", "prov", "year", "date", "doy_t", "gap", "cur", "y"]].copy()
        for k, p in P.items():
            out[k] = p
        out[~out["Pooled"].isna()].to_csv(results / f"predictions_h{h}.csv", index=False)
        m = s[s["index"] == "MEAN"].set_index("forecast").AUC
        print(f"h={h}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)

        s, dd = score_table(D, rolling(D, new_index), tag)
        NI.append(s); NId.append(dd)
        print("  new index: " + ", ".join(f"{k} {v:.3f}" for k, v in s[s['index'] == 'MEAN'].set_index('forecast').AUC.items()), flush=True)
        if h == 2:      # units whose intermediate survey was still below the threshold
            keep = (D.mid == 0).to_numpy()
            Pm = {k: np.where(keep, P[k], np.nan) for k in ("Persistence", "Climatology", "Separate", "Pooled")}
            s, dd = score_table(D, Pm, {**tag, "setting": "below threshold at the intermediate survey"})
            SE.append(s); SEd.append(dd)
            print("  intermediate below: " + ", ".join(f"{k} {v:.3f}" for k, v in s[s['index'] == 'MEAN'].set_index('forecast').AUC.items()), flush=True)

        Pa = rolling(D, ablation)
        s, dd = score_table(D, Pa, tag)
        AB.append(s); ABd.append(dd)
        print(f"  ablation: " + ", ".join(f"{k} {v:.3f}" for k, v in s[s['index'] == 'MEAN'].set_index('forecast').AUC.items()), flush=True)

        for mult in (0.5, 2.0, 3.0):
            w2 = raw.copy()
            for it in INDICES:
                w2[it] = w2[it] / (tau[it] * mult)
            D2 = make_dataset(w2, h)
            s, dd = score_table(D2, rolling(D2, {"Separate": main_variants["Separate"], "Pooled": main_variants["Pooled"]}),
                                {**tag, "setting": f"threshold x {mult}"})
            SE.append(s); SEd.append(dd)
        D3 = make_dataset(w, h, "current")
        s, dd = score_table(D3, rolling(D3, {"Separate": main_variants["Separate"], "Pooled": main_variants["Pooled"]}),
                            {**tag, "setting": "currently below threshold"})
        SE.append(s); SEd.append(dd)
        for st, g in pd.concat(SE[-4:]).groupby("setting"):
            print(f"  {st}: " + ", ".join(f"{k} {v:.3f}" for k, v in g[g['index'] == 'MEAN'].set_index('forecast').AUC.items()), flush=True)

    pd.concat(S).to_csv(tables / "A03_scores.csv", index=False)
    pd.concat(Dd).to_csv(tables / "A04_score_differences.csv", index=False)
    pd.concat(Y).to_csv(tables / "A05_scores_by_year.csv", index=False)
    pd.concat(Rl).to_csv(tables / "A06_reliability.csv", index=False)
    pd.concat(V).to_csv(tables / "A07_relative_value.csv", index=False)
    pd.concat(C).to_csv(tables / "A08_coefficients.csv", index=False)
    pd.concat(AB).to_csv(tables / "A09_ablation_scores.csv", index=False)
    pd.concat(ABd).to_csv(tables / "A10_ablation_differences.csv", index=False)
    pd.concat(SE).to_csv(tables / "A11_sensitivity_scores.csv", index=False)
    pd.concat(SEd).to_csv(tables / "A12_sensitivity_differences.csv", index=False)
    pd.concat(NI).to_csv(tables / "A13_new_index_scores.csv", index=False)
    pd.concat(NId).to_csv(tables / "A14_new_index_differences.csv", index=False)
    pd.concat(HO).to_csv(tables / "A15_held_out_index_scores.csv", index=False)
    pd.concat(HOd).to_csv(tables / "A16_held_out_index_differences.csv", index=False)


if __name__ == "__main__":
    main()
