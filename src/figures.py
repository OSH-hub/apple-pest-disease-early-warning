"""Draw Figures 1-9 of the article from the tables written by pipeline.py.

Usage: python src/figures.py --project <project root> --out <output dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INDICES = ["MB", "AB", "OFM_trap", "OFM_shoot", "OFM_fruit", "ERM"]
SHORT = {"MB": "Marssonina blotch", "AB": "Alternaria blotch", "OFM_trap": "OFM trap catch", "OFM_shoot": "OFM shoot damage",
         "OFM_fruit": "OFM fruit damage", "ERM": "European red mite"}
COL = {"Persistence": "#8c8c8c", "Climatology": "#d28e2a", "Separate": "#5b8fbd", "Pooled": "#b03a48",
       "Transfer (index unseen)": "#3f8f6b", "Pooled (gradient boosting)": "#7a5aa6"}
NAME = {"Persistence": "Persistence", "Climatology": "Climatology", "Separate": "Separate models", "Pooled": "Unified model",
        "Transfer (index unseen)": "Unified model, index unseen", "Pooled (gradient boosting)": "Unified, gradient boosting"}
INK, MUTED = "#222222", "#666666"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "axes.titlesize": 8.5, "axes.titleweight": "bold", "legend.frameon": False,
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7, "pdf.fonttype": 42})
MONTHS = [(91, "Apr"), (121, "May"), (152, "Jun"), (182, "Jul"), (213, "Aug"), (244, "Sep"), (274, "Oct")]


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf", "tif"):
        fig.savefig(out / f"{name}.{ext}", dpi=300 if ext != "pdf" else None, bbox_inches="tight",
                    **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}))
    plt.close(fig)
    from PIL import Image
    with Image.open(out / f"{name}.tif") as im:          # journals ask for RGB without an alpha channel
        rgba = im.convert("RGBA")
        rgb = Image.new("RGB", im.size, "white")
        rgb.paste(rgba, mask=rgba.split()[3])
    rgb.save(out / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))


def month_axis(ax) -> None:
    ax.set_xticks([d for d, _ in MONTHS])
    ax.set_xticklabels([m for _, m in MONTHS], fontsize=6.8)


def fig1_workflow(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=color, ec=INK, lw=0.7))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="center", fontsize=7.1, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + (h - 4.2) / 2, body, ha="center", va="center", fontsize=6.1, color=INK, linespacing=1.35)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.8, color=INK))

    box(1, 30, 21, 18, "Routine surveillance", "25 apple counties\nsurveys about every 15 days\n2015–2025\n6 pest and disease indices", "#e8eef5")
    box(27, 30, 21, 18, "At-risk units", "index divided by its threshold\ncounty still below threshold\ntarget: at or above threshold\n15 or 30 days later", "#e8eef5")
    box(53, 30, 21, 18, "Predictors known today", "current state of the county\nseason and seasonal hazard\ncounty history\nstate of the other counties", "#f5ece0")
    box(79, 30, 20, 18, "Unified model", "one logistic model\nfitted to all six indices\nwith an index term", "#f3dfe1")
    box(27, 3, 21, 18, "Reference forecasts", "persistence, climatology\nseparate model per index\ngradient boosting", "#eeeeee")
    box(53, 3, 21, 18, "Prospective test", "each year 2018–2025\npredicted from earlier years\nyear-cluster bootstrap\nintervals", "#e3f0ea")
    box(79, 3, 20, 18, "Warning", "probability of\nexceedance per county\nand index, 15 and 30 days\nahead", "#f3dfe1")
    for a, b in (((22.3, 39), (26.7, 39)), ((48.3, 39), (52.7, 39)), ((74.3, 39), (78.7, 39)), ((89, 29.7), (89, 21.3)),
                 ((48.3, 12), (52.7, 12)), ((78.7, 12), (74.3, 12)), ((37.5, 29.7), (37.5, 21.3))):
        arrow(a, b)
    save(fig, out, "figure01_workflow")


def fig2_study_area(root: Path, out: Path, tables: Path) -> None:
    cells = pd.read_csv(root / "data/agri_weather/valid_cells.csv")
    xy = cells.groupby("sigunguNm")[["lat", "lon"]].mean()
    sv = pd.read_csv(tables.parent / "results" / "surveys_relative_to_threshold.csv")
    peak = sv.groupby(["sigunguNm", "year"])[INDICES].max()
    freq = (peak >= 1).groupby("sigunguNm").mean()
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.6), sharex=True, sharey=True)
    for ax, it in zip(axes.ravel(), INDICES):
        j = xy.join(freq[it]).dropna()
        sc = ax.scatter(j.lon, j.lat, c=j[it], cmap="YlOrRd", vmin=0, vmax=1, s=42, edgecolor=INK, linewidth=0.5)
        ax.set_title(SHORT[it]); ax.set_aspect(1.25)
        ax.grid(color="#dddddd", lw=0.4)
    for ax in axes[1]:
        ax.set_xlabel("Longitude (°E)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude (°N)")
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("Share of surveyed years in which the index reached its threshold")
    save(fig, out, "figure02_study_area")


def fig3_seasonal(tables: Path, out: Path) -> None:
    s = pd.read_csv(tables / "A02_seasonal_pattern.csv")
    thr = pd.read_csv(tables / "A00_thresholds.csv").set_index("index").threshold
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.2), sharex=True)
    for ax, it in zip(axes.ravel(), INDICES):
        g = s[(s["index"] == it) & (s.surveys >= 20)]
        ax.bar(g.doy + 7, g.share_at_or_above_threshold, width=12, color="#c9d6e3", edgecolor="none")
        ax.set_ylim(0, max(0.5, g.share_at_or_above_threshold.max() * 1.15))
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
        ax2.plot(g.doy + 7, g["mean"], color="#b03a48", lw=1.3, marker="o", ms=2.5)
        ax2.axhline(thr[it], color=MUTED, lw=0.6, ls="--")
        ax2.tick_params(colors="#b03a48")
        ax.set_title(SHORT[it]); month_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of surveys at or\nabove threshold (bars)")
    fig.text(0.995, 0.5, "Mean index value (line); dashed: threshold", rotation=90, va="center", ha="right", fontsize=8, color="#b03a48")
    fig.tight_layout(rect=(0, 0, 0.975, 1))
    save(fig, out, "figure03_seasonal_pattern")


def fig4_examples(results: Path, out: Path) -> None:
    sv = pd.read_csv(results / "surveys_relative_to_threshold.csv", parse_dates=["date"])
    pr = pd.read_csv(results / "predictions_h1.csv", parse_dates=["date"])
    sv["doy"] = sv.date.dt.dayofyear
    picks = []
    for it in ("AB", "OFM_trap", "MB"):
        p = pr[pr["index"] == it]
        cut = p["Pooled"].quantile(0.8)
        ev = p[p.y == 1]
        hit = ev[ev["Pooled"] >= cut].sort_values("Pooled", ascending=False)
        miss = ev[ev["Pooled"] < cut].sort_values("Pooled")
        if len(hit):
            picks.append((it, hit.iloc[0], "warned"))
        if len(miss):
            picks.append((it, miss.iloc[0], "not warned"))
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3))
    order = [0, 2, 4, 1, 3, 5]
    for ax, k in zip(axes.ravel(), order):
        if k >= len(picks):
            ax.axis("off"); continue
        it, row, kind = picks[k]
        s = sv[(sv.sigunguNm == row.county) & (sv.year == row.year)].dropna(subset=[it])
        p = pr[(pr["index"] == it) & (pr.county == row.county) & (pr.year == row.year)]
        ax.plot(s.doy, s[it], color=INK, lw=1.1, marker="o", ms=3)
        ax.axhline(1, color=MUTED, ls="--", lw=0.6)
        ax.set_ylim(0, max(2.5, s[it].max() * 1.1))
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
        ax2.bar(p.date.dt.dayofyear, p["Pooled"], width=6, color="#b03a48", alpha=0.55)
        ax2.bar(p.date.dt.dayofyear + 6, p["Climatology"], width=6, color="#d28e2a", alpha=0.55)
        ax2.set_ylim(0, max(0.8, float(p[["Pooled", "Climatology"]].max().max()) * 1.08)); ax2.tick_params(colors="#b03a48")
        ax.set_title(f"{SHORT[it]}, {int(row.year)}: {kind}", fontsize=7.8)
        month_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Index / threshold (line)")
    fig.text(0.995, 0.5, "Probability of exceedance at the next survey (bars)",
             rotation=90, va="center", ha="right", fontsize=7.2, color="#b03a48")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([], [], color=INK, marker="o", ms=3, lw=1.1, label="Index relative to its threshold"),
                        Patch(color="#b03a48", alpha=0.55, label="Unified model"), Patch(color="#d28e2a", alpha=0.55, label="Climatology")],
               loc="upper center", ncol=3, fontsize=7.2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 0.975, 0.97))
    save(fig, out, "figure05_example_seasons")


def fig8_ablation(tables: Path, out: Path) -> None:
    d = pd.read_csv(tables / "A10_ablation_differences.csv")
    s = pd.read_csv(tables / "A09_ablation_scores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    build = ["State", "State + season", "State + season + history", "Pooled", "Pooled + weather"]
    lab = ["Current\nstate", "+ season", "+ county\nhistory", "+ other\ncounties", "+ weather"]
    for h, c, off in ((1, "#b03a48", -0.17), (2, "#5b8fbd", 0.17)):
        g = s[(s.horizon == h) & (s["index"] == "MEAN")].set_index("forecast").loc[build]
        axes[0].bar(np.arange(len(build)) + off, g.AUC, width=0.34, color=c, label=f"{15 * h} days ahead")
        axes[0].errorbar(np.arange(len(build)) + off, g.AUC, yerr=[g.AUC - g.AUC_low, g.AUC_high - g.AUC], fmt="none", ecolor=INK, lw=0.7)
    axes[0].set_xticks(range(len(build))); axes[0].set_xticklabels(lab, fontsize=6.8)
    axes[0].set_ylim(0.6, 0.90); axes[0].set_ylabel("Mean AUC of the indices"); axes[0].set_title("(A) Predictor groups added in turn")
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    drop = ["Pooled without state", "Pooled without season", "Pooled without history", "Pooled without neighbours"]
    lab2 = ["Current state", "Season", "County history", "Other counties"]
    for h, c, off in ((1, "#b03a48", -0.17), (2, "#5b8fbd", 0.17)):
        g = d[(d.horizon == h) & (d["index"] == "MEAN") & (d.forecast == "Pooled") & d.minus.isin(drop)].set_index("minus").loc[drop]
        axes[1].barh(np.arange(len(drop))[::-1] + off, g.dAUC, height=0.34, color=c)
        axes[1].errorbar(g.dAUC, np.arange(len(drop))[::-1] + off, xerr=[g.dAUC - g.low, g.high - g.dAUC], fmt="none", ecolor=INK, lw=0.7)
    axes[1].axvline(0, color=INK, lw=0.6)
    axes[1].set_yticks(np.arange(len(drop))[::-1]); axes[1].set_yticklabels(lab2)
    axes[1].set_xlabel("Loss of mean AUC when the group is removed"); axes[1].set_title("(B) Predictor groups removed one at a time")
    fig.tight_layout()
    save(fig, out, "figure07_predictor_groups")


def fig9_transfer(tables: Path, out: Path) -> None:
    s = pd.read_csv(tables / "A13_new_index_scores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True)
    fs = [("Persistence", COL["Persistence"], "Persistence"),
          ("Transfer (no record of the index)", "#a9d3bf", "Index unseen, no record"),
          ("Transfer (index unseen)", COL["Transfer (index unseen)"], "Index unseen, seasonal record"),
          ("Pooled", COL["Pooled"], "Unified model")]
    for ax, h in zip(axes, (1, 2)):
        for j, (f, col, lab) in enumerate(fs):
            g = s[(s.horizon == h) & (s.forecast == f)].set_index("index").loc[INDICES]
            x = np.arange(6) + (j - 1.5) * 0.2
            ax.bar(x, g.AUC, width=0.2, color=col, label=lab)
            ax.errorbar(x, g.AUC, yerr=[g.AUC - g.AUC_low, g.AUC_high - g.AUC], fmt="none", ecolor=INK, lw=0.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels(["Marssonina\nblotch", "Alternaria\nblotch", "OFM\ntrap", "OFM\nshoot", "OFM\nfruit", "Red\nmite"], fontsize=6.2, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylim(0.4, 1.0); ax.set_title(f"{15 * h} days ahead"); ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    axes[0].set_ylabel("Area under the ROC curve")
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.03, 1.1), ncol=4, fontsize=7)
    save(fig, out, "figure06_index_unseen")


def fig13_sensitivity(tables: Path, out: Path) -> None:
    d = pd.read_csv(tables / "A12_sensitivity_differences.csv")
    base = pd.read_csv(tables / "A04_score_differences.csv").assign(setting="main analysis")
    d = pd.concat([base, d])
    settings = ["main analysis", "threshold x 0.5", "threshold x 2.0", "threshold x 3.0", "currently below threshold"]
    lab = ["Main analysis", "Threshold halved", "Threshold doubled", "Threshold tripled", "At risk: currently\nbelow threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), sharey=True)
    for ax, h in zip(axes, (1, 2)):
        for j, (b, col) in enumerate((("Persistence", COL["Persistence"]), ("Climatology", COL["Climatology"]), ("Separate", COL["Separate"]))):
            g = d[(d.horizon == h) & (d["index"] == "MEAN") & (d.forecast == "Pooled") & (d.minus == b)].set_index("setting").loc[settings]
            y = np.arange(len(settings))[::-1] + (1 - j) * 0.22
            ax.errorbar(g.dAUC, y, xerr=[g.dAUC - g.low, g.high - g.dAUC], fmt="o", ms=3.6, lw=0.9, color=col,
                        label=f"minus {NAME[b].lower()}")
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks(np.arange(len(settings))[::-1]); ax.set_yticklabels(lab)
        ax.set_xlabel("Gain in mean AUC of the unified model"); ax.set_title(f"{15 * h} days ahead")
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.03, 1.1), ncol=3, fontsize=7)
    save(fig, out, "figure09_sensitivity")


def fig_skill(tables: Path, out: Path) -> None:
    """Figure 4: AUC by index (upper row) and mean AUC by test year (lower row)."""
    s = pd.read_csv(tables / "A03_scores.csv")
    y = pd.read_csv(tables / "A05_scores_by_year.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.0), sharey="row", gridspec_kw={"height_ratios": [1.35, 1]})
    rows = INDICES + ["MEAN"]
    labels = [SHORT[i] for i in INDICES] + ["Mean of six indices"]
    fs = ("Persistence", "Climatology", "Separate", "Pooled")
    for k, h in enumerate((1, 2)):
        ax = axes[0, k]
        for j, f in enumerate(fs):
            g = s[(s.horizon == h) & (s.forecast == f)].set_index("index").loc[rows]
            yy = np.arange(len(rows))[::-1] + (1.5 - j) * 0.18
            ax.errorbar(g.AUC, yy, xerr=[g.AUC - g.AUC_low, g.AUC_high - g.AUC], fmt="o", ms=3.6, lw=0.9, color=COL[f], label=NAME[f], capsize=0)
        ax.set_yticks(np.arange(len(rows))[::-1]); ax.set_yticklabels(labels)
        ax.set_xlabel("Area under the ROC curve"); ax.set_xlim(0.4, 1.0)
        ax.set_title(f"({'AB'[k]}) {15 * h} days ahead, by index"); ax.grid(axis="x", color="#e5e5e5", lw=0.5)
        ax = axes[1, k]
        for f in fs:
            g = y[(y.horizon == h) & (y.forecast == f)]
            ax.plot(g.year, g.mean_AUC, color=COL[f], marker="o", ms=3, lw=1.4 if f == "Pooled" else 0.9)
        ax.set_ylim(0.45, 0.95); ax.set_xlabel("Test year"); ax.grid(axis="y", color="#e5e5e5", lw=0.5)
        ax.set_title(f"({'CD'[k]}) {15 * h} days ahead, by test year")
    axes[1, 0].set_ylabel("Mean AUC of the indices")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=7.5)
    save(fig, out, "figure04_skill")


def fig_reliability_value(tables: Path, out: Path) -> None:
    """Figure 8: reliability (upper row) and relative economic value (lower row)."""
    r = pd.read_csv(tables / "A06_reliability.csv")
    v = pd.read_csv(tables / "A07_relative_value.csv")
    v["value"] = v.value.clip(lower=0)
    m = v.groupby(["horizon", "forecast", "cost_loss_ratio"]).value.mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.6))
    for k, h in enumerate((1, 2)):
        ax = axes[0, k]
        for f in ("Climatology", "Pooled"):
            g = r[(r.horizon == h) & (r.forecast == f)]
            ax.plot(g.mean_predicted, g.observed_frequency, marker="o", ms=3.5, color=COL[f], lw=1.1, label=NAME[f])
        ax.plot([0, 0.5], [0, 0.5], color="#bbbbbb", lw=0.6)
        ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.5); ax.set_aspect(1)
        ax.set_xlabel("Predicted probability"); ax.set_title(f"({'AB'[k]}) Reliability, {15 * h} days ahead")
        ax = axes[1, k]
        for f in ("Persistence", "Climatology", "Pooled"):
            g = m[(m.horizon == h) & (m.forecast == f)]
            ax.plot(g.cost_loss_ratio, g.value, color=COL[f], lw=1.4 if f == "Pooled" else 0.9, label=NAME[f])
        ax.set_xlabel("Cost\u2013loss ratio"); ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.5)
        ax.set_title(f"({'CD'[k]}) Relative value, {15 * h} days ahead")
    axes[0, 0].set_ylabel("Observed frequency"); axes[0, 0].legend(loc="upper left", fontsize=7.2)
    axes[1, 0].set_ylabel("Relative economic value\n(mean of the indices)"); axes[1, 1].legend(loc="upper right", fontsize=7.2)
    fig.tight_layout()
    save(fig, out, "figure08_reliability_value")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tables, results, figs = a.out / "tables", a.out / "results", a.out / "figures_final"
    figs.mkdir(exist_ok=True)
    fig1_workflow(figs); fig2_study_area(a.project, figs, tables); fig3_seasonal(tables, figs); fig_skill(tables, figs)
    fig4_examples(results, figs); fig9_transfer(tables, figs); fig8_ablation(tables, figs); fig_reliability_value(tables, figs)
    fig13_sensitivity(tables, figs)
    print("figures written to", figs)


if __name__ == "__main__":
    main()
