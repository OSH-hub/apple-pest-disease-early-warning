"""Usage: python src/check_scheduled_date.py --project <project root> --out <output dir>"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import pipeline as pl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    raw, _ = pl.load_panel(args.project)
    tau = pl.thresholds(raw)
    variants = {"Separate": {"groups": pl.CORE, "scope": "separate"}, "Pooled": {"groups": pl.CORE}}
    S, Dd = [], []
    for h in (1, 2):
        w = raw.copy()
        for it in pl.INDICES:
            w[it] = w[it] / tau[it]
        D0 = pl.make_dataset(w, h, "first")
        for mode in ("actual", "scheduled"):
            D = D0.copy()
            if mode == "scheduled":
                D["doy_t"] = D["doy"] + 15 * h
                D["gap"] = 15 * h
            s, dd = pl.score_table(D, pl.rolling(D, variants), {"horizon": h, "target_date": mode})
            S.append(s); Dd.append(dd)
            m = s[s["index"] == "MEAN"].set_index("forecast").AUC
            print(f"h={h} {mode}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
    pd.concat(S).to_csv(tables / "A17_scheduled_date_scores.csv", index=False)
    pd.concat(Dd).to_csv(tables / "A18_scheduled_date_differences.csv", index=False)


if __name__ == "__main__":
    main()
