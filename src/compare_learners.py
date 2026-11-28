"""Usage: python src/compare_learners.py --project <project root> --out <output dir>"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn

import pipeline as pl

SEEDS = (0, 1, 2, 3, 4)
EPOCHS, BATCH, HIDDEN = 60, 128, 16
STATIC = pl.feature_cols(["season", "history", "neighbours"])


def sequences(D: pd.DataFrame, w: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    series = {}
    for it in pl.INDICES:
        for (county, year), s in w.dropna(subset=[it]).groupby(["sigunguNm", "year"]):
            series[(it, county, year)] = (s.date.to_numpy(), np.log1p(s[it].to_numpy()), s.doy.to_numpy())
    n = int(D.n_surveys.max())
    X, L = np.zeros((len(D), n, 3), dtype=np.float32), np.zeros(len(D), dtype=np.int64)
    for i, (it, county, year, day) in enumerate(zip(D["index"], D.county, D.year, D.date)):
        date, val, doy = series[(it, county, year)]
        k = int((date <= np.datetime64(day)).sum())
        gap = np.diff(doy[:k], prepend=doy[0]) / 15.0
        X[i, :k] = np.stack([val[:k], doy[:k] / 365.0, gap], axis=1)
        L[i] = k
    return X, L


class Net(nn.Module):
    def __init__(self, n_static: int):
        super().__init__()
        self.lstm = nn.LSTM(3, HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN + n_static, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, seq, length, static):
        out, _ = self.lstm(seq)
        last = out[torch.arange(len(seq)), length - 1]
        return self.head(torch.cat([last, static], dim=1)).squeeze(1)


def lstm_predict(seq, length, static, y, tr, te) -> np.ndarray:
    st = torch.tensor(static, dtype=torch.float32)
    sq, ln, yt = torch.tensor(seq), torch.tensor(length), torch.tensor(y, dtype=torch.float32)
    tr_i, te_i = np.flatnonzero(tr), np.flatnonzero(te)
    p = np.zeros(len(te_i))
    for seed in SEEDS:
        torch.manual_seed(pl.SEED + seed)
        rng = np.random.default_rng(pl.SEED + seed)
        net = Net(st.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss = nn.BCEWithLogitsLoss()
        for _ in range(EPOCHS):
            order = rng.permutation(tr_i)
            for b in range(0, len(order), BATCH):
                k = order[b:b + BATCH]
                opt.zero_grad()
                loss(net(sq[k], ln[k], st[k]), yt[k]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            p += torch.sigmoid(net(sq[te_i], ln[te_i], st[te_i])).numpy()
    return p / len(SEEDS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    raw, _ = pl.load_panel(args.project)
    tau = pl.thresholds(raw)
    dummies = [f"is_{it}" for it in pl.INDICES]
    core = pl.feature_cols(pl.CORE) + dummies
    S, Dd = [], []
    for h in (1, 2):
        w = raw.copy()
        for it in pl.INDICES:
            w[it] = w[it] / tau[it]
        D = pl.make_dataset(w, h, "first")
        seq, length = sequences(D, w)
        names = ["Persistence", "Climatology", "Pooled", "Gradient boosting", "Random forest", "LSTM"]
        P = {k: np.full(len(D), np.nan) for k in names}
        for y in np.sort(D.year.unique())[pl.MIN_HISTORY:]:
            pl.fill_climatology(D, y)
            valid = D.clim.notna().to_numpy()
            tr, te = valid & (D.year < y).to_numpy(), valid & (D.year == y).to_numpy()
            P["Persistence"][te], P["Climatology"][te] = D.cur[te], D.clim[te]
            P["Pooled"][te] = pl.fit_predict(D, core, tr, te)
            P["Gradient boosting"][te] = pl.fit_predict(D, core, tr, te, "gb")
            x = D[core]
            x = x.fillna(x[tr].median()).fillna(0.0)
            rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=pl.SEED)
            P["Random forest"][te] = rf.fit(x[tr], D.y[tr]).predict_proba(x[te])[:, 1]
            xs = D[STATIC]
            xs = xs.fillna(xs[tr].median()).fillna(0.0)
            xs = (xs - xs[tr].mean()) / xs[tr].std().replace(0, 1.0)
            static = np.column_stack([xs.to_numpy(), D[dummies].to_numpy()])
            P["LSTM"][te] = lstm_predict(seq, length, static, D.y.to_numpy(), tr, te)
            print(f"h={h} year {y} done", flush=True)
        s, dd = pl.score_table(D, P, {"horizon": h})
        S.append(s); Dd.append(dd)
        m = s[s["index"] == "MEAN"].set_index("forecast")
        print(f"h={h}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.AUC.items()), f"units {int(m.units.iloc[0])}", flush=True)
        pd.concat(S).to_csv(tables / "A19_learner_scores.csv", index=False)
        pd.concat(Dd).to_csv(tables / "A20_learner_differences.csv", index=False)


if __name__ == "__main__":
    main()
