# Unified early warning of threshold exceedance for apple pests and diseases

Code, aggregate tables and figures for a unified early-warning model that predicts, from the
current state of routine surveillance, whether a pest or disease index that is still below its
threshold will exceed it 15 or 30 days later (apple, county surveillance, Korea, 2015-2025).

Every number and figure of the article is produced by the two scripts in `src/` and stored in
`tables/` and `figures/`.

## Contents

| Path | Purpose |
|---|---|
| `src/pipeline.py` | Data preparation, thresholds, at-risk units, predictors, models, reference forecasts, rolling-origin evaluation, year-cluster bootstrap; writes `tables/A00`-`A16` |
| `src/figures.py` | Figures 1-13 of the article |
| `tables/` | Released aggregate tables |
| `figures/` | Released figures; file names carry the figure numbers of the article |

## Tables and the article

| File | Content | Article |
|---|---|---|
| `A00_thresholds.csv` | Threshold of each index | Section 2.2, Table 1 |
| `A01_data_summary.csv` | Records, counties, county-years, share at or above threshold | Table 1 |
| `A02_seasonal_pattern.csv` | Seasonal course in 15-day periods | Figure 3 |
| `A03_scores.csv` | AUC, detection and false-detection rates, Brier score for every forecast and index | Tables 3 and 6, Figure 4 |
| `A04_score_differences.csv` | Paired differences in AUC with bootstrap intervals | Table 3 |
| `A05_scores_by_year.csv` | Mean AUC in each test year | Figure 6 |
| `A06_reliability.csv` | Reliability in eight equal-count bins | Figure 11 |
| `A07_relative_value.csv` | Relative economic value by cost-loss ratio | Figure 12 |
| `A08_coefficients.csv` | Standardized coefficients of the model fitted to all years | Figure 10 |
| `A09_ablation_scores.csv`, `A10_ablation_differences.csv` | Predictor groups added in turn and removed one at a time | Table 5, Figure 9 |
| `A11_sensitivity_scores.csv`, `A12_sensitivity_differences.csv` | Other thresholds, at-risk definition and 30-day target | Section 3.5, Figure 13 |
| `A13_new_index_scores.csv`, `A14_new_index_differences.csv` | Each index predicted by a model fitted to the other five | Table 4, Figure 8 |
| `A15_held_out_index_scores.csv`, `A16_held_out_index_differences.csv` | Index that took no part in model development | Table 4 |

Names used in the tables:

| Table | Article |
|---|---|
| `Pooled` | Unified model |
| `Separate` | Separate models |
| `Pooled (gradient boosting)` | Gradient boosting, all indices |
| `Transfer (index unseen)` | Index unseen, seasonal record of the index |
| `Transfer (no record of the index)` | Index unseen, no record of the index |
| `neighbours` | State of the other counties |
| `horizon` 1, 2 | 15 and 30 days ahead |
| `MB`, `AB` | Marssonina blotch, Alternaria blotch (damaged leaves) |
| `OFM_trap`, `OFM_shoot`, `OFM_fruit` | Oriental fruit moth: trap catch, damaged shoots, damaged fruit |
| `ERM` | European red mite |
| `LCA` | Apple leaf-curling aphid |

## Requirements

Python 3.11 with the pinned packages:

```
pip install -r requirements.txt
```

The analysis was run with Python 3.11.9 on Windows 11 (64-bit). The random seed is 20261126.

## Reproducing the tables and figures

The inputs are public records of the Rural Development Administration of Korea (National Crop
Pest Management System, https://ncpms.rda.go.kr). They are not redistributed here. Place them
under a project directory as

```
data/ncpms_api/apple_surveillance_sigungu.csv   county-level apple surveillance records
data/pairing/pairing_daily.csv                  daily weather per apple county (rows with model == "AMB")
data/agri_weather/valid_cells.csv               coordinates of the apple counties (Figure 2)
```

and run

```
python src/pipeline.py --project <project directory> --out <output directory>
python src/figures.py  --project <project directory> --out <output directory>
```

`pipeline.py` writes `tables/` and the record-level predictions `results/` under the output
directory (about 15 minutes); `figures.py` reads both and writes `figures_final/`. Figures 2, 5
and 7 need the record-level files in `results/`, which are not released; the other figures can
be redrawn from the released tables.

Columns expected in the surveillance file: `sigunguNm` (county), `sidoCode` (province),
`examinYear`, `inputStdrDatetm` (survey date, YYYYMMDD), `dbyhsNm` (index name) and
`inqireValue` (value).

## License

MIT. See `LICENSE`.
