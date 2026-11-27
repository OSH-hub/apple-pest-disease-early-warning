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
| `src/figures.py` | Figures 1-9 of the article |
| `src/check_scheduled_date.py` | Target date and interval replaced by their scheduled values; writes `tables/A17`-`A18` |
| `src/compare_learners.py` | Random forest and LSTM network on the units and design of the unified model; writes `tables/A19`-`A20` |
| `tables/` | Released aggregate tables |
| `figures/` | Released figures; file names carry the figure numbers of the article |

## Tables and the article

| File | Content | Article |
|---|---|---|
| `A00_thresholds.csv` | Threshold of each index | Section 2.2, Table 1 |
| `A01_data_summary.csv` | Records, counties, county-years, share at or above threshold | Table 1 |
| `A02_seasonal_pattern.csv` | Seasonal course in 15-day periods | Figure 3 |
| `A03_scores.csv` | AUC, detection and false-detection rates, Brier score for every forecast and index | Tables 2 and 4, Figure 4 |
| `A04_score_differences.csv` | Paired differences in AUC with bootstrap intervals | Table 2 |
| `A05_scores_by_year.csv` | Mean AUC in each test year | Figure 4 |
| `A06_reliability.csv` | Reliability in eight equal-count bins | Figure 8 |
| `A07_relative_value.csv` | Relative economic value by cost-loss ratio | Figure 8 |
| `A08_coefficients.csv` | Standardized coefficients of the model fitted to all years | Section 3.3 |
| `A09_ablation_scores.csv`, `A10_ablation_differences.csv` | Predictor groups added in turn and removed one at a time | Section 3.3, Figure 7 |
| `A11_sensitivity_scores.csv`, `A12_sensitivity_differences.csv` | Other thresholds, at-risk definition and 30-day target | Section 3.5, Figure 9 |
| `A13_new_index_scores.csv`, `A14_new_index_differences.csv` | Each index predicted by a model fitted to the other five | Table 3, Figure 6 |
| `A15_held_out_index_scores.csv`, `A16_held_out_index_differences.csv` | Index that took no part in model development | Table 3 |
| `A17_scheduled_date_scores.csv`, `A18_scheduled_date_differences.csv` | Target date and interval replaced by their scheduled values | Section 3.5 |
| `A19_learner_scores.csv`, `A20_learner_differences.csv` | Gradient boosting, random forest and LSTM network compared with the unified model | Table 2, Section 3.2 |

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
python src/check_scheduled_date.py --project <project directory> --out <output directory>
python src/compare_learners.py --project <project directory> --out <output directory>
```

`pipeline.py` writes `tables/` and the record-level predictions `results/` under the output
directory (30 minutes or more); `figures.py` reads both and writes `figures_final/`. Figures 2
and 5 need the record-level files in `results/`, which are not released; the other figures can
be redrawn from the released tables.

Columns expected in the surveillance file: `sigunguNm` (county), `sidoCode` (province),
`examinYear`, `inputStdrDatetm` (survey date, YYYYMMDD), `dbyhsNm` (index name) and
`inqireValue` (value).

## License

MIT. See `LICENSE`.
