# DBRW Sleep and Stress Prediction

Code package for **Modeling Personalized Digital Bedtime Routines for Sleep and
Stress Prediction Using Multimodal Lifelogs**, prepared from the DACON / ETRI
Human Understanding AI Paper Challenge work.

The submitted paper's central experiment compares daily features with bedtime
routine windows. The best validation window in the preserved run was
`day_plus_w21_24`, with mean log-loss `0.6575`, compared with `0.6590` for
`day_only`.

![DBRW window ablation](remade/window_ablation_logloss.png)

## Repository Layout

| Path | What it contains |
| --- | --- |
| `src/dbrw_extratrees_pipeline.py` | Consolidated one-file pipeline for validation, submission generation, feature importance, and README figures. |
| `original_sources/` | Original scripts copied with their original filenames so their role and result history remain traceable. |
| `results/` | Small preserved result tables used in the paper narrative. No raw competition data are included. |
| `remade/` | Lightweight visualizations regenerated from the preserved result tables for quick review. |
| `paper_assets/` | Paper figures and LaTeX draft assets found in the local project. |
| `ORIGINAL_SOURCE_MAP.md` | Mapping from original files to the consolidated pipeline. |

Some scripts under `original_sources/` intentionally preserve old absolute
local paths from the working computer. They are kept for traceability. Use
`src/dbrw_extratrees_pipeline.py` for portable reruns.

## Main Reproduction Script

Install dependencies:

```powershell
pip install -r requirements.txt
```

Prepare competition-derived feature files locally:

```text
data/
  feature_table_train.csv
  feature_table_test.csv
  ch2026_submission_sample.csv
```

The raw DACON/ETRI data and generated large feature tables are intentionally not
committed. To rebuild the feature tables from raw data, start from
`original_sources/make_features.py`, then the sleep-window extension scripts.

Run the DBRW window ablation:

```powershell
python src/dbrw_extratrees_pipeline.py validate-windows --data-dir data --output-dir outputs
```

Create submission candidates:

```powershell
python src/dbrw_extratrees_pipeline.py make-submission --data-dir data --output-dir outputs --window-results outputs/window_feature_validation_results.csv
```

Compute feature importances:

```powershell
python src/dbrw_extratrees_pipeline.py feature-importance --data-dir data --output-dir outputs --window-results outputs/window_feature_validation_results.csv
```

Regenerate README figures from saved small results:

```powershell
python src/dbrw_extratrees_pipeline.py make-readme-figures --results-dir results --output-dir remade
```

## Preserved Result Summary

The saved validation summary is in `results/window_feature_validation_summary.md`.

| Feature set | Mean log-loss |
| --- | ---: |
| `day_plus_w21_24` | 0.6575 |
| `day_plus_w22_24` | 0.6580 |
| `day_plus_w18_24` | 0.6588 |
| `day_only` | 0.6590 |
| `day_plus_w20_24` | 0.6598 |
| `day_plus_all_windows` | 0.6599 |
| `day_plus_w20_02` | 0.6611 |

![Label-wise DBRW comparison](remade/labelwise_window_logloss.png)

## Model

The paper-aligned model is a separate binary classifier for each label:

- `ExtraTreesClassifier`
- `n_estimators=500`
- `max_depth=4`
- `min_samples_leaf=4`
- `max_features="sqrt"`
- `class_weight="balanced"`
- median imputation inside the pipeline
- probability shrinkage: `p' = alpha * p + (1 - alpha) * prior`

The shrinkage coefficient is selected from `0.5, 0.55, ..., 1.0` using only an
inner chronological calibration split from the training rows.

## Labels

The seven prediction targets are:

| Group | Labels |
| --- | --- |
| Questionnaire | `Q1`, `Q2`, `Q3` |
| Sleep indices | `S1`, `S2`, `S3`, `S4` |

![Feature groups](remade/feature_group_importance.png)

## Notes

- `personal_*` features are dropped by default in the consolidated pipeline to
  match the submitted paper's conservative interpretation.
- Use `--keep-personal` only for exploratory runs.
- The repository is meant for code review and reproducibility. It does not
  include raw lifelog data, heavy feature tables, or final competition submission
  files.
