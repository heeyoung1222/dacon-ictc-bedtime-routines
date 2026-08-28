# Remade Visual Summary

This folder contains lightweight figures regenerated from the preserved result
tables. They are intended for quickly understanding the research story without
opening the full paper or rerunning the complete competition pipeline.

## Research Summary

The project models next-day sleep and fatigue/stress labels from smartphone and
wearable lifelogs. The core idea is the **Digital Bedtime Routine Window
(DBRW)**, a pre-sleep time interval that captures the transition from daytime
activity to bedtime behavior.

Instead of relying on one simple rule such as screen-off time or low ambient
light, the model combines several context signals:

- screen and charging state
- app-use behavior
- location and mobility summaries
- Wi-Fi and BLE context
- phone and wearable light
- wearable activity signals

## Main Finding

The preserved validation run found that adding the `21:00-24:00` bedtime window
to day-level features produced the best average log-loss:

| Feature set | Mean log-loss |
| --- | ---: |
| `day_plus_w21_24` | 0.6575 |
| `day_only` | 0.6590 |
| `day_plus_w20_02` | 0.6611 |

The effect is modest, but it supports the paper's conclusion that the immediate
pre-sleep transition contains useful behavioral context.

## Figures

| File | Meaning |
| --- | --- |
| `window_ablation_logloss.png` | Mean validation log-loss for each feature-window setting. Lower is better. |
| `labelwise_window_logloss.png` | Label-by-label comparison of log-loss across window settings. |
| `feature_group_importance.png` | Aggregated ExtraTrees feature importance by sensor/context family. |

## How These Figures Were Made

Run from the repository root:

```powershell
python src/dbrw_extratrees_pipeline.py make-readme-figures --results-dir results --output-dir remade
```

The command reads only the small CSV files in `results/`; it does not require
raw competition data.
