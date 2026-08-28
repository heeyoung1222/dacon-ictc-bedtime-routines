# %% [markdown]
# # Sleep-window Multi-day Feature Table
#
# 목적:
# - sleep-window feature table에 최근 며칠 rolling/delta/z-score feature를 추가함
# - 야간 window도 우선순위에 포함함

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ALL_PATH = PROJECT_DIR / "feature_table_all_sleepwindows.parquet"
OUT_ALL_PARQUET = PROJECT_DIR / "feature_table_all_sleepwindows_multiday.parquet"
OUT_ALL_CSV = PROJECT_DIR / "feature_table_all_sleepwindows_multiday.csv"
OUT_TRAIN_CSV = PROJECT_DIR / "feature_table_train_sleepwindows_multiday.csv"
OUT_TEST_CSV = PROJECT_DIR / "feature_table_test_sleepwindows_multiday.csv"

LABELS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
PROTECTED = {"subject_id", "sleep_date", "lifelog_date", "split", *LABELS}

KEYWORDS = [
    "screen",
    "charging",
    "app_app_total_time",
    "app_app_entropy",
    "app_app_unique_count",
    "move_day_step_sum",
    "move_day_distance_sum",
    "move_day_speed_mean",
    "move_day_burned_calories_sum",
    "phone_light",
    "watch_light",
    "gps_day_rough_radius",
    "gps_day_stationary_ratio",
    "wifi_day_scan_count_mean",
    "ble_day_scan_count_mean",
    "ambience",
    "activity_state_day_entropy",
]

NIGHT_WINDOWS = ["_w20_02_", "_w20_04_", "_w20_06_", "_w22_04_", "_w22_06_"]


def select_base_columns(df: pd.DataFrame, max_cols: int = 260) -> list[str]:
    numeric_cols = [
        c
        for c in df.columns
        if c not in PROTECTED
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("personal_")
    ]
    selected = [
        c
        for c in numeric_cols
        if any(keyword in c for keyword in KEYWORDS)
        and (
            "_mean" in c
            or "_sum" in c
            or "_ratio" in c
            or "_count" in c
            or "_entropy" in c
            or "_last_on_minute" in c
            or "_first_on_minute" in c
            or "_rough_radius" in c
            or "_stationary_ratio" in c
        )
    ]
    priority = []
    for c in selected:
        score = len(c) / 1000
        if "_day_" in c or c.startswith("day_"):
            score -= 4
        if any(token in c for token in NIGHT_WINDOWS):
            score -= 3
        if "_w21_24_" in c or c.startswith("w21_24_"):
            score -= 2
        if "_w18_24_" in c or c.startswith("w18_24_"):
            score -= 1
        priority.append((score, c))
    return [c for _, c in sorted(priority)[:max_cols]]


def add_multiday_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["lifelog_date"] = pd.to_datetime(df["lifelog_date"])
    df = df.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)
    grouped = df.groupby("subject_id", sort=False)

    new_parts = []
    for window in [3, 5, 7, 14]:
        shifted = grouped[base_cols].shift(1)
        roll_mean = shifted.groupby(df["subject_id"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        roll_std = shifted.groupby(df["subject_id"]).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
        roll_mean.columns = [f"md_roll{window}_mean_{c}" for c in base_cols]
        roll_std.columns = [f"md_roll{window}_std_{c}" for c in base_cols]
        new_parts.append(roll_mean)
        if window in [3, 7]:
            new_parts.append(roll_std)

    prev = grouped[base_cols].shift(1)
    prev2 = grouped[base_cols].shift(2)
    delta1 = df[base_cols] - prev
    delta2 = prev - prev2
    delta1.columns = [f"md_delta_today_prev_{c}" for c in base_cols]
    delta2.columns = [f"md_delta_prev_prev2_{c}" for c in base_cols]
    new_parts.extend([delta1, delta2])

    subj_mean = grouped[base_cols].transform("mean")
    subj_std = grouped[base_cols].transform("std").replace(0, np.nan)
    z = (df[base_cols] - subj_mean) / subj_std
    z.columns = [f"md_subject_z_{c}" for c in base_cols]
    new_parts.append(z)

    return pd.concat([df, pd.concat(new_parts, axis=1)], axis=1)


def drop_unusable(df: pd.DataFrame) -> pd.DataFrame:
    train_mask = df["split"].eq("train")
    drop_cols = []
    for col in df.columns:
        if col in PROTECTED:
            continue
        values = df.loc[train_mask, col]
        if values.isna().all() or values.nunique(dropna=True) <= 1:
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def main() -> None:
    df = pd.read_parquet(ALL_PATH)
    base_cols = select_base_columns(df)
    out = add_multiday_features(df, base_cols)
    out = drop_unusable(out)
    train = out[out["split"] == "train"].copy()
    test = out[out["split"] == "test"].copy()

    out.to_parquet(OUT_ALL_PARQUET, index=False)
    out.to_csv(OUT_ALL_CSV, index=False, encoding="utf-8-sig")
    train.to_csv(OUT_TRAIN_CSV, index=False, encoding="utf-8-sig")
    test.to_csv(OUT_TEST_CSV, index=False, encoding="utf-8-sig")

    print("selected base cols", len(base_cols))
    print("all", out.shape)
    print("train", train.shape)
    print("test", test.shape)
    print("saved:")
    for path in [OUT_ALL_PARQUET, OUT_ALL_CSV, OUT_TRAIN_CSV, OUT_TEST_CSV]:
        print(path)


if __name__ == "__main__":
    main()
