from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(r"C:\Users\jungheeyoung\AppData\Local\Temp")
OUT_DIR = Path(r"C:\Users\jungheeyoung\Desktop\데이콘")

TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

LABELS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
WINDOWS = [
    ("w18_24", 18 * 60, 24 * 60),
    ("w20_24", 20 * 60, 24 * 60),
    ("w21_24", 21 * 60, 24 * 60),
    ("w22_24", 22 * 60, 24 * 60),
    ("w20_02", 20 * 60, 2 * 60),
]


def minute_of_day(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute + ts.dt.second / 60


def date_key(ts: pd.Series) -> pd.Series:
    return ts.dt.normalize()


def safe_std(x: pd.Series) -> float:
    return float(x.std(ddof=0)) if len(x) else np.nan


def entropy_from_counts(counts: list[int] | np.ndarray) -> float:
    arr = np.asarray(counts, dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr[arr > 0] / total
    return float(-(p * np.log2(p)).sum())


def sanitize_name(value: object, max_len: int = 40) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", text).strip("_")
    if not text:
        text = "unknown"
    return text[:max_len]


def add_window_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = date_key(df["timestamp"])
    df["minute"] = minute_of_day(df["timestamp"])
    return df


def window_subset(df: pd.DataFrame, start_minute: int, end_minute: int) -> pd.DataFrame:
    if start_minute < end_minute:
        mask = (df["minute"] >= start_minute) & (df["minute"] < end_minute)
        out = df.loc[mask].copy()
    else:
        mask = (df["minute"] >= start_minute) | (df["minute"] < end_minute)
        out = df.loc[mask].copy()
        next_day_mask = out["minute"] < end_minute
        out.loc[next_day_mask, "date"] = out.loc[next_day_mask, "date"] - pd.Timedelta(days=1)
    return out


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in df.columns
    ]
    return df


def numeric_agg(
    df: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
    include_evening: bool = True,
) -> pd.DataFrame:
    df = add_window_flag(df)
    pieces: list[pd.DataFrame] = []

    def build(part: pd.DataFrame, part_prefix: str) -> pd.DataFrame:
        grouped = part.groupby(["subject_id", "date"], observed=True)
        agg_dict = {
            col: ["count", "mean", "std", "min", "max", "median", "sum"]
            for col in value_cols
        }
        out = grouped.agg(agg_dict)
        out = flatten_columns(out)
        out = out.add_prefix(part_prefix)
        return out.reset_index()

    pieces.append(build(df, f"{prefix}_day_"))
    if include_evening:
        for window_name, start_minute, end_minute in WINDOWS:
            pieces.append(
                build(
                    window_subset(df, start_minute, end_minute),
                    f"{prefix}_{window_name}_",
                )
            )

    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=["subject_id", "date"], how="outer")
    return out


def binary_agg(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    df = add_window_flag(df)

    def build(part: pd.DataFrame, part_prefix: str) -> pd.DataFrame:
        if part.empty:
            return pd.DataFrame(columns=["subject_id", "date"])

        def summarize(g: pd.DataFrame) -> pd.Series:
            values = g[col].astype(float)
            on = values > 0
            transitions = values.ne(values.shift()).sum() - 1
            on_minutes = g.loc[on, "minute"]
            return pd.Series(
                {
                    f"{part_prefix}obs_count": len(g),
                    f"{part_prefix}ratio": float(on.mean()) if len(g) else np.nan,
                    f"{part_prefix}sum": float(values.sum()),
                    f"{part_prefix}transitions": float(max(transitions, 0)),
                    f"{part_prefix}first_on_minute": float(on_minutes.min())
                    if len(on_minutes)
                    else np.nan,
                    f"{part_prefix}last_on_minute": float(on_minutes.max())
                    if len(on_minutes)
                    else np.nan,
                    f"{part_prefix}longest_same_state": longest_run(values.tolist()),
                }
            )

        return (
            part.sort_values("timestamp")
            .groupby(["subject_id", "date"], observed=True)
            .apply(summarize, include_groups=False)
            .reset_index()
        )

    day = build(df, f"{prefix}_day_")
    out = day
    for window_name, start_minute, end_minute in WINDOWS:
        part = build(
            window_subset(df, start_minute, end_minute),
            f"{prefix}_{window_name}_",
        )
        out = out.merge(part, on=["subject_id", "date"], how="outer")
    return out


def longest_run(values: list[float]) -> float:
    best = 0
    cur = 0
    prev = None
    for value in values:
        if value == prev:
            cur += 1
        else:
            cur = 1
            prev = value
        best = max(best, cur)
    return float(best)


def activity_code_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = add_window_flag(df)
    codes = sorted(df["m_activity"].dropna().unique().tolist())

    def build(part: pd.DataFrame, part_prefix: str) -> pd.DataFrame:
        if part.empty:
            return pd.DataFrame(columns=["subject_id", "date"])
        counts = (
            part.groupby(["subject_id", "date", "m_activity"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        for code in codes:
            if code not in counts.columns:
                counts[code] = 0
        counts = counts[codes]
        total = counts.sum(axis=1).replace(0, np.nan)
        ratio = counts.div(total, axis=0)
        ratio.columns = [f"{part_prefix}code_{int(c)}_ratio" for c in ratio.columns]
        counts.columns = [f"{part_prefix}code_{int(c)}_count" for c in counts.columns]
        out = pd.concat([counts, ratio], axis=1)
        out[f"{part_prefix}obs_count"] = total
        out[f"{part_prefix}entropy"] = counts.apply(
            lambda row: entropy_from_counts(row.values), axis=1
        )
        return out.reset_index()

    day = build(df, "activity_state_day_")
    out = day
    for window_name, start_minute, end_minute in WINDOWS:
        part = build(
            window_subset(df, start_minute, end_minute),
            f"activity_state_{window_name}_",
        )
        out = out.merge(part, on=["subject_id", "date"], how="outer")
    return out


def scan_list_agg(
    df: pd.DataFrame,
    list_col: str,
    id_key: str,
    rssi_key: str,
    prefix: str,
) -> pd.DataFrame:
    df = add_window_flag(df)

    rows = []
    for row in df[["subject_id", "timestamp", "date", "minute", list_col]].itertuples(
        index=False
    ):
        items = getattr(row, list_col)
        if items is None:
            items = []
        count = len(items)
        ids = []
        rssis = []
        for item in items:
            if isinstance(item, dict):
                ids.append(item.get(id_key))
                value = item.get(rssi_key)
                if value is not None:
                    rssis.append(value)
        rows.append(
            {
                "subject_id": row.subject_id,
                "date": row.date,
                "timestamp": row.timestamp,
                "minute": row.minute,
                "scan_count": count,
                "unique_count": len(set(x for x in ids if x is not None)),
                "rssi_mean": float(np.mean(rssis)) if rssis else np.nan,
                "rssi_max": float(np.max(rssis)) if rssis else np.nan,
                "rssi_min": float(np.min(rssis)) if rssis else np.nan,
                "rssi_std": float(np.std(rssis)) if rssis else np.nan,
            }
        )

    flat = pd.DataFrame(rows)
    if flat.empty:
        return pd.DataFrame(columns=["subject_id", "date"])

    return numeric_agg(
        flat,
        ["scan_count", "unique_count", "rssi_mean", "rssi_max", "rssi_min", "rssi_std"],
        prefix,
    )


def usage_stats_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = add_window_flag(df)
    rows = []
    for row in df[["subject_id", "timestamp", "date", "minute", "m_usage_stats"]].itertuples(
        index=False
    ):
        items = row.m_usage_stats if row.m_usage_stats is not None else []
        total_time = 0.0
        app_times: Counter[str] = Counter()
        for item in items:
            if isinstance(item, dict):
                app = str(item.get("app_name", "unknown")).strip()
                val = float(item.get("total_time", 0) or 0)
                total_time += val
                app_times[app] += val
        counts = list(app_times.values())
        rows.append(
            {
                "subject_id": row.subject_id,
                "date": row.date,
                "timestamp": row.timestamp,
                "minute": row.minute,
                "app_record_count": len(items),
                "app_unique_count": len(app_times),
                "app_total_time": total_time,
                "app_max_time": max(counts) if counts else 0.0,
                "app_entropy": entropy_from_counts(counts),
                "app_top_share": max(counts) / total_time if total_time > 0 and counts else 0.0,
            }
        )

    flat = pd.DataFrame(rows)
    out = numeric_agg(
        flat,
        [
            "app_record_count",
            "app_unique_count",
            "app_total_time",
            "app_max_time",
            "app_entropy",
            "app_top_share",
        ],
        "app",
    )

    # Add low-dimensional app identity features from the most common apps.
    exploded = []
    for row in df[["subject_id", "date", "minute", "m_usage_stats"]].itertuples(index=False):
        for item in row.m_usage_stats if row.m_usage_stats is not None else []:
            if isinstance(item, dict):
                exploded.append(
                    {
                        "subject_id": row.subject_id,
                        "date": row.date,
                        "minute": row.minute,
                        "app_name": str(item.get("app_name", "unknown")).strip(),
                        "total_time": float(item.get("total_time", 0) or 0),
                    }
                )
    if exploded:
        apps = pd.DataFrame(exploded)
        top_apps = (
            apps.groupby("app_name")["total_time"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .index.tolist()
        )
        apps = apps[apps["app_name"].isin(top_apps)].copy()
        apps["app_feature"] = apps["app_name"].map(lambda x: f"app_top_{sanitize_name(x)}")

        def pivot_app(part: pd.DataFrame, prefix: str) -> pd.DataFrame:
            if part.empty:
                return pd.DataFrame(columns=["subject_id", "date"])
            pvt = part.pivot_table(
                index=["subject_id", "date"],
                columns="app_feature",
                values="total_time",
                aggfunc="sum",
                fill_value=0,
            )
            pvt = pvt.add_prefix(prefix)
            return pvt.reset_index()

        app_day = pivot_app(apps, "day_")
        out = out.merge(app_day, on=["subject_id", "date"], how="outer")
        for window_name, start_minute, end_minute in WINDOWS:
            app_window = window_subset(apps, start_minute, end_minute)
            out = out.merge(
                pivot_app(app_window, f"{window_name}_"),
                on=["subject_id", "date"],
                how="outer",
            )

    return out


def ambience_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = add_window_flag(df)
    rows = []
    top_labels = Counter()
    for row in df[["subject_id", "timestamp", "date", "minute", "m_ambience"]].itertuples(
        index=False
    ):
        items = row.m_ambience if row.m_ambience is not None else []
        labels = []
        probs = []
        for item in items:
            if len(item) >= 2:
                labels.append(str(item[0]))
                try:
                    probs.append(float(item[1]))
                except Exception:
                    probs.append(np.nan)
        if labels:
            top_labels[labels[0]] += 1
        rows.append(
            {
                "subject_id": row.subject_id,
                "date": row.date,
                "timestamp": row.timestamp,
                "minute": row.minute,
                "ambience_prob_top1": probs[0] if probs else np.nan,
                "ambience_prob_top3_sum": float(np.nansum(probs[:3])) if probs else np.nan,
                "ambience_prob_entropy": entropy_from_counts(
                    [p for p in probs if not pd.isna(p)]
                )
                if probs
                else np.nan,
                "ambience_top1": labels[0] if labels else "unknown",
            }
        )

    flat = pd.DataFrame(rows)
    out = numeric_agg(
        flat,
        ["ambience_prob_top1", "ambience_prob_top3_sum", "ambience_prob_entropy"],
        "ambience",
    )

    keep_labels = [label for label, _ in top_labels.most_common(20)]
    flat["ambience_group"] = flat["ambience_top1"].where(
        flat["ambience_top1"].isin(keep_labels), "other"
    )

    def pivot_label(part: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if part.empty:
            return pd.DataFrame(columns=["subject_id", "date"])
        counts = (
            part.groupby(["subject_id", "date", "ambience_group"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        total = counts.sum(axis=1).replace(0, np.nan)
        ratio = counts.div(total, axis=0)
        ratio.columns = [
            f"{prefix}ambience_top1_{sanitize_name(c)}_ratio" for c in ratio.columns
        ]
        return ratio.reset_index()

    out = out.merge(pivot_label(flat, "day_"), on=["subject_id", "date"], how="outer")
    for window_name, start_minute, end_minute in WINDOWS:
        out = out.merge(
            pivot_label(window_subset(flat, start_minute, end_minute), f"{window_name}_"),
            on=["subject_id", "date"],
            how="outer",
        )
    return out


def gps_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = add_window_flag(df)
    rows = []
    for row in df[["subject_id", "timestamp", "date", "minute", "m_gps"]].itertuples(
        index=False
    ):
        points = row.m_gps if row.m_gps is not None else []
        lat = []
        lon = []
        alt = []
        speed = []
        for point in points:
            if isinstance(point, dict):
                lat.append(float(point.get("latitude", np.nan)))
                lon.append(float(point.get("longitude", np.nan)))
                alt.append(float(point.get("altitude", np.nan)))
                speed.append(float(point.get("speed", np.nan)))
        rows.append(
            {
                "subject_id": row.subject_id,
                "date": row.date,
                "timestamp": row.timestamp,
                "minute": row.minute,
                "gps_point_count": len(points),
                "gps_lat_mean": float(np.nanmean(lat)) if lat else np.nan,
                "gps_lon_mean": float(np.nanmean(lon)) if lon else np.nan,
                "gps_alt_mean": float(np.nanmean(alt)) if alt else np.nan,
                "gps_speed_mean": float(np.nanmean(speed)) if speed else np.nan,
                "gps_speed_max": float(np.nanmax(speed)) if speed else np.nan,
            }
        )
    flat = pd.DataFrame(rows)
    out = numeric_agg(
        flat,
        [
            "gps_point_count",
            "gps_lat_mean",
            "gps_lon_mean",
            "gps_alt_mean",
            "gps_speed_mean",
            "gps_speed_max",
        ],
        "gps",
    )

    def span_features(part: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if part.empty:
            return pd.DataFrame(columns=["subject_id", "date"])

        def summarize(g: pd.DataFrame) -> pd.Series:
            lat_span = g["gps_lat_mean"].max() - g["gps_lat_mean"].min()
            lon_span = g["gps_lon_mean"].max() - g["gps_lon_mean"].min()
            return pd.Series(
                {
                    f"{prefix}lat_span": lat_span,
                    f"{prefix}lon_span": lon_span,
                    f"{prefix}rough_radius": math.sqrt(
                        float(lat_span) ** 2 + float(lon_span) ** 2
                    )
                    if pd.notna(lat_span) and pd.notna(lon_span)
                    else np.nan,
                    f"{prefix}stationary_ratio": float((g["gps_speed_mean"] <= 0.1).mean()),
                }
            )

        return (
            part.groupby(["subject_id", "date"], observed=True)
            .apply(summarize, include_groups=False)
            .reset_index()
        )

    out = out.merge(span_features(flat, "gps_day_"), on=["subject_id", "date"], how="outer")
    for window_name, start_minute, end_minute in WINDOWS:
        out = out.merge(
            span_features(window_subset(flat, start_minute, end_minute), f"gps_{window_name}_"),
            on=["subject_id", "date"],
            how="outer",
        )
    return out


def add_temporal_context(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    dt = pd.to_datetime(features["lifelog_date"])
    features["weekday"] = dt.dt.weekday
    features["is_weekend"] = (features["weekday"] >= 5).astype(int)
    features["month"] = dt.dt.month
    features["dayofmonth"] = dt.dt.day
    return features


def add_personal_baselines(features: pd.DataFrame) -> pd.DataFrame:
    features = features.sort_values(["subject_id", "lifelog_date"]).copy()
    exclude = {
        "subject_id",
        "sleep_date",
        "lifelog_date",
        "split",
        *LABELS,
    }
    numeric_cols = [
        c
        for c in features.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(features[c])
    ]
    core_cols = [
        c
        for c in numeric_cols
        if any(
            key in c
            for key in [
                "screen",
                "charging",
                "app_app_total_time",
                "light",
                "step",
                "distance",
                "speed",
                "wifi_day_scan_count_mean",
                "ble_day_scan_count_mean",
                "gps_day_rough_radius",
            ]
        )
    ][:80]
    new_cols = {}
    for col in core_cols:
        subj_mean = features.groupby("subject_id", observed=True)[col].transform("mean")
        subj_std = features.groupby("subject_id", observed=True)[col].transform("std").replace(0, np.nan)
        new_cols[f"personal_delta_{col}"] = features[col] - subj_mean
        new_cols[f"personal_z_{col}"] = (features[col] - subj_mean) / subj_std
    if new_cols:
        features = pd.concat([features, pd.DataFrame(new_cols, index=features.index)], axis=1)
    return features.copy()


def drop_unusable_features(features: pd.DataFrame) -> pd.DataFrame:
    protected = {"subject_id", "sleep_date", "lifelog_date", "split", *LABELS}
    train_mask = features["split"].eq("train")
    drop_cols = []
    for col in features.columns:
        if col in protected:
            continue
        train_values = features.loc[train_mask, col]
        if train_values.isna().all():
            drop_cols.append(col)
            continue
        if train_values.nunique(dropna=True) <= 1:
            drop_cols.append(col)
    if drop_cols:
        print(f"dropping unusable feature columns: {len(drop_cols)}")
        features = features.drop(columns=drop_cols)
    return features


def load_targets() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
    train["split"] = "train"
    sample["split"] = "test"
    sample[LABELS] = np.nan
    combined = pd.concat([train, sample], ignore_index=True)
    combined["date"] = combined["lifelog_date"].dt.normalize()
    return combined


def merge_feature(base: pd.DataFrame, feature: pd.DataFrame, name: str) -> pd.DataFrame:
    before_cols = base.shape[1]
    out = base.merge(feature, on=["subject_id", "date"], how="left")
    added = out.shape[1] - before_cols
    print(f"merged {name}: +{added} columns, shape={out.shape}")
    return out


def build_features() -> pd.DataFrame:
    base = load_targets()
    print(f"target rows: {base.shape}")

    parquet_jobs = [
        (
            "steps",
            "ch2025_mActivity.parquet",
            lambda df: numeric_agg(
                df,
                [
                    "step",
                    "step_frequency",
                    "distance",
                    "speed",
                    "burned_calories",
                ],
                "move",
            ),
        ),
        (
            "activity_state",
            "ch2025_mACStatus.parquet",
            activity_code_agg,
        ),
        (
            "charging",
            "ch2025_mScreenStatus.parquet",
            lambda df: binary_agg(df, "m_charging", "charging"),
        ),
        (
            "screen",
            "ch2025_wLight.parquet",
            lambda df: binary_agg(df, "m_screen_use", "screen"),
        ),
        (
            "phone_light",
            "ch2025_wPedo.parquet",
            lambda df: numeric_agg(df, ["m_light"], "phone_light"),
        ),
        (
            "watch_light",
            "ch2025_mWifi.parquet",
            lambda df: numeric_agg(df, ["w_light"], "watch_light"),
        ),
        (
            "wifi",
            "ch2025_mGps.parquet",
            lambda df: scan_list_agg(df, "m_wifi", "bssid", "rssi", "wifi"),
        ),
        (
            "ble",
            "ch2025_mUsageStats.parquet",
            lambda df: scan_list_agg(df, "m_ble", "address", "rssi", "ble"),
        ),
        (
            "apps",
            "ch2025_mLight.parquet",
            usage_stats_agg,
        ),
        (
            "ambience",
            "ch2025_mBle.parquet",
            ambience_agg,
        ),
        (
            "gps",
            "ch2025_wHr.parquet",
            gps_agg,
        ),
    ]

    features = base
    for name, file_name, builder in parquet_jobs:
        path = BASE_DIR / file_name
        print(f"reading {file_name} for {name}")
        df = pd.read_parquet(path)
        part = builder(df)
        features = merge_feature(features, part, name)

    features = add_temporal_context(features)
    features = add_personal_baselines(features)
    features = drop_unusable_features(features)
    features = features.drop(columns=["date"])

    id_cols = ["subject_id", "sleep_date", "lifelog_date", "split"]
    label_cols = [col for col in LABELS if col in features.columns]
    other_cols = [col for col in features.columns if col not in id_cols + label_cols]
    features = features[id_cols + label_cols + other_cols]
    return features


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = build_features()

    all_path = OUT_DIR / "feature_table_all.csv"
    train_path = OUT_DIR / "feature_table_train.csv"
    test_path = OUT_DIR / "feature_table_test.csv"
    parquet_path = OUT_DIR / "feature_table_all.parquet"

    train = features[features["split"] == "train"].copy()
    test = features[features["split"] == "test"].copy()

    features.to_csv(all_path, index=False, encoding="utf-8-sig")
    train.to_csv(train_path, index=False, encoding="utf-8-sig")
    test.to_csv(test_path, index=False, encoding="utf-8-sig")
    features.to_parquet(parquet_path, index=False)

    feature_cols = [
        col
        for col in features.columns
        if col not in ["subject_id", "sleep_date", "lifelog_date", "split", *LABELS]
    ]
    with (OUT_DIR / "feature_columns.txt").open("w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    print("saved:")
    print(f"  {all_path} {features.shape}")
    print(f"  {train_path} {train.shape}")
    print(f"  {test_path} {test.shape}")
    print(f"  {parquet_path} {features.shape}")
    print(f"feature columns: {len(feature_cols)}")


if __name__ == "__main__":
    main()
