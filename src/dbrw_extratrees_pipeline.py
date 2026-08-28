from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline


"""
One-file reproduction pipeline for:

Modeling Personalized Digital Bedtime Routines for Sleep and Stress Prediction
Using Multimodal Lifelogs

This file consolidates the reproducible core from:
- original_sources/compare_window_features.py
- original_sources/03_make_submission_extratrees_shrinkage.py
- original_sources/step2_feature_importance.py
- original_sources/step4_figures_v2.py
- original_sources/step5_tables.py

Raw feature construction remains in original_sources/make_features.py and the
sleep-window extension scripts because the competition data layout is external
and should not be redistributed in this repository.
"""


LABELS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLS = ["subject_id", "sleep_date", "lifelog_date", "split"]
WINDOWS = ["w18_24", "w20_24", "w21_24", "w22_24", "w20_02"]
RANDOM_STATE = 42
CLIP_MIN = 0.02
CLIP_MAX = 0.98


def read_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["sleep_date", "lifelog_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def has_window_token(col: str, window: str) -> bool:
    return f"_{window}_" in col or col.startswith(f"{window}_")


def has_any_window_token(col: str) -> bool:
    return any(has_window_token(col, window) for window in WINDOWS)


def get_feature_sets(df: pd.DataFrame, drop_personal: bool = True) -> dict[str, list[str]]:
    protected = set(ID_COLS + LABELS)
    numeric_cols = [
        col
        for col in df.columns
        if col not in protected and pd.api.types.is_numeric_dtype(df[col])
    ]
    clean_cols = [
        col for col in numeric_cols if not (drop_personal and col.startswith("personal_"))
    ]
    base_cols = [col for col in clean_cols if not has_any_window_token(col)]

    feature_sets = {"day_only": base_cols}
    for window in WINDOWS:
        window_cols = [col for col in clean_cols if has_window_token(col, window)]
        feature_sets[f"day_plus_{window}"] = base_cols + window_cols
    feature_sets["day_plus_all_windows"] = clean_cols
    return feature_sets


def make_subject_time_split(
    df: pd.DataFrame,
    valid_ratio: float = 0.2,
    min_valid_per_subject: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices: list[int] = []
    valid_indices: list[int] = []

    for _, group in df.sort_values("lifelog_date").groupby("subject_id", sort=True):
        group = group.sort_values("lifelog_date")
        n = len(group)
        n_valid = max(min_valid_per_subject, int(np.ceil(n * valid_ratio)))
        n_valid = min(n_valid, max(1, n - 1))
        train_indices.extend(group.index[:-n_valid].tolist())
        valid_indices.extend(group.index[-n_valid:].tolist())

    return np.array(train_indices), np.array(valid_indices)


def make_inner_calibration_fold(
    df: pd.DataFrame,
    indices: np.ndarray | None = None,
    valid_ratio: float = 0.2,
    min_cal_per_subject: int = 2,
) -> np.ndarray:
    if indices is None:
        indices = df.index.to_numpy()
    fold = pd.Series(-1, index=indices, dtype=int)
    train_df = df.loc[indices].sort_values("lifelog_date")

    for _, group in train_df.groupby("subject_id", sort=True):
        group = group.sort_values("lifelog_date")
        n = len(group)
        n_cal = max(min_cal_per_subject, int(np.ceil(n * valid_ratio)))
        n_cal = min(n_cal, max(1, n - 1))
        fold.loc[group.index[-n_cal:]] = 0

    return fold.loc[indices].to_numpy()


def extra_trees_estimator() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=4,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def get_positive_probability(model: object, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(getattr(model, "classes_", np.array([0, 1])))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(x))


def choose_shrinkage_alpha(p_cal: np.ndarray, y_cal: np.ndarray, prior: float) -> float:
    best_alpha = 1.0
    best_loss = np.inf
    for alpha in np.linspace(0.5, 1.0, 11):
        p_adj = alpha * p_cal + (1.0 - alpha) * prior
        loss = log_loss(y_cal, np.clip(p_adj, 1e-15, 1 - 1e-15), labels=[0, 1])
        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)
    return best_alpha


def safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_proba))


def evaluate_feature_sets(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_fold = make_inner_calibration_fold(df, train_idx)
    result_rows = []
    pred_frames = []

    for feature_set_name, feature_cols in feature_sets.items():
        x_train = df.loc[train_idx, feature_cols]
        x_valid = df.loc[valid_idx, feature_cols]
        inner_train_mask = calibration_fold == -1
        inner_cal_mask = calibration_fold == 0

        for label in LABELS:
            y_train = df.loc[train_idx, label].astype(int)
            y_valid = df.loc[valid_idx, label].astype(int)

            inner_model = extra_trees_estimator()
            inner_model.fit(x_train.iloc[inner_train_mask], y_train.iloc[inner_train_mask])
            p_cal = get_positive_probability(inner_model, x_train.iloc[inner_cal_mask])
            y_cal = y_train.iloc[inner_cal_mask].to_numpy()
            prior = float(y_train.iloc[inner_train_mask].mean())
            alpha = choose_shrinkage_alpha(p_cal, y_cal, prior)

            model = extra_trees_estimator()
            model.fit(x_train, y_train)
            raw_proba = get_positive_probability(model, x_valid)
            final_prior = float(y_train.mean())
            y_proba = alpha * raw_proba + (1.0 - alpha) * final_prior
            y_pred = (y_proba >= 0.5).astype(int)

            result_rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": "ExtraTreesShrinkage",
                    "label": label,
                    "n_features": len(feature_cols),
                    "log_loss": log_loss(
                        y_valid,
                        np.clip(y_proba, 1e-15, 1 - 1e-15),
                        labels=[0, 1],
                    ),
                    "accuracy": accuracy_score(y_valid, y_pred),
                    "macro_f1": f1_score(y_valid, y_pred, average="macro", zero_division=0),
                    "roc_auc": safe_auc(y_valid.to_numpy(), y_proba),
                    "shrinkage_alpha": alpha,
                    "valid_positive_rate": float(y_valid.mean()),
                    "pred_proba_mean": float(np.mean(y_proba)),
                }
            )

            pred = df.loc[valid_idx, ["subject_id", "sleep_date", "lifelog_date"]].copy()
            pred["feature_set"] = feature_set_name
            pred["label"] = label
            pred["y_true"] = y_valid.to_numpy()
            pred["y_pred"] = y_pred
            pred["y_proba"] = y_proba
            pred_frames.append(pred)

    return pd.DataFrame(result_rows), pd.concat(pred_frames, ignore_index=True)


def write_window_summary(
    results: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    output_path: Path,
) -> None:
    summary = (
        results.groupby("feature_set", as_index=False)[
            ["log_loss", "accuracy", "macro_f1", "roc_auc"]
        ]
        .mean()
        .sort_values("log_loss")
    )
    label_summary = (
        results.pivot_table(index="label", columns="feature_set", values="log_loss")
        .reset_index()
    )

    lines = [
        "# DBRW Window Ablation Results",
        "",
        "## Validation Protocol",
        "",
        "- Model: ExtraTreesShrinkage",
        "- Split: first 80% of each subject timeline for training, last 20% for validation",
        "- Calibration alpha: selected only on an inner chronological training fold",
        f"- Train rows: {len(train_idx)}",
        f"- Validation rows: {len(valid_idx)}",
        "",
        "## Mean Performance by Feature Set",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Label-wise Log-Loss",
        "",
        label_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Feature Counts",
        "",
    ]
    lines.extend(f"- {name}: {len(cols)} features" for name, cols in feature_sets.items())
    output_path.write_text("\n".join(lines), encoding="utf-8")


def command_validate_windows(args: argparse.Namespace) -> None:
    train_path = args.data_dir / args.train_file
    df = read_feature_table(train_path)
    train_idx, valid_idx = make_subject_time_split(df)
    feature_sets = get_feature_sets(df, drop_personal=not args.keep_personal)
    results, predictions = evaluate_feature_sets(df, feature_sets, train_idx, valid_idx)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "window_feature_validation_results.csv"
    predictions_path = args.output_dir / "window_feature_validation_predictions.csv"
    summary_path = args.output_dir / "window_feature_validation_summary.md"

    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    write_window_summary(results, feature_sets, train_idx, valid_idx, summary_path)

    print(results.groupby("feature_set")[["log_loss", "accuracy", "macro_f1"]].mean().sort_values("log_loss"))
    print(f"\nSaved validation outputs to {args.output_dir}")


def choose_labelwise_feature_sets(results_path: Path | None) -> dict[str, str]:
    fallback = {
        "Q1": "day_plus_all_windows",
        "Q2": "day_plus_w18_24",
        "Q3": "day_only",
        "S1": "day_plus_all_windows",
        "S2": "day_plus_w21_24",
        "S3": "day_plus_all_windows",
        "S4": "day_plus_w21_24",
    }
    if results_path is None or not results_path.exists():
        return fallback
    results = pd.read_csv(results_path)
    idx = results.groupby("label")["log_loss"].idxmin()
    best = results.loc[idx, ["label", "feature_set"]]
    return dict(zip(best["label"], best["feature_set"]))


def fit_predict_label(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
    feature_cols: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    x_train = train[feature_cols]
    y_train = train[label].astype(int)
    x_test = test[feature_cols]

    calibration_fold = make_inner_calibration_fold(train)
    inner_train_mask = calibration_fold == -1
    inner_cal_mask = calibration_fold == 0

    inner_model = extra_trees_estimator()
    inner_model.fit(x_train.iloc[inner_train_mask], y_train.iloc[inner_train_mask])
    p_cal = get_positive_probability(inner_model, x_train.iloc[inner_cal_mask])
    y_cal = y_train.iloc[inner_cal_mask].to_numpy()
    prior_inner = float(y_train.iloc[inner_train_mask].mean())
    alpha = choose_shrinkage_alpha(p_cal, y_cal, prior_inner)

    model = extra_trees_estimator()
    model.fit(x_train, y_train)
    raw_proba = get_positive_probability(model, x_test)
    prior_final = float(y_train.mean())
    proba = alpha * raw_proba + (1.0 - alpha) * prior_final
    proba = np.clip(proba, CLIP_MIN, CLIP_MAX)

    info = {
        "label": label,
        "n_features": len(feature_cols),
        "alpha": alpha,
        "train_positive_rate": prior_final,
        "test_pred_mean": float(np.mean(proba)),
        "test_pred_min": float(np.min(proba)),
        "test_pred_max": float(np.max(proba)),
    }
    return proba, info


def build_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    label_to_feature_set: dict[str, str],
    output_path: Path,
) -> pd.DataFrame:
    submission = sample.copy()
    summary_rows = []

    for label in LABELS:
        feature_set_name = label_to_feature_set[label]
        feature_cols = feature_sets[feature_set_name]
        proba, info = fit_predict_label(train, test, label, feature_cols)
        submission[label] = proba
        info["feature_set"] = feature_set_name
        info["output_file"] = output_path.name
        summary_rows.append(info)

    for col in ["sleep_date", "lifelog_date"]:
        if col in submission.columns:
            submission[col] = pd.to_datetime(submission[col]).dt.strftime("%Y-%m-%d")
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(summary_rows)


def command_make_submission(args: argparse.Namespace) -> None:
    train = read_feature_table(args.data_dir / args.train_file)
    test = read_feature_table(args.data_dir / args.test_file)
    sample = read_feature_table(args.data_dir / args.sample_file)
    feature_sets = get_feature_sets(train, drop_personal=not args.keep_personal)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_path = args.output_dir / "submission_global_w21_24.csv"
    labelwise_path = args.output_dir / "submission_labelwise_best_window.csv"
    selection_path = args.output_dir / "final_model_selection_summary.csv"

    global_feature_map = {label: "day_plus_w21_24" for label in LABELS}
    labelwise_feature_map = choose_labelwise_feature_sets(args.window_results)

    global_summary = build_submission(train, test, sample, feature_sets, global_feature_map, global_path)
    labelwise_summary = build_submission(train, test, sample, feature_sets, labelwise_feature_map, labelwise_path)

    summary = pd.concat([global_summary, labelwise_summary], ignore_index=True)
    summary.to_csv(selection_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nSaved submission outputs to {args.output_dir}")


def feature_window_filter(feature: str, target_window: str) -> bool:
    if target_window == "all":
        return True
    if target_window == "day":
        return not has_any_window_token(feature)
    return target_window in feature or not has_any_window_token(feature)


def command_feature_importance(args: argparse.Namespace) -> None:
    df = read_feature_table(args.data_dir / args.train_file)
    if not args.keep_personal:
        df = df.drop(columns=[c for c in df.columns if c.startswith("personal_")], errors="ignore")

    label_to_window = choose_labelwise_feature_sets(args.window_results)
    label_to_window = {
        label: label_to_window.get(label, "day_only").replace("day_plus_", "").replace("day_only", "day")
        for label in LABELS
    }

    importance_rows = []
    for label in LABELS:
        protected = set(ID_COLS + [x for x in LABELS if x != label])
        candidates = [
            c
            for c in df.columns
            if c not in protected
            and c != label
            and pd.api.types.is_numeric_dtype(df[c])
            and feature_window_filter(c, label_to_window[label])
        ]
        X = df[candidates].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df[label].astype(int)
        model = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X, y)
        for feature, importance in zip(X.columns, model.feature_importances_):
            importance_rows.append({"Label": label, "Feature": feature, "Importance": importance})

    top = (
        pd.DataFrame(importance_rows)
        .sort_values(["Label", "Importance"], ascending=[True, False])
        .groupby("Label")
        .head(args.top_n)
        .reset_index(drop=True)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "feature_importance.csv"
    top.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {out_path}")


def categorize_feature(feature: str) -> str:
    f = feature.lower()
    if "app" in f:
        return "App Usage"
    if "wifi" in f or "ble" in f:
        return "Network Context"
    if "gps" in f or "location" in f:
        return "Location/Mobility"
    if "screen" in f or "charging" in f or "light" in f:
        return "Direct Routine Cues"
    return "Other Sensors"


def command_make_readme_figures(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / "window_feature_validation_results.csv"
    importance_path = args.results_dir / "feature_importance.csv"

    if results_path.exists():
        results = pd.read_csv(results_path)
        summary = (
            results.groupby("feature_set", as_index=False)["log_loss"]
            .mean()
            .sort_values("log_loss")
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(summary["feature_set"], summary["log_loss"], color="#4C78A8")
        ax.invert_yaxis()
        ax.set_xlabel("Mean validation log-loss")
        ax.set_ylabel("")
        ax.set_title("DBRW Window Ablation")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "window_ablation_logloss.png", dpi=180)
        plt.close(fig)

        label_summary = results.pivot_table(index="label", columns="feature_set", values="log_loss")
        ordered_cols = [c for c in summary["feature_set"] if c in label_summary.columns]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        label_summary[ordered_cols].plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Validation log-loss")
        ax.set_xlabel("Label")
        ax.set_title("Label-wise Window Comparison")
        ax.legend(fontsize=8, ncols=2)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "labelwise_window_logloss.png", dpi=180)
        plt.close(fig)

    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        importance["Category"] = importance["Feature"].apply(categorize_feature)
        grouped = (
            importance.groupby("Category", as_index=False)["Importance"]
            .sum()
            .sort_values("Importance")
        )
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(grouped["Category"], grouped["Importance"], color="#F58518")
        ax.set_xlabel("Cumulative ExtraTrees importance")
        ax.set_ylabel("")
        ax.set_title("Feature Importance by Sensor Family")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "feature_group_importance.png", dpi=180)
        plt.close(fig)

    print(f"Saved README figures to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DBRW ExtraTrees reproduction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-windows", help="re-run the DBRW window ablation")
    validate.add_argument("--data-dir", type=Path, default=Path("data"))
    validate.add_argument("--output-dir", type=Path, default=Path("outputs"))
    validate.add_argument("--train-file", default="feature_table_train.csv")
    validate.add_argument("--keep-personal", action="store_true")
    validate.set_defaults(func=command_validate_windows)

    submit = subparsers.add_parser("make-submission", help="train final models and write DACON submission CSVs")
    submit.add_argument("--data-dir", type=Path, default=Path("data"))
    submit.add_argument("--output-dir", type=Path, default=Path("outputs"))
    submit.add_argument("--train-file", default="feature_table_train.csv")
    submit.add_argument("--test-file", default="feature_table_test.csv")
    submit.add_argument("--sample-file", default="ch2026_submission_sample.csv")
    submit.add_argument("--window-results", type=Path, default=Path("outputs/window_feature_validation_results.csv"))
    submit.add_argument("--keep-personal", action="store_true")
    submit.set_defaults(func=command_make_submission)

    importance = subparsers.add_parser("feature-importance", help="compute top ExtraTrees feature importances")
    importance.add_argument("--data-dir", type=Path, default=Path("data"))
    importance.add_argument("--output-dir", type=Path, default=Path("outputs"))
    importance.add_argument("--train-file", default="feature_table_train.csv")
    importance.add_argument("--window-results", type=Path, default=Path("outputs/window_feature_validation_results.csv"))
    importance.add_argument("--top-n", type=int, default=30)
    importance.add_argument("--keep-personal", action="store_true")
    importance.set_defaults(func=command_feature_importance)

    figures = subparsers.add_parser("make-readme-figures", help="make compact figures from saved result CSVs")
    figures.add_argument("--results-dir", type=Path, default=Path("results"))
    figures.add_argument("--output-dir", type=Path, default=Path("remade"))
    figures.set_defaults(func=command_make_readme_figures)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
