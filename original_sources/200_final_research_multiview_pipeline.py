# %% [markdown]
# # Final Research Multi-View Personalized Pipeline
#
# Purpose
# - Build a paper-grade final experiment for the ETRI lifelog challenge.
# - Use multiple legitimate feature views rather than public-score-only tweaking.
# - Model personalized digital bedtime routines with late fusion, subject temporal priors,
#   and validation-only calibration.
#
# Design principles
# - No test labels, public labels, or leaderboard target fitting are used.
# - Feature selection is performed inside each validation fold using only training rows.
# - Calibration is selected from out-of-fold validation predictions.
# - The final submission is produced from models trained on the full training set.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# %% [markdown]
# ## 1. Configuration

# %%
PROJECT_DIR = Path(__file__).resolve().parent
TEMP_DIR = Path(r"C:\Users\jungheeyoung\AppData\Local\Temp")

LABELS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLS = ["subject_id", "sleep_date", "lifelog_date", "split"]

SAMPLE_PATHS = [
    PROJECT_DIR / "ch2026_submission_sample.csv",
    TEMP_DIR / "ch2026_submission_sample.csv",
]

VIEWS = {
    "routine_multiday": (
        PROJECT_DIR / "feature_table_train_publicstyle_multiday.csv",
        PROJECT_DIR / "feature_table_test_publicstyle_multiday.csv",
        450,
    ),
    "hidden_hr": (
        PROJECT_DIR / "feature_table_train_hidden_hr.csv",
        PROJECT_DIR / "feature_table_test_hidden_hr.csv",
        520,
    ),
    "sequence_routine": (
        PROJECT_DIR / "feature_table_train_sequence_routine.csv",
        PROJECT_DIR / "feature_table_test_sequence_routine.csv",
        520,
    ),
    "raw_hourly": (
        PROJECT_DIR / "feature_table_train_raw_hourly.csv",
        PROJECT_DIR / "feature_table_test_raw_hourly.csv",
        560,
    ),
}

RESULT_PATH = PROJECT_DIR / "final_research_multiview_results.csv"
SELECTION_PATH = PROJECT_DIR / "final_research_multiview_selection.csv"
SUBMISSION_PATH = PROJECT_DIR / "submission_final_research_multiview.csv"
REPORT_PATH = PROJECT_DIR / "final_research_multiview_report.md"

RANDOM_STATE = 200
CLIP_MIN = 0.015
CLIP_MAX = 0.985


# %% [markdown]
# ## 2. Utilities

# %%
def find_sample_path() -> Path:
    for path in SAMPLE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find ch2026_submission_sample.csv")


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sleep_date"] = pd.to_datetime(df["sleep_date"])
    df["lifelog_date"] = pd.to_datetime(df["lifelog_date"])
    return df


def numeric_cols(df: pd.DataFrame) -> list[str]:
    protected = set(ID_COLS + LABELS)
    cols = []
    for col in df.columns:
        if col in protected:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def add_common_calendar_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train2 = train.copy()
    test2 = test.copy()
    base_date = min(train2["lifelog_date"].min(), test2["lifelog_date"].min())
    subjects = sorted(set(train2["subject_id"]) | set(test2["subject_id"]))
    for df in [train2, test2]:
        df["lifelog_dow"] = df["lifelog_date"].dt.dayofweek
        df["sleep_dow"] = df["sleep_date"].dt.dayofweek
        df["is_weekend"] = (df["lifelog_dow"] >= 5).astype(int)
        df["days_since_start"] = (df["lifelog_date"] - base_date).dt.days
        for sid in subjects:
            df[f"subject_{sid}"] = (df["subject_id"] == sid).astype(int)
    return train2, test2


def make_subject_time_folds(train: pd.DataFrame, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological folds inside each subject.

    Fold 0 gets the earliest block for each subject, fold n-1 the latest block.
    This tests whether models transfer across dates while preserving all subjects.
    """
    fold_id = np.full(len(train), -1, dtype=int)
    ordered = train.sort_values(["subject_id", "lifelog_date"])
    for _, group in ordered.groupby("subject_id", sort=True):
        idx = group.index.to_numpy()
        parts = np.array_split(idx, n_folds)
        for f, part in enumerate(parts):
            fold_id[part] = f
    all_idx = np.arange(len(train))
    return [(all_idx[fold_id != f], all_idx[fold_id == f]) for f in range(n_folds)]


def avg_logloss(y_true: pd.DataFrame, pred: pd.DataFrame) -> tuple[float, dict[str, float]]:
    losses = {}
    for label in LABELS:
        losses[label] = float(
            log_loss(
                y_true[label].astype(int),
                np.clip(pred[label].astype(float), 1e-6, 1 - 1e-6),
                labels=[0, 1],
            )
        )
    return float(np.mean(list(losses.values()))), losses


def select_features_fold(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    tr_idx: np.ndarray,
    label: str,
    top_k: int,
) -> list[str]:
    X = train_df.loc[tr_idx, feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_ratio = X.notna().mean()
    variance = X.var(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    usable = [c for c in feature_cols if valid_ratio[c] >= 0.50 and variance[c] > 1e-10]
    if len(usable) <= top_k:
        return usable

    X_use = X[usable]
    y = train_df.loc[tr_idx, label].astype(int)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_use)
    scores, _ = f_classif(X_imp, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(scores)[::-1]
    return [usable[i] for i in order[:top_k]]


def fit_predict_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    label: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_tr = train_df.loc[tr_idx, cols].replace([np.inf, -np.inf], np.nan)
    X_va = train_df.loc[va_idx, cols].replace([np.inf, -np.inf], np.nan)
    X_te = test_df[cols].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(imputer.fit_transform(X_tr))
    X_va_s = scaler.transform(imputer.transform(X_va))
    X_te_s = scaler.transform(imputer.transform(X_te))

    y_tr = train_df.loc[tr_idx, label].astype(int).to_numpy()
    pos_rate = float(np.clip(y_tr.mean(), 0.05, 0.95))
    scale_pos_weight = float(np.clip((1.0 - pos_rate) / pos_rate, 0.55, 1.85))

    model = LGBMClassifier(
        n_estimators=260,
        learning_rate=0.018,
        num_leaves=7,
        max_depth=3,
        min_child_samples=16,
        subsample=0.82,
        colsample_bytree=0.70,
        reg_alpha=1.4,
        reg_lambda=12.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_tr_s, y_tr)
    return model.predict_proba(X_va_s)[:, 1], model.predict_proba(X_te_s)[:, 1]


def train_full_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    top_k: int,
    seed: int,
) -> np.ndarray:
    all_idx = np.arange(len(train_df))
    cols = select_features_fold(train_df, feature_cols, all_idx, label, top_k)
    X = train_df[cols].replace([np.inf, -np.inf], np.nan)
    X_te = test_df[cols].replace([np.inf, -np.inf], np.nan)
    y = train_df[label].astype(int).to_numpy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(imputer.fit_transform(X))
    X_te_s = scaler.transform(imputer.transform(X_te))

    pos_rate = float(np.clip(y.mean(), 0.05, 0.95))
    scale_pos_weight = float(np.clip((1.0 - pos_rate) / pos_rate, 0.55, 1.85))
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.016,
        num_leaves=7,
        max_depth=3,
        min_child_samples=14,
        subsample=0.84,
        colsample_bytree=0.72,
        reg_alpha=1.2,
        reg_lambda=11.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_s, y)
    return model.predict_proba(X_te_s)[:, 1]


def temporal_prior(train_known: pd.DataFrame, target: pd.DataFrame, tau: float) -> pd.DataFrame:
    global_rate = train_known[LABELS].mean().clip(CLIP_MIN, CLIP_MAX)
    rows = []
    for _, row in target.iterrows():
        hist = train_known[train_known["subject_id"].eq(row["subject_id"])]
        if hist.empty:
            rows.append(global_rate.to_dict())
            continue
        gap = np.abs((hist["sleep_date"] - row["sleep_date"]).dt.days.to_numpy(float))
        weights = np.exp(-gap / tau)
        weights *= np.where(gap <= 7, 1.8, 1.0)
        weights *= np.where(hist["sleep_date"].dt.dayofweek.to_numpy() == row["sleep_date"].dayofweek, 1.12, 1.0)
        pred = {}
        subj_rate = hist[LABELS].mean().clip(CLIP_MIN, CLIP_MAX)
        for label in LABELS:
            local = np.average(hist[label].to_numpy(float), weights=np.maximum(weights, 1e-8))
            pred[label] = float(np.clip(0.72 * local + 0.18 * subj_rate[label] + 0.10 * global_rate[label], CLIP_MIN, CLIP_MAX))
        rows.append(pred)
    return pd.DataFrame(rows, index=target.index)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(p: np.ndarray, inv_temp: float) -> np.ndarray:
    return sigmoid(logit(p) * inv_temp)


def weight_grid(n: int) -> list[np.ndarray]:
    if n == 1:
        return [np.array([1.0])]
    candidates: list[np.ndarray] = []
    # Conservative weights: no view can dominate too extremely unless it is used alone.
    for parts in itertools.product([0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0], repeat=n):
        w = np.array(parts, dtype=float)
        if abs(w.sum() - 1.0) <= 1e-9:
            candidates.append(w)
    candidates.extend([np.eye(n)[i] for i in range(n)])
    candidates.append(np.ones(n) / n)
    unique = []
    seen = set()
    for w in candidates:
        key = tuple(np.round(w, 4))
        if key not in seen:
            unique.append(w)
            seen.add(key)
    return unique


# %% [markdown]
# ## 3. Multi-view OOF training

# %%
def main() -> None:
    sample_path = find_sample_path()
    sample = pd.read_csv(sample_path)

    view_data = {}
    base_train = None
    base_test = None
    for view_name, (train_path, test_path, top_k) in VIEWS.items():
        print(f"[load] {view_name}")
        tr = read_table(train_path)
        te = read_table(test_path)
        tr, te = add_common_calendar_features(tr, te)
        view_data[view_name] = {
            "train": tr,
            "test": te,
            "top_k": top_k,
            "feature_cols": numeric_cols(tr),
        }
        if base_train is None:
            base_train = tr[["subject_id", "sleep_date", "lifelog_date"] + LABELS].copy()
            base_test = te[["subject_id", "sleep_date", "lifelog_date"]].copy()

    assert base_train is not None and base_test is not None
    folds = make_subject_time_folds(base_train, n_folds=5)

    oof_view = {
        view_name: pd.DataFrame(index=base_train.index, columns=LABELS, dtype=float)
        for view_name in view_data
    }
    test_view_folds = {
        view_name: {label: [] for label in LABELS}
        for view_name in view_data
    }
    cv_rows = []

    for view_name, data in view_data.items():
        tr = data["train"]
        te = data["test"]
        feature_cols = data["feature_cols"]
        top_k = data["top_k"]
        print(f"\n[view] {view_name}: {len(feature_cols)} numeric features")

        for label in LABELS:
            fold_losses = []
            for fold_id, (tr_idx, va_idx) in enumerate(folds):
                cols = select_features_fold(tr, feature_cols, tr_idx, label, top_k)
                va_pred, te_pred = fit_predict_lgbm(
                    tr, te, cols, tr_idx, va_idx, label, seed=RANDOM_STATE + fold_id
                )
                va_pred = np.clip(va_pred, CLIP_MIN, CLIP_MAX)
                te_pred = np.clip(te_pred, CLIP_MIN, CLIP_MAX)
                oof_view[view_name].loc[va_idx, label] = va_pred
                test_view_folds[view_name][label].append(te_pred)
                loss = log_loss(base_train.loc[va_idx, label].astype(int), va_pred, labels=[0, 1])
                fold_losses.append(float(loss))
                print(f"  {label} fold {fold_id}: {loss:.5f} ({len(cols)} cols)")
            cv_rows.append(
                {
                    "view": view_name,
                    "label": label,
                    "mean_log_loss": float(np.mean(fold_losses)),
                    "std_log_loss": float(np.std(fold_losses)),
                    "max_log_loss": float(np.max(fold_losses)),
                }
            )

    # Train full models for final test predictions. We keep this separate from OOF.
    test_view_full = {
        view_name: pd.DataFrame(index=base_test.index, columns=LABELS, dtype=float)
        for view_name in view_data
    }
    for view_name, data in view_data.items():
        print(f"\n[full train] {view_name}")
        tr = data["train"]
        te = data["test"]
        for j, label in enumerate(LABELS):
            pred = train_full_lgbm(
                tr,
                te,
                data["feature_cols"],
                label,
                data["top_k"],
                seed=RANDOM_STATE + 100 + j,
            )
            test_view_full[view_name][label] = np.clip(pred, CLIP_MIN, CLIP_MAX)

    # Temporal prior OOF and test prior.
    prior_oof = pd.DataFrame(index=base_train.index, columns=LABELS, dtype=float)
    for tr_idx, va_idx in folds:
        prior_oof.loc[va_idx, LABELS] = temporal_prior(
            base_train.iloc[tr_idx], base_train.iloc[va_idx], tau=45.0
        )[LABELS].to_numpy()
    prior_test = temporal_prior(base_train, base_test, tau=45.0)

    # Late fusion + prior blend + temperature selected by OOF only.
    view_names = list(view_data.keys())
    selection_rows = []
    fused_oof = pd.DataFrame(index=base_train.index, columns=LABELS, dtype=float)
    fused_test = pd.DataFrame(index=base_test.index, columns=LABELS, dtype=float)

    for label in LABELS:
        best = None
        y = base_train[label].astype(int)
        view_oof_stack = np.vstack([oof_view[v][label].to_numpy(float) for v in view_names])
        view_test_stack = np.vstack([test_view_full[v][label].to_numpy(float) for v in view_names])

        for w in weight_grid(len(view_names)):
            base_p = np.average(view_oof_stack, axis=0, weights=w)
            base_t = np.average(view_test_stack, axis=0, weights=w)
            for prior_w in [0.00, 0.05, 0.10, 0.16, 0.22, 0.30, 0.38]:
                p = (1.0 - prior_w) * base_p + prior_w * prior_oof[label].to_numpy(float)
                t = (1.0 - prior_w) * base_t + prior_w * prior_test[label].to_numpy(float)
                for inv_temp in [0.90, 0.96, 1.00, 1.04, 1.08, 1.12]:
                    p2 = np.clip(apply_temperature(p, inv_temp), CLIP_MIN, CLIP_MAX)
                    loss = float(log_loss(y, p2, labels=[0, 1]))
                    if best is None or loss < best["loss"]:
                        best = {
                            "loss": loss,
                            "weights": w.copy(),
                            "prior_w": prior_w,
                            "inv_temp": inv_temp,
                            "test_pred": np.clip(apply_temperature(t, inv_temp), CLIP_MIN, CLIP_MAX),
                            "oof_pred": p2,
                        }

        assert best is not None
        fused_oof[label] = best["oof_pred"]
        fused_test[label] = best["test_pred"]
        row = {
            "label": label,
            "oof_log_loss": best["loss"],
            "prior_weight": best["prior_w"],
            "inv_temperature": best["inv_temp"],
        }
        for view_name, weight in zip(view_names, best["weights"]):
            row[f"w_{view_name}"] = float(weight)
        selection_rows.append(row)

    mean_loss, label_losses = avg_logloss(base_train[LABELS], fused_oof[LABELS])
    print("\n[fused OOF]")
    print("mean_log_loss", mean_loss)
    print(label_losses)

    # Save results.
    pd.DataFrame(cv_rows).to_csv(RESULT_PATH, index=False)
    selection = pd.DataFrame(selection_rows)
    selection.loc[len(selection)] = {
        "label": "MEAN",
        "oof_log_loss": mean_loss,
        "prior_weight": np.nan,
        "inv_temperature": np.nan,
        **{f"w_{v}": np.nan for v in view_names},
    }
    selection.to_csv(SELECTION_PATH, index=False)

    out = sample.copy()
    for c in ["sleep_date", "lifelog_date"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    for label in LABELS:
        out[label] = fused_test[label].to_numpy(float)
    out.to_csv(SUBMISSION_PATH, index=False)

    report = [
        "# Final Research Multi-View Personalized Pipeline",
        "",
        f"- Mean OOF average log-loss: `{mean_loss:.6f}`",
        f"- Submission file: `{SUBMISSION_PATH.name}`",
        "- Validation: chronological subject-wise 5-fold split.",
        "- Calibration: selected only from out-of-fold predictions.",
        "- Views: routine_multiday, hidden_hr, sequence_routine, raw_hourly.",
        "",
        "## Label-wise OOF log-loss",
        "",
    ]
    for label, loss in label_losses.items():
        report.append(f"- `{label}`: `{loss:.6f}`")
    report.extend(["", "## Selected fusion weights", "", selection.to_markdown(index=False)])
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nSaved:")
    print(" ", RESULT_PATH)
    print(" ", SELECTION_PATH)
    print(" ", SUBMISSION_PATH)
    print(" ", REPORT_PATH)


if __name__ == "__main__":
    main()
