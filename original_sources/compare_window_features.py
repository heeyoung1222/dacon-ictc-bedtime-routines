# %% [markdown]
# # 취침 전 행동 모티프 시간 구간별 feature 비교
#
# 목적:
# - 모델은 `ExtraTreesShrinkage`로 고정함
# - feature 구간만 바꾸어 Average Log-Loss가 어떻게 달라지는지 확인함
# - 연구 질문인 "취침 전 행동 모티프의 시간 구간을 다르게 설정했을 때 예측 성능이 어떻게 변하는가?"를 검증함
#
# 비교 구간:
# - day only: 하루 전체 요약 feature만 사용
# - day + 18~24
# - day + 20~24
# - day + 21~24
# - day + 22~24
# - day + 20~02
# - day + all windows
#
# 검증 방식:
# - subject별 날짜순 앞 80% train, 뒤 20% validation
# - calibration alpha는 validation 정답을 사용하지 않고 train 내부 시간순 calibration fold에서만 선택함

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline


# %% [markdown]
# ## 1. 설정

# %%
PROJECT_DIR = Path(r"C:\Users\jungheeyoung\Desktop\데이콘")
TRAIN_FEATURE_PATH = PROJECT_DIR / "feature_table_train.csv"

RESULT_PATH = PROJECT_DIR / "window_feature_validation_results.csv"
PRED_PATH = PROJECT_DIR / "window_feature_validation_predictions.csv"
SUMMARY_PATH = PROJECT_DIR / "window_feature_validation_summary.md"

LABELS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLS = ["subject_id", "sleep_date", "lifelog_date", "split"]
WINDOWS = ["w18_24", "w20_24", "w21_24", "w22_24", "w20_02"]
RANDOM_STATE = 42


# %% [markdown]
# ## 2. 데이터 로드 및 time-based validation split

# %%
def load_train_features() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_FEATURE_PATH)
    df["sleep_date"] = pd.to_datetime(df["sleep_date"])
    df["lifelog_date"] = pd.to_datetime(df["lifelog_date"])
    return df


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
    train_idx: np.ndarray,
    valid_ratio: float = 0.2,
    min_cal_per_subject: int = 2,
) -> np.ndarray:
    fold = pd.Series(-1, index=train_idx, dtype=int)
    train_df = df.loc[train_idx].sort_values("lifelog_date")
    for _, group in train_df.groupby("subject_id", sort=True):
        group = group.sort_values("lifelog_date")
        n = len(group)
        n_cal = max(min_cal_per_subject, int(np.ceil(n * valid_ratio)))
        n_cal = min(n_cal, max(1, n - 1))
        fold.loc[group.index[-n_cal:]] = 0
    return fold.loc[train_idx].to_numpy()


# %% [markdown]
# ## 3. 시간 구간별 feature set 구성
#
# `personal_*` feature는 전체 날짜 분포를 반영할 가능성이 있어 이번 구간 비교에서는 제외함.

# %%
def has_window_token(col: str, window: str) -> bool:
    return f"_{window}_" in col or col.startswith(f"{window}_")


def has_any_window_token(col: str) -> bool:
    return any(has_window_token(col, window) for window in WINDOWS)


def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    protected = set(ID_COLS + LABELS)
    numeric_cols = [
        col
        for col in df.columns
        if col not in protected and pd.api.types.is_numeric_dtype(df[col])
    ]
    clean_cols = [col for col in numeric_cols if not col.startswith("personal_")]
    base_cols = [col for col in clean_cols if not has_any_window_token(col)]

    feature_sets = {"day_only": base_cols}
    for window in WINDOWS:
        window_cols = [col for col in clean_cols if has_window_token(col, window)]
        feature_sets[f"day_plus_{window}"] = base_cols + window_cols
    feature_sets["day_plus_all_windows"] = clean_cols
    return feature_sets


# %% [markdown]
# ## 4. 고정 모델: ExtraTreesShrinkage
#
# ExtraTrees 예측 확률을 그대로 사용하지 않고, train 내부 calibration fold에서 선택한 alpha로 확률을 완만하게 보정함.
# alpha 범위는 0.5~1.0으로 제한하여 과도한 보정을 방지함.

# %%
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
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        return proba[:, list(classes).index(1)]
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


# %% [markdown]
# ## 5. 평가 실행

# %%
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


# %% [markdown]
# ## 6. 결과 저장

# %%
def write_summary(
    results: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> None:
    summary = (
        results.groupby("feature_set", as_index=False)
        [["log_loss", "accuracy", "macro_f1", "roc_auc"]]
        .mean()
        .sort_values("log_loss")
    )
    label_summary = (
        results.pivot_table(
            index="label",
            columns="feature_set",
            values="log_loss",
            aggfunc="mean",
        )
        .reset_index()
    )

    lines = []
    lines.append("# 취침 전 행동 모티프 시간 구간별 feature 비교 결과")
    lines.append("")
    lines.append("## 검증 방식")
    lines.append("")
    lines.append("- 모델은 ExtraTreesShrinkage로 고정함")
    lines.append("- 각 subject별 날짜순 앞 80% train, 뒤 20% validation을 사용함")
    lines.append("- calibration alpha는 train 내부 시간순 calibration fold에서만 선택함")
    lines.append(f"- train rows: {len(train_idx)}")
    lines.append(f"- validation rows: {len(valid_idx)}")
    lines.append("")
    lines.append("## Feature set별 평균 성능")
    lines.append("")
    lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 라벨별 Log-Loss")
    lines.append("")
    lines.append(label_summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Feature 수")
    lines.append("")
    for name, cols in feature_sets.items():
        lines.append(f"- {name}: {len(cols)}개")
    SUMMARY_PATH.write_text("\\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_train_features()
    train_idx, valid_idx = make_subject_time_split(df)
    feature_sets = get_feature_sets(df)
    results, predictions = evaluate_feature_sets(df, feature_sets, train_idx, valid_idx)
    results.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    predictions.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")
    write_summary(results, feature_sets, train_idx, valid_idx)

    summary = (
        results.groupby("feature_set", as_index=False)
        [["log_loss", "accuracy", "macro_f1", "roc_auc"]]
        .mean()
        .sort_values("log_loss")
    )
    print("saved:")
    print(f"  {RESULT_PATH}")
    print(f"  {PRED_PATH}")
    print(f"  {SUMMARY_PATH}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
