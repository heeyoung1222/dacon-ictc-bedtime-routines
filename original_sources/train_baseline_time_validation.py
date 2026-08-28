# %% [markdown]
# # 데이콘 라이프로그 대회 baseline 모델링
#
# 목적:
# - `feature_table_train.csv`를 사용하여 실제 일반화 성능에 가까운 baseline을 확인함
# - 랜덤 분할이 아니라 subject별 날짜 순서를 유지한 time-based validation을 사용함
# - 대회 평가 지표인 Average Log-Loss를 중심으로 모델을 비교함
# - 이후 "취침 전 행동 모티프 시간 구간별 성능 비교" 실험의 기준점으로 활용함
#
# 주의:
# - 현재 feature table에는 개인별 평균/표준편차 기반 `personal_*` feature가 포함되어 있음
# - 이 feature는 라벨을 쓰지는 않지만, 전체 날짜 분포를 사용해 만든 값이므로 validation에서는 약한 미래 정보 누수가 될 수 있음
# - 따라서 1차 baseline은 `personal_*` feature를 제외한 clean feature set을 주 기준으로 사용함
# - 보조적으로 `personal_*`를 포함한 결과도 함께 계산하여 차이를 확인함

# %%
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# %% [markdown]
# ## 1. 경로 및 기본 설정

# %%
PROJECT_DIR = Path(r"C:\Users\jungheeyoung\Desktop\데이콘")
TRAIN_FEATURE_PATH = PROJECT_DIR / "feature_table_train.csv"

RESULT_PATH = PROJECT_DIR / "baseline_time_validation_results.csv"
PRED_PATH = PROJECT_DIR / "baseline_time_validation_predictions.csv"
SUMMARY_PATH = PROJECT_DIR / "baseline_time_validation_summary.md"

LABELS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLS = ["subject_id", "sleep_date", "lifelog_date", "split"]
RANDOM_STATE = 42


# %% [markdown]
# ## 2. 데이터 로드
#
# - 한 행은 `subject_id + lifelog_date` 단위임
# - 라벨은 다음날 `sleep_date`의 수면·감정·스트레스 지표임

# %%
def load_train_features() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_FEATURE_PATH)
    df["sleep_date"] = pd.to_datetime(df["sleep_date"])
    df["lifelog_date"] = pd.to_datetime(df["lifelog_date"])
    return df


# %% [markdown]
# ## 3. Subject별 time-based validation 분할
#
# 설계:
# - 각 subject의 데이터를 `lifelog_date` 기준으로 정렬함
# - 앞쪽 80%는 train, 뒤쪽 20%는 validation으로 사용함
# - 미래 날짜를 맞히는 상황에 가까워 랜덤 분할보다 보수적인 검증임

# %%
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
        valid_idx = group.index[-n_valid:].tolist()
        train_idx = group.index[:-n_valid].tolist()
        train_indices.extend(train_idx)
        valid_indices.extend(valid_idx)

    return np.array(train_indices), np.array(valid_indices)


# %% [markdown]
# ## 4. Feature set 구성
#
# 두 가지 feature set을 비교함.
#
# - `clean_no_personal`: 미래 분포 누수 가능성을 줄이기 위해 `personal_*` feature 제외
# - `all_features`: 현재 feature table의 모든 numeric feature 사용

# %%
def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    protected = set(ID_COLS + LABELS)
    numeric_features = [
        col
        for col in df.columns
        if col not in protected and pd.api.types.is_numeric_dtype(df[col])
    ]

    clean_features = [col for col in numeric_features if not col.startswith("personal_")]

    return {
        "clean_no_personal": clean_features,
        "all_features": numeric_features,
    }


# %% [markdown]
# ## 5. 모델 정의
#
# 현재 환경에서 별도 설치 없이 실행 가능한 모델만 사용함.
#
# - `DummyMostFrequent`: 다수 클래스 기준선
# - `LogisticL2`: 작은 데이터에서 비교적 안정적인 선형 baseline
# - `ExtraTrees`: 비선형 feature interaction을 잡는 트리 앙상블
# - `HistGradientBoosting`: sklearn 기반 gradient boosting baseline
# - `ExtraTreesCalibratedSigmoid`: outer train 내부의 시간순 calibration fold만 사용하여 sigmoid 확률 보정
# - `ExtraTreesCalibratedIsotonic`: 같은 방식의 isotonic 확률 보정, 데이터가 작으면 과적합 가능성이 있어 보조 실험으로만 해석함
# - `ExtraTreesShrinkage`: inner calibration fold에서만 선택한 alpha로 예측 확률을 train prior 쪽으로 완만하게 당김

# %%
@dataclass
class ModelSpec:
    name: str
    estimator: object
    calibration_method: str | None = None


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


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "DummyMostFrequent",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", DummyClassifier(strategy="most_frequent")),
                ]
            ),
        ),
        ModelSpec(
            "LogisticL2",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.2,
                            penalty="l2",
                            solver="liblinear",
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            max_iter=2000,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec("ExtraTrees", extra_trees_estimator()),
        ModelSpec(
            "HistGradientBoosting",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.03,
                            max_iter=180,
                            max_leaf_nodes=12,
                            l2_regularization=0.2,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "ExtraTreesCalibratedSigmoid",
            extra_trees_estimator(),
            calibration_method="sigmoid",
        ),
        ModelSpec(
            "ExtraTreesCalibratedIsotonic",
            extra_trees_estimator(),
            calibration_method="isotonic",
        ),
        ModelSpec(
            "ExtraTreesShrinkage",
            extra_trees_estimator(),
            calibration_method="shrinkage",
        ),
    ]


def make_inner_calibration_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    valid_ratio: float = 0.2,
    min_cal_per_subject: int = 2,
) -> np.ndarray:
    """Return PredefinedSplit fold ids for the outer train rows.

    -1 means inner model training row.
    0 means calibration row.
    The split is made only inside the outer train rows and follows time order.
    """
    fold = pd.Series(-1, index=train_idx, dtype=int)
    train_df = df.loc[train_idx].sort_values("lifelog_date")
    for _, group in train_df.groupby("subject_id", sort=True):
        group = group.sort_values("lifelog_date")
        n = len(group)
        n_cal = max(min_cal_per_subject, int(np.ceil(n * valid_ratio)))
        n_cal = min(n_cal, max(1, n - 1))
        cal_idx = group.index[-n_cal:]
        fold.loc[cal_idx] = 0
    return fold.loc[train_idx].to_numpy()


def can_use_calibration(y: pd.Series, fold: np.ndarray) -> bool:
    inner_train_classes = y[fold == -1].nunique()
    calibration_classes = y[fold == 0].nunique()
    return inner_train_classes == 2 and calibration_classes == 2


def choose_shrinkage_alpha(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    prior: float,
) -> float:
    best_alpha = 1.0
    best_loss = np.inf
    # Conservative range: avoid forcing probabilities too close to the class prior.
    for alpha in np.linspace(0.5, 1.0, 11):
        p_adj = alpha * p_cal + (1.0 - alpha) * prior
        loss = log_loss(y_cal, np.clip(p_adj, 1e-15, 1 - 1e-15), labels=[0, 1])
        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)
    return best_alpha


# %% [markdown]
# ## 6. 평가 함수
#
# 각 라벨별로 독립적인 이진 분류 모델을 학습함.
# Accuracy만 보면 클래스 불균형에 둔감할 수 있으므로 F1, balanced accuracy, ROC-AUC도 함께 확인함.

# %%
def safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_proba))


def get_positive_probability(model: object, x_valid: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_valid)
        classes = getattr(model, "classes_", np.array([0, 1]))
        if 1 in classes:
            return proba[:, list(classes).index(1)]
        return np.zeros(len(x_valid))
    if hasattr(model, "decision_function"):
        score = model.decision_function(x_valid)
        return 1 / (1 + np.exp(-score))
    pred = model.predict(x_valid)
    return pred.astype(float)


@dataclass
class EvalOutput:
    results: pd.DataFrame
    predictions: pd.DataFrame


def evaluate_models(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> EvalOutput:
    model_specs = build_model_specs()
    calibration_fold = make_inner_calibration_fold(df, train_idx)
    result_rows = []
    pred_frames = []

    for feature_set_name, feature_cols in feature_sets.items():
        x_train = df.loc[train_idx, feature_cols]
        x_valid = df.loc[valid_idx, feature_cols]

        for spec in model_specs:
            for label in LABELS:
                y_train = df.loc[train_idx, label].astype(int)
                y_valid = df.loc[valid_idx, label].astype(int)
                alpha = np.nan

                if spec.calibration_method == "shrinkage":
                    inner_train_mask = calibration_fold == -1
                    inner_cal_mask = calibration_fold == 0
                    inner_model = clone(spec.estimator)
                    inner_model.fit(x_train.iloc[inner_train_mask], y_train.iloc[inner_train_mask])
                    p_cal = get_positive_probability(inner_model, x_train.iloc[inner_cal_mask])
                    y_cal = y_train.iloc[inner_cal_mask].to_numpy()
                    prior = float(y_train.iloc[inner_train_mask].mean())
                    alpha = choose_shrinkage_alpha(p_cal, y_cal, prior)

                    model = clone(spec.estimator)
                    model.fit(x_train, y_train)
                    raw_proba = get_positive_probability(model, x_valid)
                    final_prior = float(y_train.mean())
                    y_proba = alpha * raw_proba + (1.0 - alpha) * final_prior
                    y_pred = (y_proba >= 0.5).astype(int)

                elif spec.calibration_method is not None:
                    if not can_use_calibration(y_train.reset_index(drop=True), calibration_fold):
                        print(
                            f"skip {spec.name} / {label}: calibration fold does not contain both classes"
                        )
                        continue
                    model = CalibratedClassifierCV(
                        estimator=clone(spec.estimator),
                        method=spec.calibration_method,
                        cv=PredefinedSplit(calibration_fold),
                        ensemble=True,
                    )
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_valid).astype(int)
                    y_proba = get_positive_probability(model, x_valid)

                else:
                    model = clone(spec.estimator)
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_valid).astype(int)
                    y_proba = get_positive_probability(model, x_valid)

                result_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "model": spec.name,
                        "calibration": spec.calibration_method or "none",
                        "shrinkage_alpha": alpha,
                        "label": label,
                        "n_train": len(train_idx),
                        "n_valid": len(valid_idx),
                        "n_features": len(feature_cols),
                        "accuracy": accuracy_score(y_valid, y_pred),
                        "balanced_accuracy": balanced_accuracy_score(y_valid, y_pred),
                        "macro_f1": f1_score(y_valid, y_pred, average="macro", zero_division=0),
                        "roc_auc": safe_auc(y_valid.to_numpy(), y_proba),
                        "log_loss": log_loss(
                            y_valid,
                            np.clip(y_proba, 1e-15, 1 - 1e-15),
                            labels=[0, 1],
                        ),
                        "valid_positive_rate": float(y_valid.mean()),
                        "pred_positive_rate": float(y_pred.mean()),
                        "pred_proba_mean": float(np.mean(y_proba)),
                    }
                )

                pred_frame = df.loc[
                    valid_idx,
                    ["subject_id", "sleep_date", "lifelog_date"],
                ].copy()
                pred_frame["feature_set"] = feature_set_name
                pred_frame["model"] = spec.name
                pred_frame["calibration"] = spec.calibration_method or "none"
                pred_frame["label"] = label
                pred_frame["y_true"] = y_valid.to_numpy()
                pred_frame["y_pred"] = y_pred
                pred_frame["y_proba"] = y_proba
                pred_frames.append(pred_frame)

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    return EvalOutput(results=results, predictions=predictions)


# %% [markdown]
# ## 7. 결과 요약 저장
#
# 모델별 전체 평균 성능과 라벨별 성능을 저장함.

# %%
def write_summary(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    results: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> None:
    summary = (
        results.groupby(["feature_set", "model"], as_index=False)
        [["log_loss", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]]
        .mean()
        .sort_values(["log_loss", "macro_f1"], ascending=[True, False])
    )
    label_summary = (
        results.groupby(["feature_set", "model", "label"], as_index=False)
        [["log_loss", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]]
        .mean()
        .sort_values(["feature_set", "model", "label"])
    )

    lines = []
    lines.append("# Baseline time-based validation 결과")
    lines.append("")
    lines.append("## 검증 방식")
    lines.append("")
    lines.append("- 각 subject별 날짜순 앞 80% train, 뒤 20% validation으로 분할함")
    lines.append("- 랜덤 분할보다 실제 미래 날짜 예측 상황에 가까운 검증 방식임")
    lines.append("- 대회 공식 평가 지표인 Average Log-Loss를 1순위 기준으로 확인함")
    lines.append("- Log-Loss는 낮을수록 좋은 점수임")
    lines.append(f"- train rows: {len(train_idx)}")
    lines.append(f"- validation rows: {len(valid_idx)}")
    lines.append("")
    lines.append("## Feature set")
    lines.append("")
    for name, cols in feature_sets.items():
        lines.append(f"- {name}: {len(cols)}개 feature")
    lines.append("")
    lines.append("## 모델별 평균 성능")
    lines.append("")
    lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 라벨별 성능")
    lines.append("")
    lines.append(label_summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 향후 실험 메모")
    lines.append("")
    lines.append("- 다음 feature 확장에서는 취침 전 행동 모티프 시간 구간을 18~24, 20~24, 21~24, 22~24, 20~02 등으로 나누어 성능 변화를 비교할 계획임")
    lines.append("- 핵심 연구 질문: 취침 전 행동 모티프의 시간 구간을 다르게 설정했을 때 예측 성능이 어떻게 변하는가?")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


# %% [markdown]
# ## 8. 실행

# %%
def main() -> None:
    df = load_train_features()
    train_idx, valid_idx = make_subject_time_split(df)
    feature_sets = get_feature_sets(df)

    output = evaluate_models(df, feature_sets, train_idx, valid_idx)
    output.results.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    output.predictions.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")
    write_summary(df, train_idx, valid_idx, output.results, feature_sets)

    summary = (
        output.results.groupby(["feature_set", "model"], as_index=False)
        [["log_loss", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]]
        .mean()
        .sort_values(["log_loss", "macro_f1"], ascending=[True, False])
    )

    print("saved:")
    print(f"  {RESULT_PATH}")
    print(f"  {PRED_PATH}")
    print(f"  {SUMMARY_PATH}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
