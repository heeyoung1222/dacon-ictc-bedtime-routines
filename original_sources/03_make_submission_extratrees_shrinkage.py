# %% [markdown]
# # 데이콘 최종 학습 및 제출 파일 생성
#
# 목적:
# - 앞 단계에서 선택한 기준 모델 `ExtraTreesShrinkage`를 사용함
# - 시간 구간별 feature validation 결과를 바탕으로 제출용 확률을 생성함
# - 대회 평가 지표가 Average Log-Loss이므로 0/1 class가 아니라 확률값을 제출함
#
# 생성 파일:
# - `submission_global_w21_24.csv`: 전체 평균 Log-Loss가 가장 좋았던 `day + 21~24` feature set 사용
# - `submission_labelwise_best_window.csv`: validation에서 라벨별로 가장 좋았던 feature set을 각각 사용
# - `final_model_selection_summary.csv`: 라벨별 feature set, alpha, 예측 확률 요약 저장
#
# 주의:
# - test 라벨은 사용하지 않음
# - shrinkage alpha는 train 내부 시간순 calibration fold에서만 선택함
# - validation 결과를 바탕으로 feature set을 선택하므로, labelwise submission은 성능 후보로 보되 과도한 leaderboard 맞춤은 피해야 함

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline


# %% [markdown]
# ## 1. 설정

# %%
PROJECT_DIR = Path(r"C:\Users\jungheeyoung\Desktop\데이콘")
TRAIN_FEATURE_PATH = PROJECT_DIR / "feature_table_train.csv"
TEST_FEATURE_PATH = PROJECT_DIR / "feature_table_test.csv"
SAMPLE_SUBMISSION_PATH = Path(
    r"C:\Users\jungheeyoung\AppData\Local\Temp\ch2026_submission_sample.csv"
)
WINDOW_RESULT_PATH = PROJECT_DIR / "window_feature_validation_results.csv"

GLOBAL_SUBMISSION_PATH = PROJECT_DIR / "submission_global_w21_24.csv"
LABELWISE_SUBMISSION_PATH = PROJECT_DIR / "submission_labelwise_best_window.csv"
MODEL_SELECTION_PATH = PROJECT_DIR / "final_model_selection_summary.csv"

LABELS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLS = ["subject_id", "sleep_date", "lifelog_date", "split"]
WINDOWS = ["w18_24", "w20_24", "w21_24", "w22_24", "w20_02"]
RANDOM_STATE = 42

# Log-Loss에서 극단 확률이 틀릴 때 큰 벌점이 생기므로 최종 확률을 아주 약하게 clipping함.
# ExtraTreesShrinkage 자체가 완만한 확률을 만들기 때문에 넓은 범위만 적용함.
CLIP_MIN = 0.02
CLIP_MAX = 0.98


# %% [markdown]
# ## 2. 데이터 로드

# %%
def load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_FEATURE_PATH)
    test = pd.read_csv(TEST_FEATURE_PATH)
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    for df in [train, test, sample]:
        df["sleep_date"] = pd.to_datetime(df["sleep_date"])
        df["lifelog_date"] = pd.to_datetime(df["lifelog_date"])

    return train, test, sample


# %% [markdown]
# ## 3. Feature set 구성
#
# `personal_*` feature는 전체 기간 분포를 반영할 수 있으므로 최종 기본 제출에서는 제외함.

# %%
def has_window_token(col: str, window: str) -> bool:
    return f"_{window}_" in col or col.startswith(f"{window}_")


def has_any_window_token(col: str) -> bool:
    return any(has_window_token(col, window) for window in WINDOWS)


def get_feature_sets(train: pd.DataFrame) -> dict[str, list[str]]:
    protected = set(ID_COLS + LABELS)
    numeric_cols = [
        col
        for col in train.columns
        if col not in protected and pd.api.types.is_numeric_dtype(train[col])
    ]
    clean_cols = [col for col in numeric_cols if not col.startswith("personal_")]
    base_cols = [col for col in clean_cols if not has_any_window_token(col)]

    feature_sets = {"day_only": base_cols}
    for window in WINDOWS:
        window_cols = [col for col in clean_cols if has_window_token(col, window)]
        feature_sets[f"day_plus_{window}"] = base_cols + window_cols
    feature_sets["day_plus_all_windows"] = clean_cols
    return feature_sets


def choose_labelwise_feature_sets() -> dict[str, str]:
    if not WINDOW_RESULT_PATH.exists():
        return {
            "Q1": "day_plus_all_windows",
            "Q2": "day_plus_w18_24",
            "Q3": "day_only",
            "S1": "day_plus_all_windows",
            "S2": "day_plus_w21_24",
            "S3": "day_plus_all_windows",
            "S4": "day_plus_w21_24",
        }

    results = pd.read_csv(WINDOW_RESULT_PATH)
    idx = results.groupby("label")["log_loss"].idxmin()
    best = results.loc[idx, ["label", "feature_set"]]
    return dict(zip(best["label"], best["feature_set"]))


# %% [markdown]
# ## 4. ExtraTreesShrinkage 모델 정의

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


def make_inner_calibration_fold(
    df: pd.DataFrame,
    valid_ratio: float = 0.2,
    min_cal_per_subject: int = 2,
) -> np.ndarray:
    fold = pd.Series(-1, index=df.index, dtype=int)
    for _, group in df.sort_values("lifelog_date").groupby("subject_id", sort=True):
        group = group.sort_values("lifelog_date")
        n = len(group)
        n_cal = max(min_cal_per_subject, int(np.ceil(n * valid_ratio)))
        n_cal = min(n_cal, max(1, n - 1))
        fold.loc[group.index[-n_cal:]] = 0
    return fold.to_numpy()


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


# %% [markdown]
# ## 5. 제출 파일 생성

# %%
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

    submission["sleep_date"] = pd.to_datetime(submission["sleep_date"]).dt.strftime("%Y-%m-%d")
    submission["lifelog_date"] = pd.to_datetime(submission["lifelog_date"]).dt.strftime("%Y-%m-%d")
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(summary_rows)


def main() -> None:
    train, test, sample = load_features()
    feature_sets = get_feature_sets(train)

    global_feature_map = {label: "day_plus_w21_24" for label in LABELS}
    labelwise_feature_map = choose_labelwise_feature_sets()

    global_summary = build_submission(
        train,
        test,
        sample,
        feature_sets,
        global_feature_map,
        GLOBAL_SUBMISSION_PATH,
    )
    labelwise_summary = build_submission(
        train,
        test,
        sample,
        feature_sets,
        labelwise_feature_map,
        LABELWISE_SUBMISSION_PATH,
    )

    summary = pd.concat([global_summary, labelwise_summary], ignore_index=True)
    summary.to_csv(MODEL_SELECTION_PATH, index=False, encoding="utf-8-sig")

    print("saved:")
    print(f"  {GLOBAL_SUBMISSION_PATH}")
    print(f"  {LABELWISE_SUBMISSION_PATH}")
    print(f"  {MODEL_SELECTION_PATH}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
