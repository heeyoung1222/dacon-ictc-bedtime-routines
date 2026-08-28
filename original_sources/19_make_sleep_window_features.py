# %% [markdown]
# # Sleep-window Feature Table
#
# 목적:
# - 기존 evening window에 20~04, 20~06, 22~04, 22~06 야간 window를 추가함
# - raw sensor만 사용하며 label은 feature 생성에 사용하지 않음
# - 기존 feature_table 파일을 덮어쓰지 않고 별도 파일로 저장함

# %%
from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = PROJECT_DIR / "make_features.py"


def load_feature_module():
    spec = importlib.util.spec_from_file_location("make_features_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    feature_module = load_feature_module()
    feature_module.OUT_DIR = PROJECT_DIR
    feature_module.WINDOWS = [
        ("w18_24", 18 * 60, 24 * 60),
        ("w20_24", 20 * 60, 24 * 60),
        ("w21_24", 21 * 60, 24 * 60),
        ("w22_24", 22 * 60, 24 * 60),
        ("w20_02", 20 * 60, 2 * 60),
        ("w20_04", 20 * 60, 4 * 60),
        ("w20_06", 20 * 60, 6 * 60),
        ("w22_04", 22 * 60, 4 * 60),
        ("w22_06", 22 * 60, 6 * 60),
    ]

    features = feature_module.build_features()
    train = features[features["split"] == "train"].copy()
    test = features[features["split"] == "test"].copy()

    all_csv = PROJECT_DIR / "feature_table_all_sleepwindows.csv"
    all_parquet = PROJECT_DIR / "feature_table_all_sleepwindows.parquet"
    train_csv = PROJECT_DIR / "feature_table_train_sleepwindows.csv"
    test_csv = PROJECT_DIR / "feature_table_test_sleepwindows.csv"

    features.to_csv(all_csv, index=False, encoding="utf-8-sig")
    features.to_parquet(all_parquet, index=False)
    train.to_csv(train_csv, index=False, encoding="utf-8-sig")
    test.to_csv(test_csv, index=False, encoding="utf-8-sig")

    print("saved:")
    print(all_csv, features.shape)
    print(all_parquet, features.shape)
    print(train_csv, train.shape)
    print(test_csv, test.shape)


if __name__ == "__main__":
    main()
