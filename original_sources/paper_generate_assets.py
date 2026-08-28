from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer


PROJECT = Path(__file__).resolve().parent
PAPER = PROJECT / "paper"
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LABELS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLS = ["subject_id", "sleep_date", "lifelog_date"]


def load_train_test():
    train = pd.read_csv(PROJECT / "feature_table_train_hidden_hr.csv")
    test = pd.read_csv(PROJECT / "feature_table_test_hidden_hr.csv")
    for df in [train, test]:
        df["sleep_date"] = pd.to_datetime(df["sleep_date"])
        df["lifelog_date"] = pd.to_datetime(df["lifelog_date"])
    return train, test


def save_label_prevalence(train: pd.DataFrame):
    rates = train[LABELS].mean()
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    colors = ["#4C78A8", "#4C78A8", "#4C78A8", "#F58518", "#F58518", "#F58518", "#F58518"]
    ax.bar(LABELS, rates.values, color=colors)
    ax.axhline(0.5, color="#444", lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Positive prevalence")
    ax.set_title("Training Label Prevalence")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_label_prevalence.png", dpi=250)
    plt.close(fig)


def save_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")
    boxes = [
        ("Raw lifelog\nparquet", 0.04),
        ("Windowed &\nroutine features", 0.23),
        ("Subject-interleaved\nvalidation", 0.44),
        ("Label-wise\nGBDT models", 0.64),
        ("Temperature +\nblock calibration", 0.83),
    ]
    for text, x in boxes:
        rect = plt.Rectangle((x, 0.30), 0.14, 0.36, facecolor="#E8F1FB", edgecolor="#2F5D8C", lw=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.07, 0.48, text, ha="center", va="center", fontsize=9)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + 0.14
        x1 = boxes[i + 1][1]
        ax.annotate("", xy=(x1, 0.48), xytext=(x0, 0.48), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
    ax.text(0.50, 0.10, "Outputs: calibrated probabilities for Q1-Q3 and S1-S4", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_pipeline.png", dpi=250)
    plt.close(fig)


def save_split_structure(train: pd.DataFrame, test: pd.DataFrame):
    train_m = train[ID_COLS].assign(split="train")
    test_m = test[ID_COLS].assign(split="test")
    all_m = pd.concat([train_m, test_m], ignore_index=True).sort_values(["subject_id", "sleep_date"])
    subjects = sorted(all_m["subject_id"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for yi, sid in enumerate(subjects):
        g = all_m[all_m["subject_id"].eq(sid)].reset_index(drop=True)
        x = np.arange(len(g))
        colors = np.where(g["split"].eq("test"), "#D64B4B", "#4C78A8")
        ax.scatter(x, np.full_like(x, yi), c=colors, s=18, marker="s")
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=8)
    ax.set_xlabel("Within-subject chronological index")
    ax.set_title("Observed Train/Test Block Structure by Subject")
    ax.text(0.99, 0.02, "blue=train, red=test", transform=ax.transAxes, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_split_structure.png", dpi=250)
    plt.close(fig)


def save_score_progress():
    log_path = PROJECT / "public_score_log.csv"
    if not log_path.exists():
        return
    df = pd.read_csv(log_path)
    df = df.dropna(subset=["public_log_loss"]).copy()
    df["public_log_loss"] = df["public_log_loss"].astype(float)
    df["short"] = df["submission_file"].str.replace("submission_", "", regex=False).str.replace(".csv", "", regex=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(range(len(df)), df["public_log_loss"], marker="o", color="#4C78A8", lw=1.5)
    best_idx = df["public_log_loss"].idxmin()
    ax.scatter([df.index.get_loc(best_idx)], [df.loc[best_idx, "public_log_loss"]], color="#D64B4B", s=55, zorder=5)
    ax.set_ylabel("Public log-loss")
    ax.set_xlabel("Submission order")
    ax.set_title("Public Leaderboard Feedback During Development")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (_, r) in enumerate(df.iterrows()):
        if i in [0, len(df) - 1] or r["public_log_loss"] == df["public_log_loss"].min() or r["public_log_loss"] > 0.62:
            ax.text(i, r["public_log_loss"] + 0.004, f"{r['public_log_loss']:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_score_progress.png", dpi=250)
    plt.close(fig)


def feature_columns_for_shap(train: pd.DataFrame, label: str, max_cols: int = 450):
    protected = set(ID_COLS + LABELS + ["split"])
    cols = [
        c
        for c in train.columns
        if c not in protected
        and pd.api.types.is_numeric_dtype(train[c])
        and train[c].notna().mean() >= 0.55
        and train[c].nunique(dropna=True) > 3
    ]
    X = train[cols].replace([np.inf, -np.inf], np.nan)
    y = train[label].astype(int)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scores, _ = f_classif(X_imp, y)
    scores = np.nan_to_num(scores, nan=0, posinf=0, neginf=0)
    order = np.argsort(scores)[::-1][: min(max_cols, len(cols))]
    return [cols[i] for i in order]


def save_shap(train: pd.DataFrame):
    label = "Q3"
    cols = feature_columns_for_shap(train, label)
    X = train[cols].replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=cols)
    y = train[label].astype(int)
    model = LGBMClassifier(
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=11,
        max_depth=3,
        min_child_samples=18,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.8,
        reg_lambda=8.0,
        random_state=2026,
        verbose=-1,
    )
    model.fit(X_imp, y)
    explainer = shap.TreeExplainer(model)
    sample = X_imp.sample(n=min(250, len(X_imp)), random_state=2026)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    plt.figure(figsize=(7.0, 4.2))
    shap.summary_plot(shap_values, sample, max_display=15, show=False, plot_size=None)
    plt.title("SHAP Summary for Q3 Prediction", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG / "fig5_shap_summary.png", dpi=250, bbox_inches="tight")
    plt.close()

    importances = np.abs(shap_values).mean(axis=0)
    top = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": importances}).sort_values("mean_abs_shap", ascending=False)
    top.to_csv(PAPER / "shap_q3_top_features.csv", index=False, encoding="utf-8-sig")


def save_summary_tables(train: pd.DataFrame):
    label_rates = train[LABELS].mean().rename("positive_rate").reset_index().rename(columns={"index": "label"})
    label_rates.to_csv(PAPER / "label_prevalence.csv", index=False, encoding="utf-8-sig")
    scores = pd.read_csv(PROJECT / "public_score_log.csv")
    scores.to_csv(PAPER / "public_score_log_used.csv", index=False, encoding="utf-8-sig")


def main():
    PAPER.mkdir(exist_ok=True)
    train, test = load_train_test()
    save_label_prevalence(train)
    save_pipeline()
    save_split_structure(train, test)
    save_score_progress()
    save_shap(train)
    save_summary_tables(train)
    print(f"Saved paper assets to {PAPER}")


if __name__ == "__main__":
    main()
