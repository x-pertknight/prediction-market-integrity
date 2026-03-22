"""
model.py — model_rf_02 + model_rf_03 + model_rf_04
Random Forest training with walk-forward validation.
Train window: 8 months. Test window: 1 month. Roll monthly.
Output: models/rf_traptrade_v1.joblib, models/features.json,
        data/processed/wf_predictions.parquet
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score

from features import FEATURE_COLS, TARGET_COL

PROC_DIR   = Path("data/processed")
MODEL_DIR  = Path("models")
TRAIN_MONTHS = 8
TEST_MONTHS  = 1


def walk_forward_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward CV on time-sorted data.
    Each fold: train on 8 months, test on next 1 month. Slide by 1 month.
    Returns DataFrame with columns from df + ['pred_prob', 'fold'].
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ym"]   = df["date"].dt.to_period("M")

    months = sorted(df["ym"].unique())
    # Need at least TRAIN_MONTHS + TEST_MONTHS to form one fold
    if len(months) < TRAIN_MONTHS + TEST_MONTHS:
        raise ValueError(f"Not enough months: {len(months)} < {TRAIN_MONTHS + TEST_MONTHS}")

    all_preds = []
    fold = 0

    for i in range(TRAIN_MONTHS, len(months) - TEST_MONTHS + 1):
        train_months = months[i - TRAIN_MONTHS : i]
        test_months  = months[i : i + TEST_MONTHS]

        train = df[df["ym"].isin(train_months)]
        test  = df[df["ym"].isin(test_months)]

        if len(train) < 50 or len(test) < 5:
            continue

        X_train = train[FEATURE_COLS].values
        y_train = train[TARGET_COL].values
        X_test  = test[FEATURE_COLS].values
        y_test  = test[TARGET_COL].values

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        pred_prob = clf.predict_proba(X_test)[:, 1]

        test_copy = test.copy()
        test_copy["pred_prob"] = pred_prob
        test_copy["fold"]      = fold

        auc = roc_auc_score(y_test, pred_prob) if len(np.unique(y_test)) > 1 else float("nan")
        ll  = log_loss(y_test, pred_prob)
        print(f"  Fold {fold:02d} | train={train_months[0]}→{train_months[-1]} "
              f"| test={test_months[0]} | n_train={len(train)} n_test={len(test)} "
              f"| AUC={auc:.3f} | LogLoss={ll:.3f}")

        all_preds.append(test_copy)
        fold += 1

    return pd.concat(all_preds, ignore_index=True)


def train_final_model(df: pd.DataFrame) -> RandomForestClassifier:
    """Train final model on all available data."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def run() -> tuple[pd.DataFrame, RandomForestClassifier]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROC_DIR / "feature_matrix.parquet")
    print(f"[Model] Loaded {len(df)} rows | {df['date'].min()} → {df['date'].max()}")
    print(f"[Model] Features: {len(FEATURE_COLS)}")
    print(f"[Model] Starting walk-forward CV (train={TRAIN_MONTHS}mo, test={TEST_MONTHS}mo)...")

    wf = walk_forward_cv(df)

    out_preds = PROC_DIR / "wf_predictions.parquet"
    wf.to_parquet(out_preds, index=False)
    print(f"\n[Model] WF predictions saved: {len(wf)} rows → {out_preds}")

    # Overall WF metrics
    auc_all = roc_auc_score(wf[TARGET_COL], wf["pred_prob"])
    ll_all  = log_loss(wf[TARGET_COL], wf["pred_prob"])
    print(f"[Model] Overall WF AUC: {auc_all:.4f} | LogLoss: {ll_all:.4f}")

    # Feature importance (from last fold — proxy)
    print("\n[Model] Training final model on all data...")
    final = train_final_model(df)

    importances = pd.Series(final.feature_importances_, index=FEATURE_COLS)
    print("\n[Model] Top 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10).to_string())

    # model_rf_04: save model + features
    model_path = MODEL_DIR / "rf_traptrade_v1.joblib"
    joblib.dump(final, model_path)
    print(f"\n[Model] Saved → {model_path}")

    features_path = MODEL_DIR / "features.json"
    with open(features_path, "w") as f:
        json.dump({"features": FEATURE_COLS, "target": TARGET_COL, "n_features": len(FEATURE_COLS)}, f, indent=2)
    print(f"[Model] Saved → {features_path}")

    return wf, final


if __name__ == "__main__":
    run()
