"""
model_dual.py
Train two RF models with walk-forward validation:
  1. Weather model  — WEATHER_FEATURES, max WF folds on weather data
  2. Generic model  — GENERIC_FEATURES, max WF folds on all-market data

Walk-forward: train on rolling 8-month window, test on 1 month.
Saves:
  models/rf_weather_v2.joblib
  models/rf_generic_v1.joblib
  data/processed/wf_weather_v2.parquet
  data/processed/wf_generic_v1.parquet
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss

from features_generic import WEATHER_FEATURES, GENERIC_FEATURES, TARGET_COL

PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MONTHS = 8
TEST_MONTHS  = 1


def walk_forward(df: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    df = df.copy()
    df["_date"] = pd.to_datetime(df["close_date"])
    df["_ym"]   = df["_date"].dt.to_period("M")
    months      = sorted(df["_ym"].unique())

    print(f"\n[WF:{label}] {len(df):,} rows | {months[0]} → {months[-1]} | "
          f"{len(months)} months | max folds={len(months)-TRAIN_MONTHS}")

    all_preds = []
    n_folds   = 0

    for i in range(TRAIN_MONTHS, len(months)):
        train_months = months[max(0, i - TRAIN_MONTHS): i]
        test_month   = months[i: i + TEST_MONTHS]
        if not test_month:
            break

        train = df[df["_ym"].isin(train_months)]
        test  = df[df["_ym"].isin(test_month)]

        if len(train) < 50 or len(test) < 3:
            continue

        X_tr = train[features].fillna(0).values
        y_tr = train[TARGET_COL].values
        X_te = test[features].fillna(0).values
        y_te = test[TARGET_COL].values

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else float("nan")
        ll  = log_loss(y_te, probs)

        test_out = test.copy()
        test_out["pred_prob"] = probs
        test_out["fold"]      = n_folds
        all_preds.append(test_out)
        n_folds += 1

        if n_folds % 5 == 0 or n_folds <= 3:
            print(f"  fold {n_folds:03d} | train {train_months[0]}→{train_months[-1]} "
                  f"| test {test_month[0]} | n={len(test):>5} | AUC={auc:.3f} | LL={ll:.3f}")

    print(f"[WF:{label}] {n_folds} folds completed.")
    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


def train_final(df: pd.DataFrame, features: list[str]) -> RandomForestClassifier:
    X = df[features].fillna(0).values
    y = df[TARGET_COL].values
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def run_model(name: str, feat_path: str, features: list[str], save_prefix: str) -> pd.DataFrame:
    df = pd.read_parquet(PROC_DIR / feat_path)
    print(f"\n{'='*60}")
    print(f"  MODEL: {name}")
    print(f"  Rows: {len(df):,}  |  Features: {len(features)}")
    print(f"{'='*60}")

    wf = walk_forward(df, features, name)
    if wf.empty:
        print(f"[{name}] No WF folds generated.")
        return wf

    out_pq = PROC_DIR / f"wf_{save_prefix}.parquet"
    wf.to_parquet(out_pq, index=False)

    # Overall metrics
    auc_all = roc_auc_score(wf[TARGET_COL], wf["pred_prob"])
    ll_all  = log_loss(wf[TARGET_COL], wf["pred_prob"])
    n_folds = wf["fold"].nunique()
    print(f"\n[{name}] WF summary: {n_folds} folds | {len(wf):,} test rows | "
          f"AUC={auc_all:.4f} | LogLoss={ll_all:.4f}")
    print(f"[{name}] Saved predictions → {out_pq}")

    # Final model
    print(f"[{name}] Training final model on all data...")
    final = train_final(df, features)
    model_path = MODEL_DIR / f"rf_{save_prefix}.joblib"
    joblib.dump(final, model_path)
    print(f"[{name}] Model saved → {model_path}")

    # Feature importances
    imp = pd.Series(final.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n[{name}] Top 10 features:")
    print(imp.head(10).to_string())

    # Save feature list
    feat_json = MODEL_DIR / f"features_{save_prefix}.json"
    with open(feat_json, "w") as fh:
        json.dump({"features": features, "target": TARGET_COL, "n_features": len(features),
                   "wf_auc": round(auc_all, 4), "wf_folds": n_folds}, fh, indent=2)
    return wf


def run() -> None:
    # Weather model
    wf_weather = run_model(
        name        = "Weather (NOAA + Kalshi)",
        feat_path   = "feature_matrix_weather.parquet",
        features    = WEATHER_FEATURES,
        save_prefix = "weather_v2",
    )

    # Generic model (all markets, capped for training speed)
    MAX_ROWS = 500_000
    df_all   = pd.read_parquet(PROC_DIR / "feature_matrix_all.parquet")
    if len(df_all) > MAX_ROWS:
        df_all = df_all.sample(MAX_ROWS, random_state=42)
        tmp_path = PROC_DIR / "feature_matrix_all_capped.parquet"
        df_all.to_parquet(tmp_path, index=False)
        feat_path_gen = "feature_matrix_all_capped.parquet"
        print(f"[ModelDual] Generic model: sampled {MAX_ROWS:,} rows")
    else:
        feat_path_gen = "feature_matrix_all.parquet"

    wf_generic = run_model(
        name        = "Generic (all markets)",
        feat_path   = feat_path_gen,
        features    = GENERIC_FEATURES,
        save_prefix = "generic_v1",
    )

    print("\n[ModelDual] Both models complete.")
    return wf_weather, wf_generic


if __name__ == "__main__":
    run()
