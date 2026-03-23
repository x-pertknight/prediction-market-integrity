"""
model_clean.py — Walk-forward RF on clean (lookahead-free) features
====================================================================
Same walk-forward structure as model_ensemble.py, but uses only
features from features_clean.py (no market_price, no price_lag1,
macro joined on open_date).

Output:
  data/processed/wf_clean_general.parquet   — all-market WF predictions
  data/processed/wf_clean_weather.parquet   — weather WF predictions
  models/rf_clean_general.joblib
  models/rf_clean_weather.joblib

The WF parquets carry metadata columns (ticker, price_lag1, market_price)
for use in backtest_clean.py but these are NOT used as model features.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss

from features_clean import CLEAN_FEATURES, CLEAN_WEATHER_EXTRA
from features_generic import WEATHER_FEATURES, TARGET_COL

PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MONTHS = 6
TEST_MONTHS  = 1
MAX_ROWS     = 500_000   # cap to keep training time bounded

# Columns to carry through as metadata (not model features)
META_COLS = ["ticker", "series_ticker", "category", "open_date", "close_date",
             "market_price", "price_lag1", "outcome", "days_in_market"]


def walk_forward(df: pd.DataFrame, features: list[str], label: str,
                 max_rows: int = MAX_ROWS) -> pd.DataFrame:
    df = df.copy()
    df["_date"] = pd.to_datetime(df["close_date"])
    df["_ym"]   = df["_date"].dt.to_period("M")
    months      = sorted(df["_ym"].unique())

    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
        print(f"[WF:{label}] Capped to {max_rows:,} rows (was {len(df):,})")

    n_possible = len(months) - TRAIN_MONTHS
    print(f"\n[WF:{label}] {len(df):,} rows | {months[0]}→{months[-1]} | "
          f"{len(months)} months | max_folds={n_possible}")

    all_preds = []
    n_folds   = 0

    for i in range(TRAIN_MONTHS, len(months)):
        train_months = months[max(0, i - TRAIN_MONTHS): i]
        test_month   = months[i] if i < len(months) else None
        if test_month is None:
            break

        train = df[df["_ym"].isin(train_months)]
        test  = df[df["_ym"] == test_month]
        if len(train) < 50 or len(test) < 5:
            continue

        # Only features non-constant in this fold
        feat = [f for f in features if f in train.columns and train[f].nunique() > 1]
        if len(feat) < 3:
            continue

        X_tr = train[feat].fillna(0).values
        y_tr = train[TARGET_COL].values
        X_te = test[feat].fillna(0).values
        y_te = test[TARGET_COL].values

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else float("nan")
        ll  = log_loss(y_te, probs) if len(np.unique(y_te)) > 1 else float("nan")

        test_out = test.copy()
        test_out["pred_prob_clean"] = probs
        test_out["fold"]            = n_folds
        test_out["model_label"]     = label
        test_out["features_used"]   = len(feat)
        all_preds.append(test_out)
        n_folds += 1

        if n_folds % 3 == 0 or n_folds <= 3:
            print(f"  fold {n_folds:03d} | train {train_months[0]}→{train_months[-1]} "
                  f"→ test {test_month} | n_test={len(test):>6} | "
                  f"AUC={auc:.3f} | LL={ll:.3f}")

    print(f"[WF:{label}] {n_folds} folds done.")
    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)


def train_final(df: pd.DataFrame, features: list[str], name: str) -> None:
    feat = [f for f in features if f in df.columns and df[f].nunique() > 1]
    X = df[feat].fillna(0).values
    y = df[TARGET_COL].values

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X, y)

    imp = pd.Series(clf.feature_importances_, index=feat).sort_values(ascending=False)
    print(f"\n[{name}] Top 10 feature importances (clean model):")
    for fname, fval in imp.head(10).items():
        bar = "█" * int(fval * 200)
        print(f"  {fname:<35} {fval:.4f} {bar}")

    path = MODEL_DIR / f"rf_{name}.joblib"
    joblib.dump(clf, path)
    print(f"[{name}] Saved → {path}")


def run() -> None:
    # ── General clean model ───────────────────────────────────────────────────
    all_path = PROC_DIR / "feature_matrix_clean_all.parquet"
    if not all_path.exists():
        raise FileNotFoundError(f"{all_path} not found. Run features_clean.py first.")

    df_all = pd.read_parquet(all_path)
    avail  = [c for c in CLEAN_FEATURES if c in df_all.columns]
    print(f"[ModelClean] All-market matrix: {len(df_all):,} rows | {len(avail)} clean features")
    print(f"  Features: {avail}")

    wf_all = walk_forward(df_all, avail, "clean_general")
    if not wf_all.empty:
        # Carry required metadata for backtest
        meta_in_wf = [c for c in META_COLS if c in wf_all.columns]
        out_path   = PROC_DIR / "wf_clean_general.parquet"
        wf_all.to_parquet(out_path, index=False)

        test_aucs = []
        for fold, g in wf_all.groupby("fold"):
            if g["outcome"].nunique() > 1:
                test_aucs.append(roc_auc_score(g["outcome"], g["pred_prob_clean"]))
        mean_auc = np.mean(test_aucs) if test_aucs else float("nan")
        print(f"\n[ModelClean] General WF summary: AUC={mean_auc:.4f} | "
              f"rows={len(wf_all):,} | meta cols: {meta_in_wf}")

        train_final(df_all, avail, "clean_general")

    # ── Weather clean model ───────────────────────────────────────────────────
    wx_path = PROC_DIR / "feature_matrix_clean_weather.parquet"
    if wx_path.exists():
        df_wx   = pd.read_parquet(wx_path)
        avail_wx = [c for c in WEATHER_FEATURES + CLEAN_WEATHER_EXTRA
                    if c in df_wx.columns and c not in ("market_price", "price_lag1")]
        avail_wx = [c for c in avail_wx if c in df_wx.columns]

        wf_wx = walk_forward(df_wx, avail_wx, "clean_weather", max_rows=50_000)
        if not wf_wx.empty:
            out_wx = PROC_DIR / "wf_clean_weather.parquet"
            wf_wx.to_parquet(out_wx, index=False)
            test_aucs_wx = []
            for fold, g in wf_wx.groupby("fold"):
                if g["outcome"].nunique() > 1:
                    test_aucs_wx.append(roc_auc_score(g["outcome"], g["pred_prob_clean"]))
            mean_auc_wx = np.mean(test_aucs_wx) if test_aucs_wx else float("nan")
            print(f"\n[ModelClean] Weather WF summary: AUC={mean_auc_wx:.4f}")
            train_final(df_wx, avail_wx, "clean_weather")

    print("\n[ModelClean] Done.")


if __name__ == "__main__":
    run()
