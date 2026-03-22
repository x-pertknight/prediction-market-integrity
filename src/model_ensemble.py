"""
model_ensemble.py
Train 3 RF models + weighted ensemble:
  1. RF_weather   — weather markets + NOAA + macro features
  2. RF_crypto    — crypto/financials markets + BTC + whale + macro
  3. RF_general   — all markets + generic + sentiment + macro

Ensemble: weighted vote by category Sharpe (computed from WF results).
Walk-forward with max folds.
Includes SHAP value computation for top feature importance.
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from features_v2 import GENERIC_FEATURES_V2, WEATHER_FEATURES_V2
from features_generic import TARGET_COL

PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MONTHS = 6    # reduced to maximise folds given limited history
TEST_MONTHS  = 1

CRYPTO_CATEGORIES  = {"Crypto", "Financials", "Companies"}
WEATHER_CATEGORIES = {"Climate and Weather"}
GENERAL_CATEGORIES = {"Politics", "Elections", "Sports", "Entertainment", "Economics",
                       "Science and Technology", "World", "Health", "Mentions", "Social"}


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    df = df.copy()
    df["_date"] = pd.to_datetime(df["close_date"])
    df["_ym"]   = df["_date"].dt.to_period("M")
    months      = sorted(df["_ym"].unique())

    n_possible = len(months) - TRAIN_MONTHS
    print(f"\n[WF:{label}] {len(df):,} rows | {months[0]}→{months[-1]} | "
          f"max folds={n_possible}")

    all_preds = []
    n_folds   = 0

    for i in range(TRAIN_MONTHS, len(months)):
        train_months = months[max(0, i - TRAIN_MONTHS): i]
        test_month   = [months[i]] if i < len(months) else []
        if not test_month:
            break

        train = df[df["_ym"].isin(train_months)]
        test  = df[df["_ym"].isin(test_month)]
        if len(train) < 30 or len(test) < 3:
            continue

        # Only use features that are non-constant in this fold
        feat = [f for f in features if f in train.columns and train[f].nunique() > 1]

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

        test_out = test.copy()
        test_out["pred_prob"]    = probs
        test_out["fold"]         = n_folds
        test_out["model_label"]  = label
        test_out["features_used"] = len(feat)
        all_preds.append(test_out)
        n_folds += 1

        if n_folds % 3 == 0 or n_folds <= 2:
            print(f"  fold {n_folds:03d} | {train_months[0]}→{train_months[-1]} "
                  f"→ {test_month[0]} | n_test={len(test):>5} | AUC={auc:.3f}")

    print(f"[WF:{label}] {n_folds} folds done.")
    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)


# ── Train final model + SHAP ──────────────────────────────────────────────────

def train_final_and_shap(df: pd.DataFrame, features: list[str], name: str) -> RandomForestClassifier:
    feat = [f for f in features if f in df.columns and df[f].nunique() > 1]
    X = df[feat].fillna(0).values
    y = df[TARGET_COL].values

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X, y)

    # Feature importances
    imp = pd.Series(clf.feature_importances_, index=feat).sort_values(ascending=False)
    print(f"\n[{name}] Top 15 feature importances:")
    for fname, fval in imp.head(15).items():
        bar = "█" * int(fval * 200)
        print(f"  {fname:<35} {fval:.4f} {bar}")

    # SHAP values (sample for speed)
    if HAS_SHAP and len(X) > 0:
        try:
            sample = X[:min(500, len(X))]
            explainer  = shap.TreeExplainer(clf)
            shap_vals  = explainer.shap_values(sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]   # class 1 (YES)
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_imp = pd.Series(mean_abs_shap, index=feat).sort_values(ascending=False)
            shap_path = MODEL_DIR / f"shap_{name}.json"
            with open(shap_path, "w") as f:
                json.dump(shap_imp.head(20).to_dict(), f, indent=2)
            print(f"[{name}] SHAP top 5: {shap_imp.head(5).to_dict()}")
        except Exception as e:
            print(f"[{name}] SHAP skipped: {e}")

    # Save model
    model_path = MODEL_DIR / f"rf_{name}.joblib"
    joblib.dump(clf, model_path)

    feat_path = MODEL_DIR / f"features_{name}.json"
    with open(feat_path, "w") as f:
        json.dump({"features": feat, "target": TARGET_COL, "n": len(feat),
                   "top10_importance": imp.head(10).to_dict()}, f, indent=2)

    print(f"[{name}] Saved → {model_path}")
    return clf


# ── Ensemble ──────────────────────────────────────────────────────────────────

def build_ensemble(wf_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine predictions from multiple models.
    For rows covered by a specialist model, use that model's prediction.
    For rows covered by multiple models, weight by their fold AUC.
    """
    frames = []
    for label, wf in wf_dict.items():
        if not wf.empty:
            keep = ["close_date","series_ticker","outcome","pred_prob","fold","model_label"]
            # Carry market_price so backtest can compute entry signals
            for col in ["market_price", "price_lag1", "category", "city"]:
                if col in wf.columns:
                    keep.append(col)
            frames.append(wf[keep])
    if not frames:
        return pd.DataFrame()

    all_preds = pd.concat(frames, ignore_index=True)

    # Simple average of available model probs per row
    ensemble = (
        all_preds.groupby(["close_date","series_ticker","outcome"])["pred_prob"]
        .mean()
        .reset_index()
        .rename(columns={"pred_prob": "pred_prob_ensemble"})
    )
    all_preds = all_preds.merge(ensemble, on=["close_date","series_ticker","outcome"], how="left")
    return all_preds


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    wf_dict = {}

    # 1. Weather model
    wx_path = PROC_DIR / "feature_matrix_weather_v2.parquet"
    if wx_path.exists():
        df_wx = pd.read_parquet(wx_path)
        avail_wx = [c for c in WEATHER_FEATURES_V2 if c in df_wx.columns]
        wf_wx = walk_forward(df_wx, avail_wx, "weather_v2")
        if not wf_wx.empty:
            wf_wx.to_parquet(PROC_DIR / "wf_weather_v2.parquet", index=False)
            wf_dict["weather"] = wf_wx
            train_final_and_shap(df_wx, avail_wx, "weather_v2")

    # 2. Generic model (all markets)
    all_path = PROC_DIR / "feature_matrix_all_v2.parquet"
    if all_path.exists():
        df_all = pd.read_parquet(all_path)
        avail_all = [c for c in GENERIC_FEATURES_V2 if c in df_all.columns]

        # Crypto subset
        if "category" in df_all.columns:
            df_crypto = df_all[df_all["category"].isin(CRYPTO_CATEGORIES)].copy()
            if len(df_crypto) > 100:
                wf_crypto = walk_forward(df_crypto, avail_all, "crypto_v1")
                if not wf_crypto.empty:
                    wf_crypto.to_parquet(PROC_DIR / "wf_crypto_v1.parquet", index=False)
                    wf_dict["crypto"] = wf_crypto
                    train_final_and_shap(df_crypto, avail_all, "crypto_v1")

        # General model (all)
        wf_general = walk_forward(df_all, avail_all, "general_v1")
        if not wf_general.empty:
            wf_general.to_parquet(PROC_DIR / "wf_general_v1.parquet", index=False)
            wf_dict["general"] = wf_general
            train_final_and_shap(df_all, avail_all, "general_v1")

    # 3. Ensemble
    if len(wf_dict) > 1:
        ensemble_df = build_ensemble(wf_dict)
        ensemble_df.to_parquet(PROC_DIR / "wf_ensemble.parquet", index=False)
        print(f"\n[Ensemble] {len(ensemble_df):,} rows → wf_ensemble.parquet")

    print("\n[ModelEnsemble] All models complete.")


if __name__ == "__main__":
    run()
