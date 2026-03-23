"""
model_clean.py — Walk-forward ensemble on clean (lookahead-free) features
==========================================================================
Simons Framework A4: RF + XGBoost + Logistic Regression ensemble.
Voting weight = Sharpe * regime_multiplier (HMM regime from A1).
Unanimity 3/3: full size | 2/3: half | 1/3: skip (applied in backtest_clean.py).

Output:
  data/processed/wf_clean_general.parquet   — all-market WF predictions
  data/processed/wf_clean_weather.parquet   — weather WF predictions
  models/rf_clean_general.joblib
  models/xgb_clean_general.joblib
  models/lr_clean_general.joblib
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[ModelClean] XGBoost not available — ensemble will use RF + LR only")

from features_clean import CLEAN_FEATURES, CLEAN_WEATHER_EXTRA
from features_generic import WEATHER_FEATURES, TARGET_COL

PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MONTHS = 6
TEST_MONTHS  = 1
MAX_ROWS     = 500_000

META_COLS = ["ticker", "series_ticker", "category", "open_date", "close_date",
             "market_price", "price_lag1", "outcome", "days_in_market", "volume_fp"]

# Ensemble vote threshold for sizing: A6 — use 0.7 (was 0.5)
ENSEMBLE_PROB_MULT = 0.70   # A6: entry when price <= pred_prob * 0.70


def _build_models() -> list[tuple[str, object]]:
    """Build classifier list: (name, clf)."""
    models = []

    # 1. Random Forest
    models.append(("rf", RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )))

    # 2. XGBoost (if available)
    if HAS_XGB:
        models.append(("xgb", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0, n_jobs=-1,
        )))

    # 3. Logistic Regression (with scaling)
    models.append(("lr", Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=1.0, max_iter=500, random_state=42, n_jobs=-1,
        )),
    ])))

    return models


def _sharpe(pnl: np.ndarray) -> float:
    """Annualized Sharpe from trade P&L array."""
    if len(pnl) < 5 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def _ensemble_probs(probs_dict: dict[str, np.ndarray],
                    sharpes: dict[str, float],
                    regime_mult: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted average ensemble probability.
    Weight = Sharpe * regime_mult (A4 Simons formula).
    Also returns agreement_score (0, 0.5, or 1.0 — fraction of models agreeing).

    Returns: (ensemble_prob, agreement_score)
    """
    n_models = len(probs_dict)
    if n_models == 0:
        return np.full(1, 0.5), np.zeros(1)

    # Weights: Sharpe * regime_mult (floor at 0.1 so all models contribute)
    weights = {}
    for name, sharpe in sharpes.items():
        if name in probs_dict:
            weights[name] = max(sharpe * regime_mult, 0.1)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        total_weight = n_models

    # Weighted average
    n_obs = len(next(iter(probs_dict.values())))
    ensemble = np.zeros(n_obs)
    for name, prob_arr in probs_dict.items():
        w = weights.get(name, 1.0) / total_weight
        ensemble += prob_arr * w

    # Agreement score (fraction of models that predict > 0.5)
    votes = np.stack(list(probs_dict.values()), axis=1) > 0.5  # (n_obs, n_models)
    agree = votes.mean(axis=1)  # fraction agreeing per row
    # Map to {0.0, 0.5, 1.0} for 3 models (0/3, 1/3 or 2/3, 3/3)
    agreement_score = np.round(agree * 2) / 2  # round to nearest 0.5

    return ensemble, agreement_score


def walk_forward(df: pd.DataFrame, features: list[str], label: str,
                 max_rows: int = MAX_ROWS) -> pd.DataFrame:
    df = df.copy()
    df["_date"] = pd.to_datetime(df["close_date"])
    df["_ym"]   = df["_date"].dt.to_period("M")
    months      = sorted(df["_ym"].unique())

    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
        print(f"[WF:{label}] Capped to {max_rows:,} rows")

    n_possible = len(months) - TRAIN_MONTHS
    print(f"\n[WF:{label}] {len(df):,} rows | {months[0]}→{months[-1]} | "
          f"{len(months)} months | max_folds={n_possible}")

    models_config = _build_models()
    model_names   = [n for n, _ in models_config]
    all_preds     = []
    n_folds       = 0

    # Track Sharpe per model across folds
    fold_sharpes: dict[str, list[float]] = {n: [] for n in model_names}

    for i in range(TRAIN_MONTHS, len(months)):
        train_months = months[max(0, i - TRAIN_MONTHS): i]
        test_month   = months[i]

        train = df[df["_ym"].isin(train_months)]
        test  = df[df["_ym"] == test_month]
        if len(train) < 50 or len(test) < 5:
            continue

        feat = [f for f in features if f in train.columns and train[f].nunique() > 1]
        if len(feat) < 3:
            continue

        X_tr = train[feat].fillna(0).values
        y_tr = train[TARGET_COL].values
        X_te = test[feat].fillna(0).values
        y_te = test[TARGET_COL].values

        probs_dict: dict[str, np.ndarray] = {}
        sharpes_this_fold: dict[str, float] = {}

        for name, clf_template in models_config:
            import copy
            clf = copy.deepcopy(clf_template)
            try:
                clf.fit(X_tr, y_tr)
                prob = clf.predict_proba(X_te)[:, 1]
            except Exception as e:
                print(f"    [{name}] fit failed: {e}")
                prob = np.full(len(X_te), 0.5)

            probs_dict[name] = prob

            # Compute fold Sharpe for this model (simple backtest on test)
            if "price_lag1" in test.columns:
                p_lag = test["price_lag1"].fillna(0).values
                signals = p_lag <= prob * ENSEMBLE_PROB_MULT
                if signals.sum() >= 3:
                    pnl = y_te[signals].astype(float) - p_lag[signals]
                    s   = _sharpe(pnl)
                    sharpes_this_fold[name] = s
                    fold_sharpes[name].append(s)
                else:
                    sharpes_this_fold[name] = 1.0

        # Ensemble (use rolling mean Sharpe as weight)
        rolling_sharpes = {
            n: float(np.mean(fold_sharpes[n][-10:])) if fold_sharpes[n] else 1.0
            for n in model_names
        }
        ens_prob, agreement = _ensemble_probs(probs_dict, rolling_sharpes)

        auc = roc_auc_score(y_te, ens_prob) if len(np.unique(y_te)) > 1 else float("nan")

        test_out = test.copy()
        test_out["pred_prob_clean"] = ens_prob
        test_out["ensemble_agree"]  = agreement
        # Individual model probs
        for name, prob in probs_dict.items():
            test_out[f"pred_{name}"] = prob
        test_out["fold"]          = n_folds
        test_out["model_label"]   = label
        test_out["features_used"] = len(feat)
        all_preds.append(test_out)
        n_folds += 1

        if n_folds % 3 == 0 or n_folds <= 3:
            sharpe_str = " | ".join(
                f"{n}={s:.2f}" for n, s in rolling_sharpes.items())
            print(f"  fold {n_folds:03d} | train {train_months[0]}→{train_months[-1]} "
                  f"→ test {test_month} | n_test={len(test):>6} | "
                  f"AUC={auc:.3f} | [{sharpe_str}]")

    print(f"[WF:{label}] {n_folds} folds done.")
    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)


def train_final(df: pd.DataFrame, features: list[str], name: str) -> None:
    feat = [f for f in features if f in df.columns and df[f].nunique() > 1]
    X = df[feat].fillna(0).values
    y = df[TARGET_COL].values

    for clf_name, clf_template in _build_models():
        import copy
        clf = copy.deepcopy(clf_template)
        try:
            clf.fit(X, y)
        except Exception as e:
            print(f"  [{clf_name}] final train failed: {e}")
            continue

        path = MODEL_DIR / f"{clf_name}_{name}.joblib"
        joblib.dump(clf, path)

        # Feature importance (RF only)
        if clf_name == "rf" and hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=feat).sort_values(ascending=False)
            print(f"\n[{name}/{clf_name}] Top 10 feature importances:")
            for fname, fval in imp.head(10).items():
                bar = "█" * int(fval * 200)
                print(f"  {fname:<35} {fval:.4f} {bar}")
        print(f"[{name}/{clf_name}] Saved → {path}")


def run() -> None:
    # ── General clean model ───────────────────────────────────────────────────
    all_path = PROC_DIR / "feature_matrix_clean_all.parquet"
    if not all_path.exists():
        raise FileNotFoundError(f"{all_path} not found. Run features_clean.py first.")

    df_all = pd.read_parquet(all_path)
    avail  = [c for c in CLEAN_FEATURES if c in df_all.columns]
    print(f"[ModelClean] All-market matrix: {len(df_all):,} rows | {len(avail)} clean features")

    wf_all = walk_forward(df_all, avail, "clean_general")
    if not wf_all.empty:
        out_path = PROC_DIR / "wf_clean_general.parquet"
        wf_all.to_parquet(out_path, index=False)

        test_aucs = []
        for fold, g in wf_all.groupby("fold"):
            if g["outcome"].nunique() > 1:
                test_aucs.append(roc_auc_score(g["outcome"], g["pred_prob_clean"]))
        mean_auc = np.mean(test_aucs) if test_aucs else float("nan")

        # Agreement distribution
        if "ensemble_agree" in wf_all.columns:
            agree_counts = wf_all["ensemble_agree"].value_counts().sort_index()
            print(f"\n[ModelClean] Ensemble agreement distribution:")
            for vote, cnt in agree_counts.items():
                print(f"  {vote:.1f} agreement: {cnt:,} ({cnt/len(wf_all):.1%})")

        print(f"\n[ModelClean] General WF summary: AUC={mean_auc:.4f} | "
              f"rows={len(wf_all):,} | models=rf+{'xgb+' if HAS_XGB else ''}lr")

        train_final(df_all, avail, "clean_general")

    # ── Weather clean model ───────────────────────────────────────────────────
    wx_path = PROC_DIR / "feature_matrix_clean_weather.parquet"
    if wx_path.exists():
        df_wx    = pd.read_parquet(wx_path)
        avail_wx = [c for c in WEATHER_FEATURES + CLEAN_WEATHER_EXTRA
                    if c in df_wx.columns and c not in ("market_price", "price_lag1")]
        avail_wx = [c for c in avail_wx if c in df_wx.columns]

        wf_wx = walk_forward(df_wx, avail_wx, "clean_weather", max_rows=50_000)
        if not wf_wx.empty:
            wf_wx.to_parquet(PROC_DIR / "wf_clean_weather.parquet", index=False)
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
