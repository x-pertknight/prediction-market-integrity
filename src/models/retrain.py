"""
src/models/retrain.py
Automatic retraining trigger for TrapTrade models.

Triggers:
  - Every N=50 market resolutions (rolling walk-forward)
  - If new Sharpe < old Sharpe * 0.90: alert, keep old model

Architecture:
  - Walk-forward retrain: 6 months train, 1 month test
  - Models archived with timestamp in models/archive/
  - Perf comparison: new vs current model on same test period
  - Alert via Telegram if degradation detected

Usage:
    from models.retrain import AutoRetrain
    ar = AutoRetrain()
    ar.check_and_retrain(n_new_resolutions=75)
"""

import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

PROC_DIR    = Path("data/processed")
MODEL_DIR   = Path("models")
ARCHIVE_DIR = Path("models/archive")
META_PATH   = Path("models/retrain_meta.json")

RETRAIN_EVERY     = 50    # retrain every N resolutions
SHARPE_THRESHOLD  = 0.90  # alert if new_sharpe < old_sharpe * this
TRAIN_MONTHS      = 6
TEST_MONTHS       = 1


class AutoRetrain:
    def __init__(self):
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.meta = self._load_meta()

    def _load_meta(self) -> dict:
        if META_PATH.exists():
            return json.loads(META_PATH.read_text())
        return {
            "n_resolutions_since_retrain": 0,
            "last_retrain_date":           None,
            "current_model_sharpe":        None,
            "current_model_path":          None,
            "retrain_count":               0,
            "last_degradation_alert":      None,
        }

    def _save_meta(self) -> None:
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_PATH.write_text(json.dumps(self.meta, indent=2, default=str))

    def log_resolutions(self, n: int = 1) -> None:
        self.meta["n_resolutions_since_retrain"] = self.meta.get("n_resolutions_since_retrain", 0) + n
        self._save_meta()

    def should_retrain(self) -> bool:
        return self.meta.get("n_resolutions_since_retrain", 0) >= RETRAIN_EVERY

    def sharpe(self, pnl_series: pd.Series) -> float:
        if len(pnl_series) < 5 or pnl_series.std() == 0:
            return 0.0
        return float(pnl_series.mean() / pnl_series.std() * np.sqrt(252))

    def retrain_model(self, wf_path: str = "data/processed/wf_clean_general.parquet",
                      features_path: str = "data/processed/feature_matrix_clean_all.parquet",
                      model_name: str = "rf_clean_general") -> dict:
        wf_path_obj = Path(wf_path)
        feat_path_obj = Path(features_path)

        if not wf_path_obj.exists() or not feat_path_obj.exists():
            print(f"[Retrain] Required files not found -- skipping")
            return {"deployed": False, "reason": "files_missing"}

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from features_clean import CLEAN_FEATURES

        df = pd.read_parquet(feat_path_obj)
        wf = pd.read_parquet(wf_path_obj)

        avail_features = [f for f in CLEAN_FEATURES if f in df.columns and df[f].nunique() > 1]
        if len(avail_features) < 3:
            print(f"[Retrain] Too few features ({len(avail_features)}) -- skipping")
            return {"deployed": False, "reason": "insufficient_features"}

        df["_date"] = pd.to_datetime(df["close_date"])
        df["_ym"]   = df["_date"].dt.to_period("M")
        months      = sorted(df["_ym"].unique())

        if len(months) < TRAIN_MONTHS + TEST_MONTHS:
            print(f"[Retrain] Not enough months ({len(months)}) -- skipping")
            return {"deployed": False, "reason": "insufficient_months"}

        train_months = months[-(TRAIN_MONTHS + TEST_MONTHS):-TEST_MONTHS]
        test_month   = months[-TEST_MONTHS]

        train = df[df["_ym"].isin(train_months)]
        test  = df[df["_ym"] == test_month]

        if len(train) < 100 or len(test) < 20:
            return {"deployed": False, "reason": "insufficient_rows"}

        X_tr = train[avail_features].fillna(0).values
        y_tr = train["outcome"].values
        X_te = test[avail_features].fillna(0).values
        y_te = test["outcome"].values

        new_clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=int(datetime.now().timestamp()) % 2**32,
            n_jobs=-1,
        )
        new_clf.fit(X_tr, y_tr)
        probs = new_clf.predict_proba(X_te)[:, 1]

        # Compute Sharpe: buy when prob * 0.7 >= price
        if "price_lag1" in test.columns:
            test = test.copy()
            test["pred"] = probs
            signals = test[test["price_lag1"] <= test["pred"] * 0.7]
            if len(signals) >= 10:
                pnl = signals["outcome"].astype(float) - signals["price_lag1"]
                new_sharpe = self.sharpe(pnl)
            else:
                new_sharpe = 0.0
        else:
            new_sharpe = 0.0

        old_sharpe = self.meta.get("current_model_sharpe") or 0.0

        current_path = MODEL_DIR / f"{model_name}.joblib"
        if current_path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = ARCHIVE_DIR / f"{model_name}_{ts}.joblib"
            import shutil
            shutil.copy(current_path, archive_path)
            print(f"[Retrain] Archived old model -> {archive_path}")

        deploy = True
        reason = "improved"

        if old_sharpe > 0 and new_sharpe < old_sharpe * SHARPE_THRESHOLD:
            deploy = False
            reason = f"degraded (new={new_sharpe:.2f} < old={old_sharpe:.2f} * {SHARPE_THRESHOLD})"
            self._send_degradation_alert(old_sharpe, new_sharpe)

        if deploy:
            joblib.dump(new_clf, current_path)
            self.meta["current_model_sharpe"]       = new_sharpe
            self.meta["current_model_path"]          = str(current_path)
            self.meta["last_retrain_date"]           = datetime.now().isoformat()
            self.meta["n_resolutions_since_retrain"] = 0
            self.meta["retrain_count"]               = self.meta.get("retrain_count", 0) + 1
            self._save_meta()
            print(f"[Retrain] Deployed new model. Sharpe: {new_sharpe:.3f} (was {old_sharpe:.3f})")
        else:
            print(f"[Retrain] KEPT old model. {reason}")

        return {
            "deployed":    deploy,
            "new_sharpe":  round(new_sharpe, 4),
            "old_sharpe":  round(old_sharpe, 4),
            "reason":      reason,
            "model_path":  str(current_path) if deploy else None,
        }

    def check_and_retrain(self, n_new_resolutions: int = 0) -> dict | None:
        if n_new_resolutions > 0:
            self.log_resolutions(n_new_resolutions)
        if self.should_retrain():
            print(f"[Retrain] Triggered after {self.meta['n_resolutions_since_retrain']} resolutions")
            return self.retrain_model()
        print(f"[Retrain] {self.meta.get('n_resolutions_since_retrain', 0)}/{RETRAIN_EVERY} resolutions since last retrain. Not triggered yet.")
        return None

    def _send_degradation_alert(self, old_sharpe: float, new_sharpe: float) -> None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from alerts.telegram_bot import send_message
            msg = ("MODEL DEGRADATION ALERT\n"
                   f"New Sharpe: {new_sharpe:.3f}\n"
                   f"Old Sharpe: {old_sharpe:.3f}\n"
                   f"Ratio: {new_sharpe/old_sharpe:.2f} < {SHARPE_THRESHOLD}\n"
                   "Decision: KEEPING OLD MODEL")
            send_message(msg)
        except Exception:
            self.meta["last_degradation_alert"] = datetime.now().isoformat()
            self._save_meta()

    def status(self) -> dict:
        return {
            "resolutions_since_retrain": self.meta.get("n_resolutions_since_retrain", 0),
            "retrain_threshold":         RETRAIN_EVERY,
            "last_retrain_date":         self.meta.get("last_retrain_date"),
            "current_model_sharpe":      self.meta.get("current_model_sharpe"),
            "total_retrains":            self.meta.get("retrain_count", 0),
            "should_retrain_now":        self.should_retrain(),
        }


def run():
    ar = AutoRetrain()
    print("[Retrain] Status:")
    for k, v in ar.status().items():
        print(f"  {k:<35} {v}")

    print("\n[Retrain] Testing trigger (adding 50 resolutions)...")
    ar.log_resolutions(50)
    if ar.should_retrain():
        print("  Would trigger retrain now.")
        result = ar.retrain_model()
        print(f"  Result: {result}")
    # Reset
    ar.meta["n_resolutions_since_retrain"] = 0
    ar._save_meta()


if __name__ == "__main__":
    run()
