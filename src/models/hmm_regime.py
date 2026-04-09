"""
src/models/hmm_regime.py
Hidden Markov Model Regime Detection (Simons Framework — A1).

3-state Gaussian HMM per category:
  State 0 = TRENDING       (Kelly 1.0×, follow direction)
  State 1 = MEAN_REVERTING (Kelly 1.5×, bet on reversion)
  State 2 = VOLATILE       (Kelly 0.5×, reduce sizing)

State assignment is done post-hoc by matching state means to regimes:
  Volatile     = highest variance state
  Trending     = highest mean-absolute return state (not volatile)
  Mean-reverting = lowest variance / lowest mean-abs state

Ref: Baum-Welch algorithm (Baum 1972),
     Renaissance Technologies (Zuckerman "The Man Who Solved the Market")

Usage:
    from models.hmm_regime import HMMRegimeDetector
    hrd = HMMRegimeDetector()
    hrd.fit(returns_series)
    regime = hrd.predict_current(returns_series)   # 'TRENDING'|'MEAN_REVERTING'|'VOLATILE'
    mult   = hrd.kelly_multiplier(regime)           # 1.0 | 1.5 | 0.5
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REGIME_LABELS   = {0: "TRENDING", 1: "MEAN_REVERTING", 2: "VOLATILE"}
KELLY_MULTS     = {"TRENDING": 1.0, "MEAN_REVERTING": 1.5, "VOLATILE": 0.5}
N_STATES        = 3
N_ITER          = 200
MIN_OBS         = 60   # need at least 60 days of returns to fit HMM


class HMMRegimeDetector:
    """
    Gaussian HMM with 3 latent states for market regime detection.

    One detector per category (regimes differ significantly across categories).
    Can also run on all-market aggregated returns.
    """

    def __init__(self, n_states: int = N_STATES, n_iter: int = N_ITER):
        self.n_states  = n_states
        self.n_iter    = n_iter
        self.model     = None
        self.state_map: dict[int, str] = {}  # raw state → regime label
        self.fitted    = False

    def _label_states(self, means: np.ndarray, covars: np.ndarray) -> dict[int, str]:
        """
        Assign semantic regime labels to HMM states based on learned parameters.
        - Volatile: highest variance
        - Trending: highest |mean| among non-volatile states
        - Mean-reverting: lowest |mean| and lowest variance
        """
        variances = covars.flatten()
        means_abs = np.abs(means.flatten())

        # Volatile = highest variance
        volatile_state = int(np.argmax(variances))

        remaining = [i for i in range(self.n_states) if i != volatile_state]
        # Trending = highest |mean| among remaining
        trending_state = max(remaining, key=lambda i: means_abs[i])
        # Mean-reverting = the other one
        mr_state = [i for i in remaining if i != trending_state][0]

        return {
            volatile_state: "VOLATILE",
            trending_state: "TRENDING",
            mr_state:       "MEAN_REVERTING",
        }

    def fit(self, returns: pd.Series | np.ndarray) -> "HMMRegimeDetector":
        """
        Fit Gaussian HMM on a return series.

        Args:
            returns: 1D array/Series of returns (not prices!)
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("hmmlearn not installed. Run: pip install hmmlearn")

        if isinstance(returns, pd.Series):
            r = returns.dropna().values
        else:
            r = np.asarray(returns, dtype=float)
            r = r[~np.isnan(r)]

        if len(r) < MIN_OBS:
            print(f"[HMM] Too few observations ({len(r)} < {MIN_OBS}) — using fallback NORMAL state")
            self.fitted = False
            return self

        X = r.reshape(-1, 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                hmm = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=self.n_iter,
                    random_state=42,
                    tol=1e-3,
                )
                hmm.fit(X)
                self.model   = hmm
                self.state_map = self._label_states(hmm.means_, hmm.covars_)
                self.fitted  = True
            except Exception as e:
                print(f"[HMM] Fit failed: {e}")
                self.fitted = False

        return self

    def predict_states(self, returns: pd.Series | np.ndarray) -> np.ndarray:
        """
        Predict hidden states for a return series using Viterbi algorithm.
        Returns array of state indices (0, 1, 2).
        """
        if not self.fitted or self.model is None:
            # Fallback: all TRENDING
            n = len(returns) if hasattr(returns, "__len__") else 1
            return np.zeros(n, dtype=int)

        if isinstance(returns, pd.Series):
            r = returns.dropna().values
        else:
            r = np.asarray(returns, dtype=float)
            r = r[~np.isnan(r)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return self.model.predict(r.reshape(-1, 1))
            except Exception:
                return np.zeros(len(r), dtype=int)

    def predict_regime_series(self, returns: pd.Series) -> pd.Series:
        """
        Predict regime labels for a return series.
        Returns pd.Series of regime labels aligned with input index.
        """
        if not self.fitted or self.model is None:
            return pd.Series("TRENDING", index=returns.dropna().index)

        r = returns.dropna()
        states = self.predict_states(r)
        labels = [self.state_map.get(s, "TRENDING") for s in states]
        return pd.Series(labels, index=r.index)

    def predict_current(self, returns: pd.Series | np.ndarray) -> str:
        """
        Predict the current regime (last observation).
        Returns: 'TRENDING' | 'MEAN_REVERTING' | 'VOLATILE'
        """
        if not self.fitted or self.model is None:
            return "TRENDING"
        states = self.predict_states(returns)
        if len(states) == 0:
            return "TRENDING"
        last_state = int(states[-1])
        return self.state_map.get(last_state, "TRENDING")

    def transition_probs(self) -> pd.DataFrame | None:
        """Return transition probability matrix as DataFrame."""
        if not self.fitted or self.model is None:
            return None
        labels = [self.state_map.get(i, f"S{i}") for i in range(self.n_states)]
        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels,
        )

    def kelly_multiplier(self, regime: str) -> float:
        """Return Kelly size multiplier for a given regime."""
        return KELLY_MULTS.get(regime, 1.0)

    def confidence_score(self, returns: pd.Series | np.ndarray) -> float:
        """Return continuous confidence score 0.0-1.0 based on current regime.

        MEAN_REVERTING -> 0.7 (favorable for mean-reversion strategy)
        TRENDING       -> 0.5 (neutral)
        VOLATILE       -> 0.2 (unfavorable)
        """
        regime = self.predict_current(returns)
        _REGIME_CONFIDENCE = {
            "MEAN_REVERTING": 0.7,
            "TRENDING": 0.5,
            "VOLATILE": 0.2,
        }
        return _REGIME_CONFIDENCE.get(regime, 0.5)

    def save(self, path: Path) -> None:
        joblib.dump({"model": self.model, "state_map": self.state_map,
                     "fitted": self.fitted}, path)

    @classmethod
    def load(cls, path: Path) -> "HMMRegimeDetector":
        data = joblib.load(path)
        obj  = cls()
        obj.model     = data["model"]
        obj.state_map = data["state_map"]
        obj.fitted    = data["fitted"]
        return obj


class CategoryHMMRegistry:
    """
    Manages one HMM per category + one global HMM.
    Fits on wf_clean_general.parquet or any DataFrame with (category, close_date, outcome).
    """

    def __init__(self):
        self.detectors: dict[str, HMMRegimeDetector] = {}
        self.global_detector = HMMRegimeDetector()

    def _compute_returns(self, df: pd.DataFrame,
                         outcome_col: str = "outcome",
                         price_col: str = "price_lag1",
                         date_col: str = "close_date") -> pd.Series:
        """
        Compute daily PnL returns from trades.
        Returns: pd.Series with DatetimeIndex.
        """
        df2 = df.copy()
        df2["_date"] = pd.to_datetime(df2[date_col])
        # Realized P&L proxy: outcome - price_lag1
        if price_col in df2.columns:
            df2["_pnl"] = df2[outcome_col].astype(float) - df2[price_col].fillna(0.5)
        else:
            df2["_pnl"] = df2[outcome_col].astype(float) - 0.5
        # Daily aggregated return
        daily = df2.groupby("_date")["_pnl"].mean().sort_index()
        return daily

    def fit_all(self, df: pd.DataFrame,
                category_col: str = "category") -> "CategoryHMMRegistry":
        """Fit HMM for each category and global."""
        # Global
        global_returns = self._compute_returns(df)
        print(f"[HMM] Fitting global HMM on {len(global_returns)} daily returns...")
        self.global_detector.fit(global_returns)

        # Per-category
        if category_col in df.columns:
            for cat, cat_df in df.groupby(category_col):
                r = self._compute_returns(cat_df)
                if len(r) >= MIN_OBS:
                    hrd = HMMRegimeDetector()
                    hrd.fit(r)
                    self.detectors[str(cat)] = hrd
                    regime = hrd.predict_current(r)
                    mult   = hrd.kelly_multiplier(regime)
                    print(f"  [{cat:<25}] {len(r):>4} days | current={regime:<15} | kelly={mult}×")

        return self

    def annotate_wf(self, df: pd.DataFrame,
                    date_col: str = "close_date",
                    category_col: str = "category") -> pd.DataFrame:
        """
        Add HMM regime features to walk-forward DataFrame.

        Adds: hmm_regime, hmm_kelly_mult, hmm_is_volatile, hmm_is_trending, hmm_is_mr
        """
        out = df.copy()
        out["hmm_regime"]     = "TRENDING"
        out["hmm_kelly_mult"] = 1.0
        out["hmm_is_volatile"]= 0
        out["hmm_is_trending"]= 1
        out["hmm_is_mr"]      = 0

        if not self.detectors and not self.global_detector.fitted:
            return out

        out["_date_dt"] = pd.to_datetime(out[date_col])

        for cat, hrd in self.detectors.items():
            if not hrd.fitted:
                continue
            mask = out[category_col] == cat
            if not mask.any():
                continue

            cat_df = out[mask].copy()
            cat_returns = self._compute_daily_returns_for_cat(cat_df)
            regime_series = hrd.predict_regime_series(cat_returns)

            # Map daily regimes back to rows via date
            regime_by_date = regime_series.to_dict()
            for idx in out.index[mask]:
                d = out.at[idx, "_date_dt"]
                # Find nearest date in regime_series
                regime = regime_by_date.get(d, "TRENDING")
                out.at[idx, "hmm_regime"]      = regime
                out.at[idx, "hmm_kelly_mult"]  = KELLY_MULTS.get(regime, 1.0)
                out.at[idx, "hmm_is_volatile"] = 1 if regime == "VOLATILE" else 0
                out.at[idx, "hmm_is_trending"] = 1 if regime == "TRENDING" else 0
                out.at[idx, "hmm_is_mr"]       = 1 if regime == "MEAN_REVERTING" else 0

        out = out.drop(columns=["_date_dt"], errors="ignore")
        return out

    def _compute_daily_returns_for_cat(self, cat_df: pd.DataFrame) -> pd.Series:
        """Helper: compute daily returns for rows in cat_df."""
        cat_df = cat_df.copy()
        if "price_lag1" in cat_df.columns:
            pnl = cat_df["outcome"].astype(float) - cat_df["price_lag1"].fillna(0.5)
        else:
            pnl = cat_df["outcome"].astype(float) - 0.5
        cat_df["_pnl"] = pnl
        cat_df["_date"] = pd.to_datetime(cat_df["close_date"])
        return cat_df.groupby("_date")["_pnl"].mean().sort_index()

    def save_all(self) -> None:
        for cat, hrd in self.detectors.items():
            safe = cat.replace(" ", "_").replace("/", "_")
            hrd.save(MODEL_DIR / f"hmm_{safe}.joblib")
        if self.global_detector.fitted:
            self.global_detector.save(MODEL_DIR / "hmm_global.joblib")

    def get_regime(self, category: str) -> str:
        hrd = self.detectors.get(category, self.global_detector)
        if not hrd.fitted:
            return "TRENDING"
        # Returns last known regime (needs fresh returns to be accurate)
        return "TRENDING"  # safe default until annotate_wf is called


def fit_and_save(wf_path: str = "data/processed/wf_clean_general.parquet") -> CategoryHMMRegistry:
    """Load WF data, fit all HMMs, save models."""
    path = Path(wf_path)
    if not path.exists():
        print(f"[HMM] {wf_path} not found — run model_clean.py first")
        return CategoryHMMRegistry()

    df = pd.read_parquet(path)
    print(f"[HMM] Loaded {len(df):,} rows from {wf_path}")

    registry = CategoryHMMRegistry()
    registry.fit_all(df)
    registry.save_all()

    print("\n[HMM] Global transition probabilities:")
    tp = registry.global_detector.transition_probs()
    if tp is not None:
        print(tp.round(3).to_string())

    return registry


if __name__ == "__main__":
    fit_and_save()
