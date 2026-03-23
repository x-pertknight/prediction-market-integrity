"""
src/models/cross_market.py
Cross-market correlation and cointegration analysis.

For pairs of markets within the same category:
  - Rolling 30-day correlation of market prices
  - Engle-Granger cointegration test
  - Spread Z-score for mean-reversion signals

Features added to model:
  spread_z_score, cross_corr_30d, cointegration_pvalue

If spread diverges >2 sigma: mean-reversion signal (buy the laggard, sell the leader).

Usage:
    from models.cross_market import CrossMarketAnalyzer
    cma = CrossMarketAnalyzer()
    features = cma.compute_features(wf_df)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from statsmodels.tsa.stattools import coint
import warnings

CACHE_DIR = Path("data/processed")
ROLLING_WINDOW = 30
SPREAD_THRESHOLD = 2.0   # sigma threshold for mean-reversion signal


class CrossMarketAnalyzer:
    """
    Computes cross-market correlation and cointegration features.
    
    For each category with multiple series, finds pairs that are
    cointegrated and tracks spread divergence.
    """

    def __init__(self, rolling_window: int = ROLLING_WINDOW):
        self.window = rolling_window
        self.cointegrated_pairs: dict[str, list[tuple]] = {}

    def _get_price_series(self, df: pd.DataFrame,
                          series_col: str = "series_ticker",
                          price_col: str = "price_lag1",
                          date_col: str = "close_date") -> pd.DataFrame:
        """
        Pivot market data into a wide price matrix.
        Rows = dates, Columns = series_ticker.
        """
        df2 = df[[date_col, series_col, price_col]].dropna()
        df2[date_col] = pd.to_datetime(df2[date_col])
        # One price per (date, series): take last
        df2 = df2.sort_values(date_col)
        pivot = df2.groupby([date_col, series_col])[price_col].last().unstack(
            series_col).sort_index()
        return pivot

    def find_cointegrated_pairs(self, price_matrix: pd.DataFrame,
                                category: str = "") -> list[tuple]:
        """
        Find cointegrated pairs in a price matrix using Engle-Granger test.
        Returns list of (series_a, series_b, pvalue).
        """
        pairs = []
        cols = price_matrix.dropna(axis=1, how="all").columns.tolist()
        if len(cols) < 2:
            return pairs

        for col_a, col_b in combinations(cols[:20], 2):  # limit pairs for speed
            s_a = price_matrix[col_a].dropna()
            s_b = price_matrix[col_b].dropna()
            common = s_a.index.intersection(s_b.index)
            if len(common) < 30:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, pvalue, _ = coint(s_a.loc[common], s_b.loc[common])
                if pvalue < 0.05:
                    pairs.append((col_a, col_b, round(pvalue, 4)))
            except Exception:
                continue

        if category and pairs:
            self.cointegrated_pairs[category] = pairs
        return pairs

    def rolling_correlation(self, s_a: pd.Series, s_b: pd.Series) -> pd.Series:
        """Rolling 30-day correlation between two price series."""
        common = s_a.index.intersection(s_b.index)
        if len(common) < self.window:
            return pd.Series(dtype=float)
        df = pd.DataFrame({"a": s_a, "b": s_b}).loc[common]
        return df["a"].rolling(self.window).corr(df["b"])

    def spread_zscore(self, s_a: pd.Series, s_b: pd.Series,
                      window: int = 30) -> pd.Series:
        """
        Compute rolling spread Z-score between two cointegrated series.
        Z > +2: spread too wide, expect convergence
        Z < -2: spread too narrow, expect divergence
        """
        common = s_a.index.intersection(s_b.index)
        if len(common) < window:
            return pd.Series(dtype=float)
        spread = s_a.loc[common] - s_b.loc[common]
        rolling_mean = spread.rolling(window).mean()
        rolling_std  = spread.rolling(window).std()
        return (spread - rolling_mean) / (rolling_std + 1e-8)

    def compute_features(self, df: pd.DataFrame,
                         category_col: str = "category",
                         series_col: str = "series_ticker",
                         price_col: str = "price_lag1",
                         date_col: str = "close_date") -> pd.DataFrame:
        """
        Add cross-market features to walk-forward DataFrame.
        
        Adds:
          cross_corr_30d       -- rolling 30-day correlation with most correlated peer
          spread_z_score       -- spread Z-score vs most correlated peer
          cointegration_pvalue -- min p-value from Engle-Granger test
          cross_mean_rev_signal -- 1 if spread > 2 sigma, else 0
        """
        out = df.copy()
        out["cross_corr_30d"]        = 0.5   # default (no info)
        out["spread_z_score"]        = 0.0
        out["cointegration_pvalue"]  = 0.5
        out["cross_mean_rev_signal"] = 0

        if category_col not in out.columns or series_col not in out.columns:
            return out

        for category, cat_df in out.groupby(category_col):
            series_list = cat_df[series_col].unique()
            if len(series_list) < 2:
                continue

            price_matrix = self._get_price_series(
                cat_df, series_col, price_col, date_col)
            if price_matrix.shape[1] < 2:
                continue

            # Find cointegrated pairs (expensive, cache result)
            pairs = self.find_cointegrated_pairs(price_matrix, str(category))
            if not pairs:
                continue

            # For each market in category, find best correlated peer and compute spread
            for series_a, series_b, pval in pairs[:5]:  # top 5 cointegrated pairs
                if series_a not in price_matrix.columns or                    series_b not in price_matrix.columns:
                    continue

                s_a = price_matrix[series_a].dropna()
                s_b = price_matrix[series_b].dropna()
                corr = self.rolling_correlation(s_a, s_b)
                zscore = self.spread_zscore(s_a, s_b)

                # Update rows for series_a
                mask_a = (out[category_col] == category) &                          (out[series_col] == series_a)
                if mask_a.any() and not corr.empty:
                    dates_a = pd.to_datetime(out.loc[mask_a, date_col])
                    for idx in out.index[mask_a]:
                        d = pd.to_datetime(out.at[idx, date_col])
                        if d in corr.index:
                            c = corr.get(d, 0.5)
                            z = zscore.get(d, 0.0)
                            out.at[idx, "cross_corr_30d"]        = c if not np.isnan(c) else 0.5
                            out.at[idx, "spread_z_score"]        = z if not np.isnan(z) else 0.0
                            out.at[idx, "cointegration_pvalue"]  = pval
                            out.at[idx, "cross_mean_rev_signal"] = 1 if abs(z) > SPREAD_THRESHOLD else 0

        return out

    def sizing_multiplier(self, spread_z: float) -> float:
        """
        Kelly multiplier based on spread Z-score.
        Mean-reverting signal (|z| > 2): 1.5x size
        No signal: 1.0x
        """
        if abs(spread_z) > SPREAD_THRESHOLD:
            return 1.5  # mean-reversion opportunity
        return 1.0


def run():
    print("[CrossMarket] Loading wf_clean_general.parquet...")
    path = Path("data/processed/wf_clean_general.parquet")
    if not path.exists():
        print("  File not found. Run features_clean.py + model_clean.py first.")
        return

    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows, {df['category'].nunique()} categories")

    cma = CrossMarketAnalyzer()
    # Test on a small subset (full computation is expensive)
    subset = df.head(5000)
    features = cma.compute_features(subset)
    print("\n[CrossMarket] Features added:")
    for col in ["cross_corr_30d", "spread_z_score", "cointegration_pvalue", "cross_mean_rev_signal"]:
        if col in features.columns:
            print(f"  {col}: mean={features[col].mean():.4f}, non-zero={features[col].ne(0).sum()}")


if __name__ == "__main__":
    run()
