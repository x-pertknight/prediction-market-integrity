"""
src/alpha/favorite_longshot.py
Favorite-Longshot Bias Detector for Kalshi prediction markets.

Ref: UCD Working Paper WP2025_19 "Makers and Takers" (Kalshi data).
Longshot bias: low-probability contracts (1-10¢) are systematically overpriced.
Favorite bias: high-probability contracts (90-99¢) are systematically underpriced.

bias[bucket] = historical_win_rate[bucket] - midpoint_of_bucket
Positive bias → overpriced (market_price > true prob)
Negative bias → underpriced (buying opportunity)

Usage:
    from alpha.favorite_longshot import FavoriteLongshotBias
    flb = FavoriteLongshotBias()
    flb.fit(trades_df)   # trades_df has entry_price, outcome columns
    bias = flb.get_bias(0.05)   # returns bias for a 5¢ contract
    df = flb.annotate(wf_df)    # adds fl_bias, fl_bucket, fl_adjusted_prob columns
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

CACHE_PATH = Path("data/processed/fl_bias.json")
BUCKETS = [(i/10, (i+1)/10) for i in range(10)]  # 10 buckets: 0-0.1, 0.1-0.2, ...


class FavoriteLongshotBias:
    """
    Computes and applies the favorite-longshot bias correction.
    
    Attributes:
        bias_table: dict {bucket_midpoint: bias_value}
            bias > 0 → market is overpriced vs calibrated probability
            bias < 0 → market is underpriced (opportunity)
    """

    def __init__(self):
        self.bias_table: dict[float, float] = {}
        self.win_rate_table: dict[float, float] = {}
        self.count_table: dict[float, int] = {}

    def _bucket_midpoint(self, price: float) -> float:
        """Return the midpoint of the price bucket for a given price."""
        price = float(np.clip(price, 0.0, 1.0))
        bucket_idx = min(int(price * 10), 9)
        low, high = bucket_idx / 10, (bucket_idx + 1) / 10
        return (low + high) / 2

    def fit(self, trades: pd.DataFrame,
            price_col: str = "entry_price",
            outcome_col: str = "outcome") -> "FavoriteLongshotBias":
        """
        Fit bias table from historical trades.
        
        Args:
            trades: DataFrame with price_col and outcome_col
            price_col: column with entry price (0-1 scale)
            outcome_col: column with binary outcome (1=YES won, 0=NO won)
        """
        if trades.empty or price_col not in trades.columns:
            return self

        df = trades[[price_col, outcome_col]].dropna()
        df = df[(df[price_col] > 0) & (df[price_col] < 1)]

        for i in range(10):
            low, high = i / 10, (i + 1) / 10
            mid = (low + high) / 2
            bucket = df[(df[price_col] >= low) & (df[price_col] < high)]
            if len(bucket) < 5:
                self.bias_table[mid] = 0.0
                self.win_rate_table[mid] = mid
                self.count_table[mid] = len(bucket)
                continue
            actual_win_rate = bucket[outcome_col].mean()
            # bias = actual_win_rate - midpoint
            # positive: actual > market price → market was overpriced
            # negative: actual < market price → market was underpriced (buy!)
            self.bias_table[mid] = round(actual_win_rate - mid, 4)
            self.win_rate_table[mid] = round(actual_win_rate, 4)
            self.count_table[mid] = len(bucket)

        self.save()
        return self

    def get_bias(self, price: float) -> float:
        """Return bias for a given price. Positive = overpriced."""
        mid = self._bucket_midpoint(price)
        return self.bias_table.get(mid, 0.0)

    def adjusted_prob(self, price: float) -> float:
        """
        Return calibration-adjusted probability.
        adjusted = market_price - bias
        If bias = +0.05: market overprices by 5¢ → adjust down
        """
        return float(np.clip(price - self.get_bias(price), 0.01, 0.99))

    def is_underpriced(self, price: float, threshold: float = 0.05) -> bool:
        """True if market is systematically underpriced (bias < -threshold)."""
        return self.get_bias(price) < -threshold

    def is_overpriced(self, price: float, threshold: float = 0.05) -> bool:
        """True if market is systematically overpriced (bias > threshold)."""
        return self.get_bias(price) > threshold

    def annotate(self, df: pd.DataFrame,
                 price_col: str = "price_lag1") -> pd.DataFrame:
        """
        Add favorite-longshot bias columns to a DataFrame.
        Adds: fl_bias, fl_bucket_mid, fl_adjusted_prob, fl_underpriced, fl_overpriced
        """
        out = df.copy()
        if price_col not in out.columns or not self.bias_table:
            out["fl_bias"] = 0.0
            out["fl_bucket_mid"] = 0.05
            out["fl_adjusted_prob"] = out.get(price_col, pd.Series(0.05, index=out.index))
            out["fl_underpriced"] = False
            out["fl_overpriced"] = False
            return out

        prices = out[price_col].clip(0.001, 0.999).fillna(0.05)
        out["fl_bucket_mid"]      = prices.map(self._bucket_midpoint)
        out["fl_bias"]            = out["fl_bucket_mid"].map(
                                        lambda m: self.bias_table.get(m, 0.0))
        out["fl_adjusted_prob"]   = (prices - out["fl_bias"]).clip(0.01, 0.99)
        out["fl_underpriced"]     = out["fl_bias"] < -0.05  # buying opportunity
        out["fl_overpriced"]      = out["fl_bias"] > 0.05   # avoid going long
        return out

    def summary_table(self) -> pd.DataFrame:
        """Return a summary of bias by bucket."""
        rows = []
        for mid, bias in sorted(self.bias_table.items()):
            rows.append({
                "bucket":      f"{mid-0.05:.0%}–{mid+0.05:.0%}",
                "mid":         mid,
                "win_rate":    self.win_rate_table.get(mid, mid),
                "bias":        bias,
                "n":           self.count_table.get(mid, 0),
                "signal":      "OVERPRICED" if bias > 0.05 else
                               "UNDERPRICED" if bias < -0.05 else "NEUTRAL",
            })
        return pd.DataFrame(rows)

    def save(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bias_table":     {str(k): v for k, v in self.bias_table.items()},
            "win_rate_table": {str(k): v for k, v in self.win_rate_table.items()},
            "count_table":    {str(k): v for k, v in self.count_table.items()},
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path = CACHE_PATH) -> "FavoriteLongshotBias":
        obj = cls()
        if not path.exists():
            return obj
        data = json.loads(path.read_text())
        obj.bias_table     = {float(k): v for k, v in data.get("bias_table", {}).items()}
        obj.win_rate_table = {float(k): v for k, v in data.get("win_rate_table", {}).items()}
        obj.count_table    = {float(k): v for k, v in data.get("count_table", {}).items()}
        return obj


def compute_and_save(trades_path: str = "data/processed/trades_clean.parquet") -> FavoriteLongshotBias:
    """Load trades, fit FLB model, print summary."""
    if not Path(trades_path).exists():
        print(f"[FLB] {trades_path} not found — returning empty model")
        return FavoriteLongshotBias()

    trades = pd.read_parquet(trades_path)
    flb = FavoriteLongshotBias()
    flb.fit(trades)
    print("[FLB] Favorite-Longshot Bias by bucket:")
    tbl = flb.summary_table()
    for _, row in tbl.iterrows():
        bar = "▓" * int(abs(row["bias"]) * 100)
        direction = "+" if row["bias"] > 0 else ""
        print(f"  {row['bucket']:12s}  n={row['n']:>6}  win={row['win_rate']:.3f}  "
              f"bias={direction}{row['bias']:.3f} {bar}  [{row['signal']}]")
    return flb


if __name__ == "__main__":
    compute_and_save()
