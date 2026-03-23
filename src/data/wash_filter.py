"""
src/data/wash_filter.py
Wash Trading Filter for prediction markets.

Ref: Columbia University research (Nov 2025) — 14% of wallets on Polymarket
show activity consistent with wash trading; 20-60% of volume may be artificial.

Filter: >60% volume in <5% of wallets → price not informative → exclude from model.
When wallet data is unavailable, uses heuristic proxies (volume Z-score, round numbers).

Usage:
    from data.wash_filter import WashFilter
    wf = WashFilter()
    clean = wf.filter_markets(markets_df)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

CACHE_PATH                     = Path("data/processed/wash_filter_cache.json")
VOLUME_CONCENTRATION_THRESHOLD = 0.60   # >60% volume in <5% wallets → wash
WALLET_FRACTION_THRESHOLD      = 0.05
MIN_WALLETS                    = 10
WASH_SCORE_THRESHOLD           = 0.85   # top 15% by wash score = flagged


class WashFilter:
    """
    Detects probable wash-traded markets using volume heuristics.
    """

    def __init__(self):
        self.wash_tickers:  set[str] = set()
        self.clean_tickers: set[str] = set()
        self.flags: dict[str, dict]  = {}

    def heuristic_wash_score(self, df: pd.DataFrame,
                              volume_col: str = "volume_fp") -> pd.DataFrame:
        """
        Assign wash trading probability [0,1] using heuristics.
        Adds: wash_score, wash_flag columns.
        """
        out = df.copy()
        if volume_col not in out.columns:
            out["wash_score"] = 0.0
            out["wash_flag"]  = False
            return out

        vol = out[volume_col].fillna(0)

        # H1: volume Z-score by category
        if "category" in out.columns:
            vol_z = out.groupby("category")[volume_col].transform(
                lambda x: (x - x.median()) / (x.std() + 1e-8))
        else:
            vol_z = (vol - vol.median()) / (vol.std() + 1e-8)

        # H2: round-number volume (suspicious pattern)
        round_flag = (vol % 100 == 0) & (vol > 0)

        vol_z_norm         = vol_z.clip(0, 5) / 5.0
        out["wash_score"]  = (vol_z_norm * 0.7 + round_flag.astype(float) * 0.3).clip(0, 1)
        out["wash_flag"]   = out["wash_score"] > WASH_SCORE_THRESHOLD
        return out

    def concentration_score(self, wallet_volumes: dict[str, float]) -> float:
        """Herfindahl-Hirschman Index for volume concentration."""
        total = sum(wallet_volumes.values())
        if total <= 0:
            return 0.0
        shares = [v / total for v in wallet_volumes.values()]
        return sum(s ** 2 for s in shares)

    def is_wash_by_concentration(self, wallet_volumes: dict[str, float]) -> bool:
        if len(wallet_volumes) < MIN_WALLETS:
            return False
        total = sum(wallet_volumes.values())
        if total <= 0:
            return False
        top_n = max(1, int(len(wallet_volumes) * WALLET_FRACTION_THRESHOLD))
        top_vols = sorted(wallet_volumes.values(), reverse=True)[:top_n]
        return sum(top_vols) / total > VOLUME_CONCENTRATION_THRESHOLD

    def filter_markets(self, df: pd.DataFrame,
                       volume_col: str = "volume_fp") -> pd.DataFrame:
        """Remove wash-trading-flagged rows. Returns clean subset."""
        scored = self.heuristic_wash_score(df, volume_col)
        clean  = scored[~scored["wash_flag"]].drop(
            columns=["wash_score", "wash_flag"], errors="ignore")
        removed = len(df) - len(clean)
        if removed > 0:
            print(f"[WashFilter] Removed {removed:,} rows ({removed/len(df):.1%}) as suspected wash")
        return clean

    def save(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "wash_tickers":  list(self.wash_tickers),
            "clean_tickers": list(self.clean_tickers),
            "flags":         self.flags,
        }, indent=2))

    @classmethod
    def load(cls, path: Path = CACHE_PATH) -> "WashFilter":
        obj = cls()
        if not path.exists():
            return obj
        data = json.loads(path.read_text())
        obj.wash_tickers  = set(data.get("wash_tickers", []))
        obj.clean_tickers = set(data.get("clean_tickers", []))
        obj.flags         = data.get("flags", {})
        return obj


if __name__ == "__main__":
    path = Path("data/processed/trades_clean.parquet")
    if not path.exists():
        print("[WashFilter] trades_clean.parquet not found")
    else:
        trades = pd.read_parquet(path)
        wf = WashFilter()
        scored = wf.heuristic_wash_score(trades)
        n = scored["wash_flag"].sum()
        print(f"[WashFilter] {len(trades):,} total | {n:,} flagged ({n/len(trades):.1%})")
        if "category" in scored.columns:
            by_cat = scored.groupby("category")["wash_flag"].mean().sort_values(ascending=False)
            for cat, rate in by_cat.items():
                print(f"  {cat:<30} {rate:.1%}")
