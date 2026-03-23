"""
src/risk/expiry_gamma.py
Gamma-Near-Expiry Position Sizing (Academic Alpha — B4).

Ref: Navnoor Bawa "Mathematical Execution Behind Prediction Market Alpha" (Dec 2025).

Binary options near expiry exhibit exponential "gamma" (sensitivity to price moves).
Position sizing near expiry:

    size = base_size * sqrt(days_remaining / total_market_duration)

Examples:
  7 days remaining on 30-day market: size = base * sqrt(7/30) = base * 0.483
  1 day remaining on 30-day market:  size = base * sqrt(1/30) = base * 0.183
  Last week (7d): reduce 65%  → size = base * 0.35
  Last day:       reduce 82%  → size = base * 0.18

Intuition: near expiry, prediction markets converge to binary outcomes rapidly.
The implied volatility (gamma) spikes because small probability shifts = large
price moves. Large positions become too risky relative to potential gain.

Usage:
    from risk.expiry_gamma import ExpiryGamma
    eg = ExpiryGamma()
    mult = eg.sizing_multiplier(days_remaining=3, total_duration=30)
    size = base_size * mult
    df = eg.annotate(trades_df)  # adds gamma_mult column
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Thresholds (in days)
GAMMA_CRITICAL_DAYS = 1   # last day: 82% reduction
GAMMA_WARNING_DAYS  = 7   # last week: 65% reduction
GAMMA_CAUTION_DAYS  = 14  # last 2 weeks: ~43% reduction

# Min/max multipliers
MIN_MULT = 0.05   # never go below 5% size
MAX_MULT = 1.00   # never exceed base size


class ExpiryGamma:
    """
    Computes position size multiplier based on time-to-expiry (gamma exposure).

    The key insight from Bawa (2025): near expiry in prediction markets,
    position sizing must account for nonlinear price sensitivity.
    The sqrt model is a conservative approximation of the true gamma curve.
    """

    def sizing_multiplier(self, days_remaining: float,
                          total_duration: float,
                          model: str = "sqrt") -> float:
        """
        Compute position size multiplier.

        Args:
            days_remaining: days until market settlement
            total_duration: total market duration in days
            model: 'sqrt' (Bawa default) | 'linear' | 'exponential'

        Returns: multiplier in [MIN_MULT, 1.0]
        """
        if total_duration <= 0:
            return 1.0
        days_remaining = max(0.0, float(days_remaining))
        ratio = days_remaining / max(total_duration, 1.0)
        ratio = min(ratio, 1.0)

        if model == "sqrt":
            mult = np.sqrt(ratio)
        elif model == "linear":
            mult = ratio
        elif model == "exponential":
            # More aggressive: exp(-lambda * (1-ratio))
            mult = np.exp(-2.0 * (1 - ratio))
        else:
            mult = np.sqrt(ratio)

        return float(np.clip(mult, MIN_MULT, MAX_MULT))

    def expiry_bucket(self, days_remaining: float) -> str:
        """Classify market into expiry risk bucket."""
        if days_remaining <= GAMMA_CRITICAL_DAYS:
            return "CRITICAL"   # last day: -82%
        elif days_remaining <= GAMMA_WARNING_DAYS:
            return "WARNING"    # last week: -65%
        elif days_remaining <= GAMMA_CAUTION_DAYS:
            return "CAUTION"    # last 2 weeks: -43%
        else:
            return "NORMAL"     # full size

    def reduction_pct(self, days_remaining: float,
                      total_duration: float = 30.0) -> float:
        """Return % reduction from base size (0.65 = 65% reduced = 35% of base)."""
        mult = self.sizing_multiplier(days_remaining, total_duration)
        return 1.0 - mult

    def annotate(self, df: pd.DataFrame,
                 open_date_col: str = "open_date",
                 close_date_col: str = "close_date",
                 ref_date_col: str | None = None) -> pd.DataFrame:
        """
        Add gamma sizing columns to a trades/WF DataFrame.

        Uses close_date - open_date as total_duration.
        Uses close_date - ref_date (or today) as days_remaining.

        Adds:
          gamma_days_remaining   — days until settlement
          gamma_total_duration   — total market lifetime
          gamma_mult             — size multiplier [0.05, 1.0]
          gamma_bucket           — 'NORMAL'|'CAUTION'|'WARNING'|'CRITICAL'
          gamma_reduction_pct    — 0.0 to 0.95 (fraction of base to reduce)
        """
        out = df.copy()
        out["gamma_mult"]          = 1.0
        out["gamma_days_remaining"]= np.nan
        out["gamma_total_duration"]= np.nan
        out["gamma_bucket"]        = "NORMAL"
        out["gamma_reduction_pct"] = 0.0

        if open_date_col not in out.columns or close_date_col not in out.columns:
            return out

        close_dt = pd.to_datetime(out[close_date_col], errors="coerce")
        open_dt  = pd.to_datetime(out[open_date_col],  errors="coerce")

        total_dur = (close_dt - open_dt).dt.days.clip(lower=1).fillna(1)

        if ref_date_col and ref_date_col in out.columns:
            ref_dt = pd.to_datetime(out[ref_date_col], errors="coerce")
        else:
            # Use close_date as reference (backtest mode — trade entered at open)
            ref_dt = open_dt

        days_rem = (close_dt - ref_dt).dt.days.clip(lower=0).fillna(0)

        out["gamma_days_remaining"] = days_rem
        out["gamma_total_duration"] = total_dur

        out["gamma_mult"] = np.vectorize(self.sizing_multiplier)(
            days_rem.values, total_dur.values)
        out["gamma_bucket"] = np.vectorize(self.expiry_bucket)(days_rem.values)
        out["gamma_reduction_pct"] = (1 - out["gamma_mult"]).round(4)

        return out

    def apply_to_pnl(self, trades: pd.DataFrame,
                     pnl_col: str = "pnl",
                     size_col: str = "gamma_mult") -> pd.DataFrame:
        """
        Scale P&L by gamma multiplier (simulates reduced position size).
        Creates gamma_pnl column.
        """
        out = trades.copy()
        if size_col not in out.columns:
            out = self.annotate(out)
        if pnl_col in out.columns and size_col in out.columns:
            out[f"gamma_{pnl_col}"] = out[pnl_col] * out[size_col]
        return out


def compute_gamma_table() -> pd.DataFrame:
    """Reference table of gamma multipliers for common market durations."""
    eg = ExpiryGamma()
    rows = []
    for total_dur in [1, 3, 7, 14, 30, 60, 90]:
        for days_rem in [0.5, 1, 2, 3, 5, 7, 14, 21, total_dur]:
            if days_rem > total_dur:
                continue
            mult = eg.sizing_multiplier(days_rem, total_dur)
            rows.append({
                "total_duration": total_dur,
                "days_remaining": days_rem,
                "multiplier":     round(mult, 4),
                "reduction_pct":  f"{(1-mult)*100:.0f}%",
                "bucket":         eg.expiry_bucket(days_rem),
            })
    return pd.DataFrame(rows)


def run():
    eg = ExpiryGamma()
    print("[ExpiryGamma] Gamma sizing reference table (30-day market):")
    print("-" * 70)
    print(f"{'Days remaining':>15} {'Multiplier':>12} {'Reduction':>12} {'Bucket':>12}")
    print("-" * 70)
    for days_rem in [30, 21, 14, 7, 3, 2, 1, 0.5]:
        mult   = eg.sizing_multiplier(days_rem, 30)
        bucket = eg.expiry_bucket(days_rem)
        print(f"{days_rem:>15.1f} {mult:>12.4f} {(1-mult)*100:>11.1f}% {bucket:>12}")

    print("\n[ExpiryGamma] Key thresholds (Bawa 2025):")
    print(f"  Last day  (d=1,  T=30): mult = {eg.sizing_multiplier(1, 30):.3f} "
          f"→ {eg.reduction_pct(1, 30)*100:.0f}% reduction (Bawa: ~82%)")
    print(f"  Last week (d=7,  T=30): mult = {eg.sizing_multiplier(7, 30):.3f} "
          f"→ {eg.reduction_pct(7, 30)*100:.0f}% reduction (Bawa: ~65%)")

    # Test annotate on sample data
    sample = pd.DataFrame({
        "open_date":  ["2025-01-01", "2025-01-01", "2025-01-01"],
        "close_date": ["2025-01-31", "2025-01-08", "2025-01-02"],
        "pnl": [0.95, 0.95, 0.95],
    })
    annotated = eg.annotate(sample)
    print("\n[ExpiryGamma] Sample annotation:")
    print(annotated[["gamma_days_remaining", "gamma_total_duration",
                      "gamma_mult", "gamma_bucket"]].to_string())


if __name__ == "__main__":
    run()
