"""
src/models/block_bootstrap.py — Block Bootstrap for autocorrelated trade P&L.

Standard i.i.d. bootstrap underestimates variance when trades within the
same series are correlated. Block bootstrap preserves intra-series dependence
by resampling contiguous blocks (grouped by series_ticker).

Method:
  1. Group trades by series_ticker → blocks of consecutive trades.
  2. Resample blocks WITH replacement until total trades ≈ original N.
  3. Compute statistic (Sharpe, total P&L, etc.) on resampled trades.
  4. Repeat B times → distribution of the statistic.
  5. CI from percentiles of the bootstrap distribution.

Ref: Politis & Romano (1994) "The Stationary Bootstrap."
     Lahiri (2003) "Resampling Methods for Dependent Data."
"""

import numpy as np
import pandas as pd
from typing import Callable


def _default_sharpe(pnl: np.ndarray) -> float:
    """Annualized Sharpe ratio (√252 scaling)."""
    if len(pnl) < 2 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def block_bootstrap(trades: pd.DataFrame,
                    block_col: str = "series_ticker",
                    pnl_col: str = "pnl",
                    stat_fn: Callable[[np.ndarray], float] | None = None,
                    n_resamples: int = 1000,
                    ci: float = 0.95,
                    seed: int = 42) -> dict:
    """
    Block bootstrap CI for a statistic computed on trade P&L.

    Args:
        trades:       DataFrame with at least `block_col` and `pnl_col`.
        block_col:    Column to group blocks by (default: series_ticker).
        pnl_col:      Column with per-trade P&L.
        stat_fn:      Function f(pnl_array) → float. Default: annualized Sharpe.
        n_resamples:  Number of bootstrap resamples (default 1000).
        ci:           Confidence level (default 0.95).
        seed:         Random seed for reproducibility.

    Returns:
        {
            "stat":         float   — point estimate on original data
            "ci_lower":     float   — lower CI bound
            "ci_upper":     float   — upper CI bound
            "se":           float   — bootstrap standard error
            "n_blocks":     int     — number of unique blocks
            "n_resamples":  int     — number of resamples performed
            "block_sizes":  dict    — {min, max, mean, median} block sizes
        }
    """
    if stat_fn is None:
        stat_fn = _default_sharpe

    if block_col not in trades.columns:
        # Fallback: treat each trade as its own block (standard bootstrap)
        block_col = None

    # Build blocks
    if block_col is not None:
        blocks = []
        for _, group in trades.groupby(block_col, sort=False):
            blocks.append(group[pnl_col].values)
    else:
        # Each trade is a block of size 1
        blocks = [np.array([v]) for v in trades[pnl_col].values]

    n_blocks = len(blocks)
    if n_blocks == 0:
        return {
            "stat": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "se": 0.0, "n_blocks": 0, "n_resamples": 0,
            "block_sizes": {"min": 0, "max": 0, "mean": 0, "median": 0},
        }

    # Point estimate
    all_pnl = trades[pnl_col].values
    point_estimate = stat_fn(all_pnl)

    # Block sizes summary
    block_lens = [len(b) for b in blocks]
    block_sizes = {
        "min": int(np.min(block_lens)),
        "max": int(np.max(block_lens)),
        "mean": round(float(np.mean(block_lens)), 1),
        "median": int(np.median(block_lens)),
    }

    # Resample
    rng = np.random.default_rng(seed)
    target_n = len(all_pnl)
    stats = []

    for _ in range(n_resamples):
        sampled_pnl = []
        total = 0
        while total < target_n:
            idx = rng.integers(0, n_blocks)
            sampled_pnl.append(blocks[idx])
            total += len(blocks[idx])
        # Trim to target_n
        concatenated = np.concatenate(sampled_pnl)[:target_n]
        stats.append(stat_fn(concatenated))

    stats = np.array(stats)
    alpha = (1 - ci) / 2

    return {
        "stat": float(point_estimate),
        "ci_lower": float(np.percentile(stats, alpha * 100)),
        "ci_upper": float(np.percentile(stats, (1 - alpha) * 100)),
        "se": float(np.std(stats)),
        "n_blocks": n_blocks,
        "n_resamples": n_resamples,
        "block_sizes": block_sizes,
    }


def compare_bootstrap_methods(trades: pd.DataFrame,
                              block_col: str = "series_ticker",
                              pnl_col: str = "pnl",
                              n_resamples: int = 1000,
                              ci: float = 0.95,
                              seed: int = 42) -> dict:
    """
    Compare block bootstrap vs standard (i.i.d.) bootstrap.

    Returns:
        {
            "block":    block_bootstrap result dict,
            "standard": standard bootstrap result dict,
            "ci_wider_pct": how much wider the block CI is (%)
        }
    """
    # Block bootstrap
    block_result = block_bootstrap(
        trades, block_col=block_col, pnl_col=pnl_col,
        n_resamples=n_resamples, ci=ci, seed=seed,
    )

    # Standard (i.i.d.) bootstrap — each trade is its own block
    std_result = block_bootstrap(
        trades, block_col=None, pnl_col=pnl_col,
        n_resamples=n_resamples, ci=ci, seed=seed,
    )

    block_width = block_result["ci_upper"] - block_result["ci_lower"]
    std_width = std_result["ci_upper"] - std_result["ci_lower"]
    wider_pct = ((block_width / max(std_width, 1e-10)) - 1) * 100

    return {
        "block": block_result,
        "standard": std_result,
        "ci_wider_pct": round(wider_pct, 1),
    }


if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.default_rng(42)
    n_series = 50
    trades_list = []
    for i in range(n_series):
        n_trades = rng.integers(5, 30)
        base_pnl = rng.normal(0.02, 0.05)
        pnls = rng.normal(base_pnl, 0.08, n_trades)
        for p in pnls:
            trades_list.append({"series_ticker": f"SER-{i:03d}", "pnl": p})

    df = pd.DataFrame(trades_list)
    print(f"Synthetic: {len(df)} trades, {n_series} series\n")

    comp = compare_bootstrap_methods(df)
    print("Block Bootstrap:")
    print(f"  Sharpe: {comp['block']['stat']:.2f}  "
          f"CI: [{comp['block']['ci_lower']:.2f}, {comp['block']['ci_upper']:.2f}]")
    print(f"  SE: {comp['block']['se']:.3f}  Blocks: {comp['block']['n_blocks']}")
    print(f"  Block sizes: {comp['block']['block_sizes']}")
    print()
    print("Standard Bootstrap:")
    print(f"  Sharpe: {comp['standard']['stat']:.2f}  "
          f"CI: [{comp['standard']['ci_lower']:.2f}, {comp['standard']['ci_upper']:.2f}]")
    print(f"  SE: {comp['standard']['se']:.3f}")
    print()
    print(f"Block CI is {comp['ci_wider_pct']:+.1f}% wider than standard CI")
