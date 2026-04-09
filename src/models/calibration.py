"""
src/models/calibration.py
Model Calibration Analysis (Academic Alpha — B5).

Ref: Tetlock "Superforecasting", Brier score methodology.

A well-calibrated model has:
  - When it predicts 70%, the event happens 70% of the time
  - Brier Score (BS) close to 0 (perfect = 0, random = 0.25)

Tools provided:
  1. Brier Score — overall and per-category
  2. Reliability diagram (predicted vs actual freq by decile)
  3. Calibration offset — bias per prediction bucket
  4. Calibration-adjusted probability — applies historical bias correction

If BS > 0.20 for a category: model is miscalibrated → reduce sizing by 50%.

Usage:
    from models.calibration import CalibrationAnalyzer
    ca = CalibrationAnalyzer()
    ca.fit(wf_df, prob_col='pred_prob_clean', outcome_col='outcome')
    adjusted = ca.adjust_probs(wf_df)
    ca.print_report()
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

CACHE_PATH = Path("data/processed/calibration_cache.json")
N_BINS = 10
BS_MISCALIBRATED_THRESHOLD = 0.20  # Brier Score > this → reduce sizing
SIZING_REDUCTION_MISCAL    = 0.50  # 50% size reduction for miscalibrated cats


class CalibrationAnalyzer:
    """
    Analyzes and corrects model calibration (probability accuracy).

    A model can have high Sharpe but poor calibration — this means it makes
    money not because it knows the true probability, but because it exploits
    systematic mispricings. Calibration analysis reveals where this is happening.
    """

    def __init__(self, n_bins: int = N_BINS):
        self.n_bins = n_bins
        self.global_brier_score: float | None = None
        self.category_brier: dict[str, float] = {}
        self.global_offsets: dict[float, float] = {}  # bin_mid → calibration_offset
        self.category_offsets: dict[str, dict[float, float]] = {}
        self.reliability_data: dict[str, list] = {}   # for plotting

    def brier_score(self, probs: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Brier Score = mean((predicted_prob - outcome)^2)
        BS = 0: perfect | BS = 0.25: random | BS > 0.25: worse than random
        """
        p = np.asarray(probs, dtype=float)
        o = np.asarray(outcomes, dtype=float)
        mask = ~(np.isnan(p) | np.isnan(o))
        if mask.sum() == 0:
            return 0.25
        return float(np.mean((p[mask] - o[mask]) ** 2))

    def reliability_bins(self, probs: np.ndarray,
                         outcomes: np.ndarray) -> list[dict]:
        """
        Compute reliability diagram data: bin predictions into deciles,
        compute mean predicted prob and actual win rate per bin.
        """
        p = np.asarray(probs, dtype=float)
        o = np.asarray(outcomes, dtype=float)
        mask = ~(np.isnan(p) | np.isnan(o))
        p, o = p[mask], o[mask]

        bins = np.linspace(0, 1, self.n_bins + 1)
        rows = []
        for i in range(self.n_bins):
            lo, hi = bins[i], bins[i + 1]
            in_bin = (p >= lo) & (p < hi)
            if i == self.n_bins - 1:
                in_bin = (p >= lo) & (p <= hi)
            n = in_bin.sum()
            if n == 0:
                rows.append({"bin_mid": (lo + hi) / 2, "mean_pred": (lo+hi)/2,
                              "actual_rate": np.nan, "n": 0,
                              "offset": 0.0, "calibrated": True})
                continue
            mean_pred   = float(p[in_bin].mean())
            actual_rate = float(o[in_bin].mean())
            offset      = mean_pred - actual_rate  # positive = overconfident
            rows.append({
                "bin_mid":     round((lo + hi) / 2, 2),
                "mean_pred":   round(mean_pred, 4),
                "actual_rate": round(actual_rate, 4),
                "n":           int(n),
                "offset":      round(offset, 4),
                "calibrated":  abs(offset) < 0.05,
            })
        return rows

    def _bin_mid(self, prob: float) -> float:
        """Map a probability to its bin midpoint."""
        bins = np.linspace(0, 1, self.n_bins + 1)
        idx  = min(int(prob * self.n_bins), self.n_bins - 1)
        return round((bins[idx] + bins[idx + 1]) / 2, 2)

    def fit(self, df: pd.DataFrame,
            prob_col: str = "pred_prob_clean",
            outcome_col: str = "outcome",
            category_col: str = "category") -> "CalibrationAnalyzer":
        """
        Fit calibration model on walk-forward predictions.
        """
        if prob_col not in df.columns or outcome_col not in df.columns:
            print(f"[Calibration] Missing columns: {prob_col} or {outcome_col}")
            return self

        probs    = df[prob_col].values
        outcomes = df[outcome_col].values

        # Global Brier Score
        self.global_brier_score = self.brier_score(probs, outcomes)

        # Global reliability bins → offsets
        global_bins = self.reliability_bins(probs, outcomes)
        self.reliability_data["global"] = global_bins
        self.global_offsets = {r["bin_mid"]: r["offset"] for r in global_bins
                               if not np.isnan(r.get("actual_rate", np.nan))}

        # Per-category
        if category_col in df.columns:
            for cat, cat_df in df.groupby(category_col):
                if len(cat_df) < 20:
                    continue
                p = cat_df[prob_col].values
                o = cat_df[outcome_col].values
                bs = self.brier_score(p, o)
                self.category_brier[str(cat)] = round(bs, 5)

                cat_bins = self.reliability_bins(p, o)
                self.reliability_data[str(cat)] = cat_bins
                self.category_offsets[str(cat)] = {
                    r["bin_mid"]: r["offset"] for r in cat_bins
                    if not np.isnan(r.get("actual_rate", np.nan))
                }

        self.save()
        return self

    def adjusted_prob(self, raw_prob: float,
                      category: str | None = None) -> float:
        """
        Apply calibration offset to a raw model probability.
        adjusted = raw_prob - offset (where offset = mean_pred - actual_rate)
        """
        bin_mid = self._bin_mid(raw_prob)

        # Use category-specific offset if available
        if category and category in self.category_offsets:
            offset = self.category_offsets[category].get(bin_mid, 0.0)
        else:
            offset = self.global_offsets.get(bin_mid, 0.0)

        return float(np.clip(raw_prob - offset, 0.01, 0.99))

    def calibration_offset(self, raw_prob: float,
                           category: str | None = None) -> float:
        """Return the calibration offset (positive = model overestimates)."""
        bin_mid = self._bin_mid(raw_prob)
        if category and category in self.category_offsets:
            return self.category_offsets[category].get(bin_mid, 0.0)
        return self.global_offsets.get(bin_mid, 0.0)

    def sizing_multiplier(self, category: str) -> float:
        """
        Return position sizing multiplier based on calibration quality.
        BS > 0.20 → reduce sizing by 50%.
        BS < 0.20 → full size.
        """
        bs = self.category_brier.get(category, self.global_brier_score or 0.0)
        if bs is None:
            return 1.0
        if bs > BS_MISCALIBRATED_THRESHOLD:
            return SIZING_REDUCTION_MISCAL  # 0.5× for miscalibrated category
        return 1.0

    def confidence_score(self, raw_prob: float,
                         category: str | None = None) -> float:
        """Return calibration-adjusted confidence score in [0.0, 1.0].

        Applies calibration offset to correct model overconfidence/underconfidence.
        Used by CalibrationSignalProvider in the alpha combination framework.
        """
        return float(np.clip(self.adjusted_prob(raw_prob, category), 0.0, 1.0))

    def annotate(self, df: pd.DataFrame,
                 prob_col: str = "pred_prob_clean",
                 category_col: str = "category") -> pd.DataFrame:
        """
        Add calibration-adjusted columns to DataFrame.
        Adds: cal_offset, cal_adjusted_prob, cal_sizing_mult, cal_brier
        """
        out = df.copy()
        out["cal_offset"]       = 0.0
        out["cal_adjusted_prob"]= out.get(prob_col, pd.Series(0.5, index=out.index))
        out["cal_sizing_mult"]  = 1.0
        out["cal_brier"]        = self.global_brier_score or 0.25

        if prob_col not in out.columns:
            return out

        probs = out[prob_col].fillna(0.5)
        cats  = out.get(category_col, pd.Series("", index=out.index))

        for idx in out.index:
            p   = probs.iloc[idx] if idx < len(probs) else 0.5
            cat = cats.iloc[idx]  if idx < len(cats)  else ""
            out.at[idx, "cal_offset"]        = self.calibration_offset(p, cat)
            out.at[idx, "cal_adjusted_prob"] = self.adjusted_prob(p, cat)
            out.at[idx, "cal_sizing_mult"]   = self.sizing_multiplier(cat)
            out.at[idx, "cal_brier"]         = self.category_brier.get(
                cat, self.global_brier_score or 0.25)

        return out

    def print_report(self) -> None:
        """Print calibration report to stdout."""
        print("\n[Calibration] === CALIBRATION REPORT ===")
        print(f"  Global Brier Score: {self.global_brier_score:.5f}")
        print(f"  Benchmark: random=0.25 | perfect=0.00")
        print(f"  Threshold for reduced sizing: >{BS_MISCALIBRATED_THRESHOLD}")
        print()

        # Global reliability table
        global_bins = self.reliability_data.get("global", [])
        if global_bins:
            print("  Global Reliability Diagram:")
            print(f"  {'Bin':>6} {'Predicted':>10} {'Actual':>10} {'Offset':>8} {'N':>6} {'Status':>12}")
            for row in global_bins:
                if np.isnan(row.get("actual_rate", np.nan)):
                    continue
                bar = "▓" * int(abs(row["offset"]) * 30)
                status = "✓ Calibrated" if row["calibrated"] else "✗ Off"
                print(f"  {row['bin_mid']:>6.2f} {row['mean_pred']:>10.4f} "
                      f"{row['actual_rate']:>10.4f} {row['offset']:>+8.4f} "
                      f"{row['n']:>6} {status:>12} {bar}")

        print()
        print("  Per-category Brier Score:")
        for cat in sorted(self.category_brier):
            bs = self.category_brier[cat]
            mult = self.sizing_multiplier(cat)
            flag = " ⚠️  MISCALIBRATED" if bs > BS_MISCALIBRATED_THRESHOLD else ""
            bar  = "█" * int(bs * 60)
            print(f"    {cat:<30} BS={bs:.4f}  sizing={mult:.1f}×  {bar}{flag}")

    def save(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "global_brier": self.global_brier_score,
            "category_brier": self.category_brier,
            "global_offsets": {str(k): v for k, v in self.global_offsets.items()},
            "category_offsets": {
                cat: {str(k): v for k, v in offsets.items()}
                for cat, offsets in self.category_offsets.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path = CACHE_PATH) -> "CalibrationAnalyzer":
        obj = cls()
        if not path.exists():
            return obj
        data = json.loads(path.read_text())
        obj.global_brier_score  = data.get("global_brier")
        obj.category_brier      = data.get("category_brier", {})
        obj.global_offsets      = {float(k): v for k, v in
                                   data.get("global_offsets", {}).items()}
        obj.category_offsets    = {
            cat: {float(k): v for k, v in offsets.items()}
            for cat, offsets in data.get("category_offsets", {}).items()
        }
        return obj


def run(wf_path: str = "data/processed/wf_clean_general.parquet") -> CalibrationAnalyzer:
    """Fit calibration model on WF predictions and print report."""
    path = Path(wf_path)
    if not path.exists():
        print(f"[Calibration] {wf_path} not found — run model_clean.py first")
        return CalibrationAnalyzer()

    df = pd.read_parquet(path)
    ca = CalibrationAnalyzer()
    ca.fit(df)
    ca.print_report()
    return ca


if __name__ == "__main__":
    run()
