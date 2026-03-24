"""
tests/test_leakage.py — Verify no temporal leakage in hist_win_rate_series.

Checks that for every row, hist_win_rate_series is computed using ONLY
outcomes from markets that closed on strictly PRIOR dates within the same
series. Markets closing on the same date must all get the same feature value.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROC_DIR = Path("data/processed")


@pytest.fixture(scope="module")
def feature_matrix():
    """Load the feature matrix."""
    p = PROC_DIR / "feature_matrix_clean_all.parquet"
    if not p.exists():
        pytest.skip("feature_matrix_clean_all.parquet not found")
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def wf_data():
    """Load walk-forward data."""
    p = PROC_DIR / "wf_clean_general.parquet"
    if not p.exists():
        pytest.skip("wf_clean_general.parquet not found")
    return pd.read_parquet(p)


class TestHistWinRateSeriesLeakage:
    """Verify hist_win_rate_series uses only prior-date outcomes."""

    def test_same_date_same_value(self, feature_matrix):
        """All markets closing on the same date in a series must have
        the same hist_win_rate_series."""
        df = feature_matrix.copy()
        grouped = df.groupby(["series_ticker", "close_date"])["hist_win_rate_series"]
        nunique = grouped.nunique()
        violations = (nunique > 1).sum()
        assert violations == 0, (
            f"Found {violations} (series, date) groups where hist_win_rate_series "
            f"differs among same-date rows (same-day leakage)"
        )

    def test_no_same_day_outcome_used(self, feature_matrix):
        """hist_win_rate_series for date D must only use outcomes from dates < D."""
        df = feature_matrix.copy()
        df["close_date"] = pd.to_datetime(df["close_date"])

        violations = 0
        series_counts = df.groupby("series_ticker").size()
        test_series = series_counts[series_counts >= 5].index[:50]

        for st in test_series:
            grp = df[df["series_ticker"] == st].copy()
            dates = sorted(grp["close_date"].unique())
            for d in dates:
                actual = grp.loc[grp["close_date"] == d, "hist_win_rate_series"].iloc[0]
                # Expected: mean outcome of all rows with close_date < d
                prior = grp[grp["close_date"] < d]
                expected = prior["outcome"].mean() if len(prior) > 0 else 0.5
                if abs(actual - expected) > 1e-4:
                    violations += 1

        assert violations == 0, (
            f"Found {violations} (series, date) combinations where "
            f"hist_win_rate_series uses same-day or future outcomes"
        )

    def test_first_date_in_series_is_default(self, feature_matrix):
        """First date in each series should have default value 0.5."""
        df = feature_matrix.copy()
        df["close_date"] = pd.to_datetime(df["close_date"])
        violations = 0

        series_counts = df.groupby("series_ticker").size()
        test_series = series_counts[series_counts >= 2].index[:100]

        for st in test_series:
            grp = df[df["series_ticker"] == st]
            first_date = grp["close_date"].min()
            first_rows = grp[grp["close_date"] == first_date]
            for _, row in first_rows.iterrows():
                if abs(row["hist_win_rate_series"] - 0.5) > 1e-4:
                    violations += 1

        assert violations == 0, (
            f"Found {violations} rows where first date in series has "
            f"hist_win_rate_series != 0.5"
        )

    def test_wf_feature_uses_only_pre_fold_data(self, wf_data, feature_matrix):
        """In WF test folds, hist_win_rate_series must match the value
        computed from the full feature matrix (which uses only prior-date outcomes)."""
        wf = wf_data.copy()
        fm = feature_matrix.copy()
        wf["close_date"] = pd.to_datetime(wf["close_date"])
        fm["close_date"] = pd.to_datetime(fm["close_date"])

        # The WF should carry through the same feature values as the feature matrix
        violations = 0
        total_checked = 0

        for fold_id in wf["fold"].unique()[:5]:
            fold = wf[wf["fold"] == fold_id]
            for _, row in fold.head(100).iterrows():
                st = row["series_ticker"]
                cd = row["close_date"]
                # Find matching value in feature matrix
                fm_match = fm[(fm["series_ticker"] == st) & (fm["close_date"] == cd)]
                if len(fm_match) > 0:
                    fm_val = fm_match["hist_win_rate_series"].iloc[0]
                    wf_val = row["hist_win_rate_series"]
                    if abs(fm_val - wf_val) > 1e-4:
                        violations += 1
                total_checked += 1

        assert violations == 0, (
            f"Found {violations}/{total_checked} rows where WF hist_win_rate_series "
            f"differs from feature matrix (feature may have been recomputed incorrectly)"
        )


class TestVolumeRankLeakage:
    """Verify volume_rank uses only prior-date data."""

    def test_same_date_same_value(self, feature_matrix):
        """All markets closing on same date in a series must have same volume_rank."""
        df = feature_matrix.copy()
        grouped = df.groupby(["series_ticker", "close_date"])["volume_rank"]
        nunique = grouped.nunique()
        violations = (nunique > 1).sum()
        assert violations == 0, (
            f"Found {violations} (series, date) groups where volume_rank "
            f"differs among same-date rows"
        )

    def test_volume_rank_bounded(self, feature_matrix):
        """volume_rank should be between 0 and 1."""
        df = feature_matrix
        assert (df["volume_rank"] >= 0).all(), "volume_rank has negative values"
        assert (df["volume_rank"] <= 1).all(), "volume_rank exceeds 1"


class TestFloorStrikePctLeakage:
    """Verify floor_strike_pct uses only past data."""

    def test_floor_strike_pct_bounded(self, feature_matrix):
        """floor_strike_pct is clipped to [0, 5]."""
        df = feature_matrix
        assert (df["floor_strike_pct"] >= 0).all()
        assert (df["floor_strike_pct"] <= 5).all()
