"""
src/whale/smart_money.py
Compute Smart Money Index features for use in the model.
Loads whale_positions_*.csv from data/whale/ and computes per-market features.
"""
import pandas as pd
from pathlib import Path
from datetime import date

WHALE_DIR = Path("data/whale")
PROC_DIR  = Path("data/processed")


def load_whale_history() -> pd.DataFrame:
    """Load all historical whale position scans."""
    files = sorted(WHALE_DIR.glob("whale_positions_*.csv"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def compute_market_whale_features(positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (market_slug, scan_date):
      - whale_count_yes / whale_count_no
      - whale_exposure_yes / whale_exposure_no
      - smart_money_index (0-100)
      - whale_cluster_detected (bool: 3+ same direction)
      - whale_consensus_direction (yes/no/split)
      - whale_avg_position_size
    """
    if positions_df.empty:
        return pd.DataFrame()

    rows = []
    for (slug, scan_date), grp in positions_df.groupby(["market_slug", "scan_date"]):
        yes = grp[grp["outcome"] == "yes"]
        no  = grp[grp["outcome"] == "no"]

        yes_exp = yes["current_value"].sum()
        no_exp  = no["current_value"].sum()
        total   = yes_exp + no_exp

        smi = (yes_exp / total * 100) if total > 0 else 50.0
        direction = (
            "yes"  if yes_exp > no_exp * 1.5
            else "no"  if no_exp  > yes_exp * 1.5
            else "split"
        )
        rows.append({
            "market_slug":            slug,
            "scan_date":              scan_date,
            "whale_count_yes":        len(yes),
            "whale_count_no":         len(no),
            "whale_count_total":      len(grp),
            "whale_exposure_yes":     round(yes_exp, 2),
            "whale_exposure_no":      round(no_exp, 2),
            "smart_money_index":      round(smi, 1),
            "whale_cluster_detected": int(max(len(yes), len(no)) >= 3),
            "whale_consensus_direction": direction,
            "whale_avg_position_size": round(grp["current_value"].mean(), 2),
        })

    return pd.DataFrame(rows)


def save_features() -> pd.DataFrame:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    pos = load_whale_history()
    if pos.empty:
        print("[SmartMoney] No whale history found. Run whale/tracker.py first.")
        return pd.DataFrame()
    df = compute_market_whale_features(pos)
    out = PROC_DIR / "whale_features.parquet"
    df.to_parquet(out, index=False)
    print(f"[SmartMoney] {len(df)} market-date rows → {out}")
    return df


if __name__ == "__main__":
    save_features()
