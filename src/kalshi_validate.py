"""
kalshi_validate.py — data_noaa_07
Validate Kalshi raw data schema and fill floor_strike from ticker when missing.
Output: validated DataFrames (used by noaa_kalshi_join.py).
"""

import re
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
CITIES  = ["nyc", "chicago", "miami", "seattle", "atlanta", "dallas"]


def parse_strike_from_ticker(ticker: str) -> float | None:
    """Extract temperature threshold from ticker suffix.
    KXHIGHNY-26MAR21-T60  → 60.0
    KXHIGHNY-26MAR21-B59.5 → 59.5
    """
    m = re.search(r"-[TB]([\d.]+)$", ticker)
    return float(m.group(1)) if m else None


def validate_kalshi(city_lower: str) -> pd.DataFrame:
    path = RAW_DIR / f"kalshi_{city_lower}.csv"
    df   = pd.read_csv(path)

    # Fill floor_strike from ticker where missing
    mask = df["floor_strike"].isna()
    df.loc[mask, "floor_strike"] = df.loc[mask, "ticker"].apply(parse_strike_from_ticker)

    # Assert result is in valid set
    invalid_result = df[~df["result"].isin(["yes", "no", ""])]
    assert len(invalid_result) == 0, f"[{city_lower}] invalid result values: {invalid_result['result'].unique()}"

    # Assert price in [0, 1]
    for col in ["yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"]:
        out_of_range = df[(df[col].notna()) & ((df[col] < 0) | (df[col] > 1))]
        assert len(out_of_range) == 0, f"[{city_lower}] {col} out of [0,1]: {len(out_of_range)} rows"

    # Keep only settled (result yes/no)
    settled = df[df["result"].isin(["yes", "no"])].copy()
    settled["close_date"] = pd.to_datetime(settled["close_time"]).dt.date
    settled["city"]       = city_lower.capitalize()

    print(f"  [{city_lower:<8}] total={len(df):>4} | settled={len(settled):>4} | "
          f"strikes: {int(settled['floor_strike'].min())}–{int(settled['floor_strike'].max())} °F | "
          f"dates: {settled['close_date'].min()} → {settled['close_date'].max()}")

    return settled


def validate_all() -> dict[str, pd.DataFrame]:
    print("[Kalshi] Validating all cities...")
    result = {}
    for city_lower in CITIES:
        result[city_lower] = validate_kalshi(city_lower)
    print(f"[Kalshi] Validation complete. All assertions passed.")
    return result


if __name__ == "__main__":
    validate_all()
