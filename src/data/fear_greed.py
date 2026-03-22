"""
src/data/fear_greed.py
Alternative.me Crypto Fear & Greed Index — no auth required.
https://api.alternative.me/fng/
"""
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

API = "https://api.alternative.me/fng/"
RAW_DIR = Path("data/raw")


def fetch(days: int = 365) -> pd.DataFrame:
    r = requests.get(API, params={"limit": days, "format": "json"}, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])
    rows = []
    for d in data:
        rows.append({
            "date":            pd.to_datetime(int(d["timestamp"]), unit="s").date(),
            "fear_greed_value": int(d["value"]),
            "fear_greed_label": d["value_classification"],
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def fetch_and_save() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch(days=500)
    df.to_csv(RAW_DIR / "fear_greed.csv", index=False)
    print(f"[FearGreed] {len(df)} rows | {df['date'].min()} → {df['date'].max()}")
    return df


def validate(df: pd.DataFrame) -> None:
    assert "fear_greed_value" in df.columns
    assert df["fear_greed_value"].between(0, 100).all()


if __name__ == "__main__":
    df = fetch_and_save()
    validate(df)
    print(df.tail(5).to_string())
