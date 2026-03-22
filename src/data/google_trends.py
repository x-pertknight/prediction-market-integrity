"""
src/data/google_trends.py
Google Trends via pytrends — narrative spike detection.
No auth required. pip install pytrends.
"""
import time
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

KEYWORDS = {
    "crypto":    ["Bitcoin price", "Ethereum", "crypto crash"],
    "weather":   ["heat wave", "extreme cold", "temperature record"],
    "macro":     ["inflation", "interest rate", "recession"],
    "politics":  ["election", "president", "congress"],
}


def fetch_trends(kw_list: list[str], timeframe: str = "today 12-m") -> pd.DataFrame:
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10,25))
        pytrends.build_payload(kw_list[:5], cat=0, timeframe=timeframe, geo="US")
        df = pytrends.interest_over_time()
        if df.empty:
            return pd.DataFrame()
        df = df.drop(columns=["isPartial"], errors="ignore")
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    except Exception as e:
        print(f"  [GoogleTrends] {e}")
        return pd.DataFrame()


def fetch_and_save() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for category, keywords in KEYWORDS.items():
        print(f"  [GoogleTrends] {category}: {keywords[:2]}...")
        df = fetch_trends(keywords[:2])
        if not df.empty:
            df["category"] = category
            all_frames.append(df)
        time.sleep(2)  # avoid rate limiting

    if not all_frames:
        print("[GoogleTrends] No data retrieved (rate limit or network issue)")
        empty = pd.DataFrame(columns=["date","category","trend_score"])
        empty.to_csv(RAW_DIR / "google_trends.csv", index=False)
        return empty

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(RAW_DIR / "google_trends.csv", index=False)
    print(f"[GoogleTrends] {len(combined)} rows | {combined['date'].min()} → {combined['date'].max()}")
    return combined


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        return
    assert "date" in df.columns


if __name__ == "__main__":
    fetch_and_save()
