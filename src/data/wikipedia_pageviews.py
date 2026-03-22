"""
src/data/wikipedia_pageviews.py
Wikimedia Pageviews API — no auth required.
Tracks traffic anomalies for crypto/weather/topic pages.
https://wikimedia.org/api/rest_v1/
"""
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

API     = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents"
RAW_DIR = Path("data/raw")
HDR     = {"User-Agent": "traptrade/1.0 (research; marco@traptrade)"}

# Pages relevant to our prediction market categories
PAGES = {
    "Bitcoin":              "crypto",
    "Ethereum":             "crypto",
    "Temperature":          "weather",
    "Heat_wave":            "weather",
    "Federal_Reserve":      "macro",
    "Inflation":            "macro",
    "United_States_presidential_election": "politics",
    "Super_Bowl":           "sports",
    "Hurricane":            "weather",
}


def fetch_page(article: str, start: date, end: date) -> pd.DataFrame:
    start_s = start.strftime("%Y%m%d")
    end_s   = end.strftime("%Y%m%d")
    url     = f"{API}/{article}/daily/{start_s}/{end_s}"
    try:
        r = requests.get(url, headers=HDR, timeout=15)
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()
        items = r.json().get("items", [])
        rows = []
        for item in items:
            rows.append({
                "date":     pd.to_datetime(item["timestamp"], format="%Y%m%d00").date(),
                "article":  article,
                "views":    int(item["views"]),
                "category": PAGES.get(article, "other"),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  [WikiPageviews] {article}: {e}")
        return pd.DataFrame()


def fetch_all(days: int = 365) -> pd.DataFrame:
    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)
    frames = [fetch_page(article, start, end) for article in PAGES]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["article", "date"]).reset_index(drop=True)


def compute_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Add z-score anomaly vs 30-day rolling mean."""
    df = df.copy()
    df["views_roll30"] = (
        df.groupby("article")["views"]
        .transform(lambda x: x.rolling(30, min_periods=5).mean())
    )
    df["views_std30"] = (
        df.groupby("article")["views"]
        .transform(lambda x: x.rolling(30, min_periods=5).std())
    )
    df["views_zscore"] = (df["views"] - df["views_roll30"]) / df["views_std30"].replace(0, 1)
    return df


def fetch_and_save() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_all(365)
    if df.empty:
        print("[WikiPageviews] No data")
        return df
    df = compute_anomaly(df)
    df.to_csv(RAW_DIR / "wikipedia_pageviews.csv", index=False)
    print(f"[WikiPageviews] {len(df)} rows | {df['article'].nunique()} articles | {df['date'].min()} → {df['date'].max()}")
    return df


def validate(df: pd.DataFrame) -> None:
    assert "views" in df.columns
    assert (df["views"] >= 0).all()


if __name__ == "__main__":
    df = fetch_and_save()
    validate(df)
