"""
src/data/newsapi_fetch.py
NewsAPI — headline volume and sentiment proxy.
Free tier: 100 req/day. Register at https://newsapi.org
Set NEWSAPI_KEY env var.
"""
import os
import time
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
BASE        = "https://newsapi.org/v2/everything"
RAW_DIR     = Path("data/raw")

QUERIES = {
    "crypto":   "bitcoin OR ethereum OR crypto",
    "weather":  "extreme weather OR heat wave OR hurricane OR temperature record",
    "macro":    "federal reserve OR inflation OR interest rate",
    "politics": "election OR congress OR president",
    "sports":   "super bowl OR NBA OR NFL OR championship",
}


def fetch_headlines(query: str, days_back: int = 7) -> dict:
    """Return headline count and top titles for a query (last N days)."""
    if not NEWSAPI_KEY:
        return {"count": 0, "titles": []}
    from_date = (date.today() - timedelta(days=days_back)).isoformat()
    params = {
        "q":          query,
        "from":       from_date,
        "sortBy":     "publishedAt",
        "pageSize":   100,
        "language":   "en",
        "apiKey":     NEWSAPI_KEY,
    }
    try:
        r = requests.get(BASE, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return {
                "count":  data.get("totalResults", 0),
                "titles": [a.get("title","") for a in data.get("articles", [])[:5]],
            }
        elif r.status_code == 401:
            print("[NewsAPI] 401 — set NEWSAPI_KEY env var")
    except Exception as e:
        print(f"[NewsAPI] {e}")
    return {"count": 0, "titles": []}


def fetch_and_save() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not NEWSAPI_KEY:
        print("[NewsAPI] NEWSAPI_KEY not set — creating placeholder")
        df = pd.DataFrame(columns=["date","category","headline_count"])
        df.to_csv(RAW_DIR / "newsapi.csv", index=False)
        return df

    rows = []
    today = date.today()
    for category, query in QUERIES.items():
        result = fetch_headlines(query, days_back=7)
        rows.append({
            "date":           today,
            "category":       category,
            "headline_count": result["count"],
        })
        time.sleep(1)

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "newsapi.csv", index=False)
    print(f"[NewsAPI] {len(df)} category rows saved")
    return df


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        return
    assert "headline_count" in df.columns


if __name__ == "__main__":
    fetch_and_save()
