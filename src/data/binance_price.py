"""
src/data/binance_price.py
Binance public REST API — daily OHLCV for BTC/USDT and ETH/USDT.
No auth required for public market data.
https://api.binance.com/api/v3/klines
"""
import time
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

API     = "https://api.binance.com/api/v3/klines"
RAW_DIR = Path("data/raw")

SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def fetch_klines(symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end).timestamp() * 1000)
    all_rows = []
    current  = start_ms

    while current < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current,
            "endTime":   end_ms,
            "limit":     1000,
        }
        for attempt in range(4):
            try:
                r = requests.get(API, params=params, timeout=15)
                if r.status_code == 200:
                    batch = r.json()
                    all_rows.extend(batch)
                    break
                elif r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                else:
                    break
            except Exception:
                time.sleep(3)

        if not batch:
            break
        current = int(batch[-1][0]) + 1  # next candle start

    if not all_rows:
        return pd.DataFrame()

    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_vol","n_trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["date"]  = pd.to_datetime(df["open_time"], unit="ms").dt.date
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["symbol"] = symbol
    return df[["date","symbol","close","volume","high","low","open"]].drop_duplicates("date")


def fetch_all(days: int = 730) -> pd.DataFrame:
    end   = date.today()
    start = end - timedelta(days=days)
    frames = []
    for sym in SYMBOLS:
        print(f"  [Binance] {sym}...")
        df = fetch_klines(sym, start, end)
        frames.append(df)
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True).sort_values(["symbol","date"]).reset_index(drop=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol","date"])
    df["ret_1d"]   = df.groupby("symbol")["close"].pct_change(1)
    df["ret_7d"]   = df.groupby("symbol")["close"].pct_change(7)
    df["vol_20d"]  = df.groupby("symbol")["ret_1d"].transform(lambda x: x.rolling(20).std())
    df["price_zscore_30"] = (
        df.groupby("symbol")["close"].transform(
            lambda x: (x - x.rolling(30).mean()) / x.rolling(30).std().replace(0,1)
        )
    )
    return df


def fetch_and_save() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_all(730)
    if df.empty:
        return df
    df = compute_features(df)
    df.to_csv(RAW_DIR / "binance_prices.csv", index=False)
    for sym, grp in df.groupby("symbol"):
        print(f"  [Binance] {sym}: {len(grp)} days | {grp['date'].min()} → {grp['date'].max()}")
    return df


def validate(df: pd.DataFrame) -> None:
    assert "close" in df.columns
    assert (df["close"] > 0).all()


if __name__ == "__main__":
    df = fetch_and_save()
    validate(df)
