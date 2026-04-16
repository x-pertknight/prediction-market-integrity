"""
src/data/fetch_all.py
Master data fetch script — runs all data sources in parallel.
Call this to refresh all external data before running the model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env BEFORE importing modules (they read os.environ at import time)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from data.fear_greed       import fetch_and_save as fetch_fg
from data.wikipedia_pageviews import fetch_and_save as fetch_wiki
from data.binance_price    import fetch_and_save as fetch_binance
from data.coinalyze        import fetch_and_save as fetch_coinalyze
from data.fred_macro       import fetch_and_save as fetch_fred
from data.google_trends    import fetch_and_save as fetch_trends
from data.newsapi_fetch    import fetch_and_save as fetch_news
from data.polymarket_gamma import fetch_and_save as fetch_poly_gamma
from data.polymarket_clob  import fetch_and_save as fetch_poly_clob


def run_all(skip_slow: bool = False) -> None:
    print("[DataFetchAll] Starting all data sources...")

    # Fast / no-auth sources
    print("\n── Fear & Greed Index ───")
    fetch_fg()

    print("\n── Binance Prices (BTC/ETH) ───")
    fetch_binance()

    print("\n── Wikipedia Pageviews ───")
    fetch_wiki()

    print("\n── Polymarket Gamma (active markets) ───")
    fetch_poly_gamma()

    print("\n── Polymarket CLOB ───")
    fetch_poly_clob()

    # Slower / optional
    if not skip_slow:
        print("\n── FRED Macro ───")
        fetch_fred()

        print("\n── Coinalyze (OI/Funding) ───")
        fetch_coinalyze()

        print("\n── Google Trends ───")
        fetch_trends()

        print("\n── NewsAPI ───")
        fetch_news()

    print("\n[DataFetchAll] All data sources complete.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-slow", action="store_true", help="Skip FRED, Coinalyze, Trends, News")
    args = p.parse_args()
    run_all(skip_slow=args.skip_slow)
