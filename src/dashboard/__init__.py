"""dashboard package — re-exports from terminal for convenience."""
from .terminal import (
    render_once,
    run_live,
    load_trades_summary,
    load_whale_alerts,
    load_risk_state,
    load_fear_greed,
    load_btc_price,
    make_pnl_panel,
    make_risk_panel,
    make_whale_table,
    make_market_panel,
)

# Backward-compatible alias
render_dashboard = render_once

__all__ = [
    "render_once",
    "render_dashboard",
    "run_live",
    "load_trades_summary",
    "load_whale_alerts",
]
