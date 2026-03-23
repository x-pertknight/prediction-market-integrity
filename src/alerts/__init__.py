"""alerts package — re-exports from telegram_bot for convenience."""
from .telegram_bot import (
    alert_entry_signal,
    alert_exit_signal,
    alert_whale_cluster,
    alert_daily_pnl,
    alert_error,
    _send as send_telegram,
)

# Convenience aliases matching the new naming conventions
def alert_mispricing(ticker: str, pred_prob: float, entry_price: float,
                     edge: float, category: str = "") -> None:
    alert_entry_signal(
        market       = ticker,
        category     = category or "general",
        model_prob   = pred_prob,
        market_price = entry_price,
        kelly_size   = 0.0,
        whale_smi    = None,
        confidence   = "HIGH" if edge > 0.3 else "MEDIUM",
    )


def alert_exit(ticker: str, pnl: float, reason: str = "") -> None:
    alert_exit_signal(
        market      = ticker,
        entry_price = 0.0,
        exit_price  = 0.0,
        pnl         = pnl,
        reason      = reason or "signal",
    )


def alert_regime_change(old_regime: str, new_regime: str, kelly_mult: float) -> None:
    from .telegram_bot import _send
    _send(
        f"⚠️ <b>REGIME CHANGE</b>\n"
        f"{old_regime} → <b>{new_regime}</b>\n"
        f"Kelly multiplier: {kelly_mult:.2f}x"
    )


def send_daily_summary(date_str: str, n_trades: int, total_pnl: float,
                       sharpe: float, win_rate: float,
                       n_signals: int = 0, regime: str = "") -> None:
    from .telegram_bot import alert_daily_pnl
    alert_daily_pnl(
        total_pnl  = total_pnl,
        n_trades   = n_trades,
        win_rate   = win_rate,
        bankroll   = 0.0,
    )


def send_startup(date_str: str = "") -> None:
    from datetime import date as _date
    from .telegram_bot import _send
    d = date_str or _date.today().isoformat()
    _send(f"🚀 <b>TrapTrade online</b> — {d}\nMonitoring Kalshi markets.")


def dispatch_whale_alerts(whale_latest_path: str = "data/whale/whale_latest.json") -> int:
    """Read latest whale scan and fire alerts for strong signals. Returns alert count."""
    import json
    from pathlib import Path
    path = Path(whale_latest_path)
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    sent = 0
    for sig in data.get("top_signals", []):
        if sig.get("signal_strength") == "STRONG":
            alert_whale_cluster(
                market_slug  = sig["market_slug"],
                direction    = sig["direction"],
                whale_count  = sig["whale_count"],
                exposure_usd = sig["total_whale_usd"],
                smi          = sig["smart_money_idx"],
            )
            sent += 1
    return sent


__all__ = [
    "alert_entry_signal",
    "alert_exit_signal",
    "alert_whale_cluster",
    "alert_daily_pnl",
    "alert_error",
    "alert_mispricing",
    "alert_exit",
    "alert_regime_change",
    "send_daily_summary",
    "send_startup",
    "dispatch_whale_alerts",
    "send_telegram",
]
