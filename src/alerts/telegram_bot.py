"""
src/alerts/telegram_bot.py
Telegram alert system for TrapTrade signals.
Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID env vars.
python-telegram-bot 22.7 (sync send via requests for simplicity).
"""
import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
BASE    = f"https://api.telegram.org/bot{TOKEN}"

LOG_DIR = Path("logs")


def _send(text: str, parse_mode: str = "HTML") -> bool:
    if not TOKEN or not CHAT_ID:
        print(f"[Telegram] (PLACEHOLDER) {text[:120]}")
        return False
    try:
        r = requests.post(
            f"{BASE}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return False


def _log(event_type: str, data: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "telegram_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "type": event_type, **data}) + "\n")


def alert_entry_signal(
    market:      str,
    category:    str,
    model_prob:  float,
    market_price: float,
    kelly_size:  float,
    whale_smi:   float | None = None,
    confidence:  str = "MEDIUM",
) -> None:
    edge = (model_prob - market_price) / model_prob * 100
    whale_line = f"\n🐳 <b>SMI:</b> {whale_smi:.0f}/100" if whale_smi is not None else ""
    text = (
        f"🟢 <b>ENTRY SIGNAL</b> [{category.upper()}]\n"
        f"📊 <b>Market:</b> {market[:60]}\n"
        f"🤖 <b>Model prob:</b> {model_prob:.1%}   "
        f"💰 <b>Market price:</b> {market_price:.1%}\n"
        f"📈 <b>Edge:</b> +{edge:.1f}%   "
        f"📦 <b>Kelly size:</b> ${kelly_size:.2f}\n"
        f"🎯 <b>Confidence:</b> {confidence}"
        f"{whale_line}"
    )
    _send(text)
    _log("entry_signal", {"market": market, "model_prob": model_prob, "price": market_price})


def alert_exit_signal(
    market:      str,
    entry_price: float,
    exit_price:  float,
    pnl:         float,
    reason:      str = "expiry",
) -> None:
    emoji = "✅" if pnl >= 0 else "❌"
    text = (
        f"{emoji} <b>EXIT</b> — {reason.upper()}\n"
        f"📊 <b>Market:</b> {market[:60]}\n"
        f"💵 Entry: {entry_price:.1%}  →  Exit: {exit_price:.1%}\n"
        f"📊 P&L: <b>${pnl:+.2f}</b>"
    )
    _send(text)
    _log("exit_signal", {"market": market, "entry": entry_price, "exit": exit_price, "pnl": pnl})


def alert_whale_cluster(
    market_slug:  str,
    direction:    str,
    whale_count:  int,
    exposure_usd: float,
    smi:          float,
) -> None:
    emoji = "🐳⚡"
    text = (
        f"{emoji} <b>WHALE CLUSTER DETECTED</b>\n"
        f"📊 <b>Market:</b> {market_slug[:60]}\n"
        f"📐 <b>Direction:</b> {direction.upper()}  "
        f"👥 <b>Whales:</b> {whale_count}\n"
        f"💰 <b>Total exposure:</b> ${exposure_usd:,.0f}\n"
        f"🧠 <b>Smart Money Index:</b> {smi:.0f}/100"
    )
    _send(text)
    _log("whale_cluster", {"market": market_slug, "direction": direction, "whales": whale_count})


def alert_daily_pnl(
    total_pnl:    float,
    n_trades:     int,
    win_rate:     float,
    bankroll:     float,
    fear_greed:   int | None = None,
) -> None:
    emoji = "📈" if total_pnl >= 0 else "📉"
    fg_line = f"\n😱 <b>Fear & Greed:</b> {fear_greed}/100" if fear_greed is not None else ""
    text = (
        f"{emoji} <b>DAILY P&L SUMMARY</b>\n"
        f"💰 <b>P&L:</b> ${total_pnl:+.2f}   "
        f"🎯 <b>Win rate:</b> {win_rate:.0%}\n"
        f"📦 <b>Trades:</b> {n_trades}   "
        f"🏦 <b>Bankroll:</b> ${bankroll:,.2f}"
        f"{fg_line}"
    )
    _send(text)
    _log("daily_pnl", {"pnl": total_pnl, "n_trades": n_trades, "win_rate": win_rate})


def alert_error(source: str, message: str) -> None:
    text = f"⚠️ <b>ERROR</b> [{source}]\n{message[:200]}"
    _send(text)
    _log("error", {"source": source, "message": message})


if __name__ == "__main__":
    # Test with placeholder
    alert_entry_signal(
        market       = "Will NYC high temp exceed 75°F on Apr 1, 2026?",
        category     = "weather",
        model_prob   = 0.35,
        market_price = 0.12,
        kelly_size   = 8.50,
        whale_smi    = 72.0,
        confidence   = "HIGH",
    )
