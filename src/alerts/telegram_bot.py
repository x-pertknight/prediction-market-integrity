"""
src/alerts/telegram_bot.py — Telegram bot for TrapTrade.

Commands:
  /status  — P&L, positions, HMM regime, rolling Sharpe
  /trades  — last 10 trades
  /stop    — stop paper trader
  /start   — start paper trader

Auto-alerts (no command needed):
  ENTRY SIGNAL, EXIT SIGNAL, WHALE CLUSTER, CIRCUIT BREAKER, DAILY SUMMARY

Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env.
If placeholder, logs to console instead.
"""

import os
import json
import logging
import threading
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BASE    = f"https://api.telegram.org/bot{TOKEN}"

LOG_DIR    = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADES_CSV = Path("data/paper_trades.csv")

log = logging.getLogger("telegram_bot")

# ── Send / Log ───────────────────────────────────────────────────────────────

def _send(text: str, parse_mode: str = "HTML") -> bool:
    """Send message to Telegram. Falls back to console log if not configured."""
    if not TOKEN or not CHAT_ID or TOKEN.startswith("placeholder"):
        log.info(f"[TG-LOG] {text[:200]}")
        return False
    try:
        r = requests.post(
            f"{BASE}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        if r.status_code != 200:
            log.warning(f"Telegram send failed: {r.status_code} {r.text[:100]}")
        return r.status_code == 200
    except Exception as e:
        log.error(f"Telegram error: {e}")
        return False


def _log_event(event_type: str, data: dict) -> None:
    log_path = LOG_DIR / "telegram_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": event_type, **data
        }) + "\n")


# ── Alert functions (called by paper_trader, signal_generator, etc.) ─────────

def alert_entry_signal(market: str, category: str, model_prob: float,
                       market_price: float, kelly_size: float = 0,
                       convexity: float = 0, regime: str = "",
                       whale_smi: float | None = None) -> None:
    edge = (model_prob - market_price) / max(model_prob, 0.01) * 100
    regime_line = f"\n📊 <b>Regime:</b> {regime}" if regime else ""
    whale_line = f"\n🐳 <b>SMI:</b> {whale_smi:.0f}/100" if whale_smi is not None else ""
    text = (
        f"🟢 <b>ENTRY SIGNAL</b> [{category.upper()}]\n"
        f"📊 <b>Market:</b> {market[:60]}\n"
        f"🤖 <b>Model:</b> {model_prob:.1%}  💰 <b>Price:</b> {market_price:.1%}\n"
        f"📈 <b>Edge:</b> +{edge:.1f}%  📦 <b>Size:</b> ${kelly_size:.2f}\n"
        f"🎯 <b>Convexity:</b> {convexity:.3f}"
        f"{regime_line}{whale_line}"
    )
    _send(text)
    _log_event("entry_signal", {"market": market, "prob": model_prob, "price": market_price})


def alert_exit_signal(market: str, entry_price: float, exit_price: float,
                      pnl: float, reason: str = "expiry") -> None:
    emoji = "✅" if pnl >= 0 else "❌"
    text = (
        f"{emoji} <b>EXIT</b> — {reason.upper()}\n"
        f"📊 <b>Market:</b> {market[:60]}\n"
        f"💵 Entry: {entry_price:.1%} → Exit: {exit_price:.1%}\n"
        f"💰 P&L: <b>${pnl:+.4f}</b>"
    )
    _send(text)
    _log_event("exit_signal", {"market": market, "entry": entry_price,
                                "exit": exit_price, "pnl": pnl})


def alert_whale_cluster(market: str, n_whales: int, consensus: str,
                        smi: float) -> None:
    text = (
        f"🐳⚡ <b>WHALE CLUSTER</b>\n"
        f"📊 <b>Market:</b> {market[:60]}\n"
        f"👥 <b>Whales:</b> {n_whales}  📐 <b>Consensus:</b> {consensus}\n"
        f"🧠 <b>SMI:</b> {smi:.0f}/100"
    )
    _send(text)
    _log_event("whale_cluster", {"market": market, "whales": n_whales, "smi": smi})


def alert_circuit_breaker(reason: str) -> None:
    text = (
        f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b>\n"
        f"⛔ <b>Reason:</b> {reason}\n"
        f"🛑 Trading HALTED. Manual review required."
    )
    _send(text)
    _log_event("circuit_breaker", {"reason": reason})


def alert_daily_summary(total_pnl: float, n_trades: int, win_rate: float,
                        rolling_sharpe: float, open_positions: int = 0) -> None:
    emoji = "📈" if total_pnl >= 0 else "📉"
    text = (
        f"{emoji} <b>DAILY SUMMARY</b> — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
        f"💰 <b>P&L:</b> ${total_pnl:+.4f}\n"
        f"📦 <b>Trades:</b> {n_trades}  🎯 <b>Win:</b> {win_rate:.0%}\n"
        f"📊 <b>Rolling Sharpe (30):</b> {rolling_sharpe:.2f}\n"
        f"📂 <b>Open positions:</b> {open_positions}"
    )
    _send(text)
    _log_event("daily_summary", {"pnl": total_pnl, "trades": n_trades,
                                  "sharpe": rolling_sharpe})


# ── Command handlers ─────────────────────────────────────────────────────────

def _handle_status() -> str:
    """Build /status response."""
    lines = ["📊 <b>TrapTrade Status</b>\n"]

    # Load paper trades
    if TRADES_CSV.exists():
        df = pd.read_csv(TRADES_CSV)
        sells = df[df["side"] == "PAPER_SELL"]
        if not sells.empty:
            cum_pnl = sells["cumulative_pnl"].iloc[-1] if "cumulative_pnl" in sells.columns else 0
            n_trades = len(sells)
            wins = (sells["pnl_net"].astype(float) > 0).sum() if "pnl_net" in sells.columns else 0
            wr = wins / max(n_trades, 1)
            pnls = sells["pnl_net"].astype(float).values
            sharpe_30 = 0.0
            if len(pnls) >= 5:
                recent = pnls[-30:]
                if np.std(recent) > 0:
                    sharpe_30 = np.mean(recent) / np.std(recent) * np.sqrt(252)
            lines.append(f"💰 <b>Cum P&L:</b> ${cum_pnl:+.4f}")
            lines.append(f"📦 <b>Trades:</b> {n_trades}  🎯 <b>Win:</b> {wr:.0%}")
            lines.append(f"📊 <b>Sharpe (30):</b> {sharpe_30:.2f}")
        buys = df[df["side"] == "PAPER_BUY"]
        open_ids = set(buys["market_id"]) - set(sells["market_id"]) if not sells.empty else set(buys["market_id"])
        lines.append(f"📂 <b>Open positions:</b> {len(open_ids)}")
    else:
        lines.append("No trades yet.")

    return "\n".join(lines)


def _handle_trades() -> str:
    """Build /trades response — last 10."""
    if not TRADES_CSV.exists():
        return "No trades yet."
    df = pd.read_csv(TRADES_CSV).tail(10)
    lines = ["📋 <b>Last 10 Trades</b>\n"]
    for _, r in df.iterrows():
        side = r.get("side", "?")
        q = str(r.get("market_question", ""))[:40]
        price = r.get("entry_price", 0)
        pnl = r.get("pnl_net", "")
        pnl_str = f" P&L=${float(pnl):+.4f}" if pnl != "" and pd.notna(pnl) else ""
        emoji = "🟢" if side == "PAPER_BUY" else "🔴"
        lines.append(f"{emoji} {side} {q} @{price}{pnl_str}")
    return "\n".join(lines)


# ── Polling loop for commands ────────────────────────────────────────────────

_paper_trader_ref = None  # set by paper_trader when it starts


def set_paper_trader(trader) -> None:
    """Register paper trader instance for /stop and /start commands."""
    global _paper_trader_ref
    _paper_trader_ref = trader


def _poll_commands() -> None:
    """Long-poll Telegram for commands. Runs in a background thread."""
    if not TOKEN or not CHAT_ID or TOKEN.startswith("placeholder"):
        log.info("[TG] No token configured — command polling disabled")
        return

    offset = 0
    while True:
        try:
            r = requests.get(f"{BASE}/getUpdates",
                             params={"offset": offset, "timeout": 30}, timeout=35)
            if r.status_code != 200:
                time.sleep(5)
                continue
            updates = r.json().get("result", [])
            for u in updates:
                offset = u["update_id"] + 1
                msg = u.get("message", {})
                text = (msg.get("text") or "").strip()
                chat = str(msg.get("chat", {}).get("id", ""))
                if chat != CHAT_ID:
                    continue
                if text == "/status":
                    _send(_handle_status())
                elif text == "/trades":
                    _send(_handle_trades())
                elif text == "/stop":
                    if _paper_trader_ref:
                        _paper_trader_ref.stop()
                        _send("🛑 Paper trader stopped.")
                    else:
                        _send("⚠️ Paper trader not running.")
                elif text == "/start":
                    _send("ℹ️ Use: nohup python src/execution/paper_trader.py &")
        except Exception as e:
            log.error(f"Command poll error: {e}")
            time.sleep(10)


def start_command_listener() -> threading.Thread:
    """Start command polling in background thread."""
    t = threading.Thread(target=_poll_commands, daemon=True, name="tg-commands")
    t.start()
    return t


# ── Daily summary scheduler ─────────────────────────────────────────────────

def _daily_summary_loop() -> None:
    """Send daily summary at 23:00 UTC."""
    while True:
        now = datetime.now(timezone.utc)
        # Next 23:00 UTC
        target_h = 23
        if now.hour >= target_h:
            # Already past today's summary, wait for tomorrow
            sleep_s = (24 - now.hour + target_h) * 3600 - now.minute * 60
        else:
            sleep_s = (target_h - now.hour) * 3600 - now.minute * 60
        time.sleep(max(sleep_s, 60))

        # Build summary
        if TRADES_CSV.exists():
            df = pd.read_csv(TRADES_CSV)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_df = df[df["timestamp"].str.startswith(today)]
            sells = today_df[today_df["side"] == "PAPER_SELL"]
            if not sells.empty:
                pnl = sells["pnl_net"].astype(float).sum()
                n = len(sells)
                wr = (sells["pnl_net"].astype(float) > 0).mean()
                all_sells = df[df["side"] == "PAPER_SELL"]
                pnls = all_sells["pnl_net"].astype(float).values[-30:]
                sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
                alert_daily_summary(pnl, n, wr, sharpe)


def start_daily_summary() -> threading.Thread:
    """Start daily summary scheduler in background thread."""
    t = threading.Thread(target=_daily_summary_loop, daemon=True, name="tg-daily")
    t.start()
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"TOKEN: {'SET' if TOKEN and not TOKEN.startswith('placeholder') else 'NOT SET'}")
    print(f"CHAT_ID: {'SET' if CHAT_ID else 'NOT SET'}")
    # Test alerts
    alert_entry_signal("Test BTC market", "Crypto", 0.35, 0.12, 8.5, 0.23, "TRENDING")
    alert_circuit_breaker("Rolling Sharpe 0.3 < 0.5")
    print(_handle_status())
    print(_handle_trades())
