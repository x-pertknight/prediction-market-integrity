"""
src/dashboard/terminal.py
Rich terminal dashboard for TrapTrade.
Shows: open positions, P&L, whale alerts, model signals, fear & greed, BTC price.
Run: python src/dashboard/terminal.py
"""
import json
import sys
import time
from pathlib import Path
from datetime import date

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich import box
    from rich.live import Live
    from rich.layout import Layout
except ImportError:
    print("Install rich: pip install rich")
    sys.exit(1)

WHALE_DIR = Path("data/whale")
PROC_DIR  = Path("data/processed")
STATE_F   = Path("data/risk_state.json")
RAW_DIR   = Path("data/raw")


def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def load_risk_state() -> dict:
    return load_json(STATE_F)


def load_whale_alerts() -> list:
    f = WHALE_DIR / "whale_latest.json"
    d = load_json(f)
    return d.get("top_alerts", [])


def load_fear_greed() -> dict:
    import pandas as pd
    f = RAW_DIR / "fear_greed.csv"
    if f.exists():
        df = pd.read_csv(f)
        if not df.empty:
            row = df.iloc[-1]
            return {"value": int(row["fear_greed_value"]), "label": str(row["fear_greed_label"])}
    return {"value": 50, "label": "Neutral"}


def load_btc_price() -> dict:
    import pandas as pd
    f = RAW_DIR / "binance_prices.csv"
    if f.exists():
        df = pd.read_csv(f)
        btc = df[df["symbol"] == "BTCUSDT"].tail(2)
        if len(btc) >= 1:
            cur  = float(btc.iloc[-1]["close"])
            prev = float(btc.iloc[-2]["close"]) if len(btc) >= 2 else cur
            return {"price": cur, "change_pct": (cur - prev) / prev * 100}
    return {"price": 0, "change_pct": 0}


def load_trades_summary() -> dict:
    """Load rolling Sharpe, fragility, and P&L from clean trades parquet."""
    import numpy as np
    import pandas as pd
    path = PROC_DIR / "trades_clean.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
        pnl = df["pnl"]
        total_pnl  = float(pnl.sum())
        win_rate   = float(df["win"].mean()) if "win" in df.columns else 0.0
        n_trades   = len(df)

        # Rolling Sharpe (last 90 trades)
        pnl90  = pnl.tail(90)
        sharpe = float(pnl90.mean() / pnl90.std() * np.sqrt(252)) if pnl90.std() > 0 else 0.0

        # Fragility proxy: negative skew = fragile
        skew      = float(pnl.skew())
        fragility = -skew   # positive fragility = left tail exposure

        # Max drawdown
        cum = pnl.cumsum()
        mdd = float((cum - cum.cummax()).min())

        return {
            "n_trades":   n_trades,
            "total_pnl":  total_pnl,
            "win_rate":   win_rate,
            "sharpe90":   sharpe,
            "fragility":  fragility,
            "skew":       skew,
            "max_dd":     mdd,
        }
    except Exception as e:
        return {"error": str(e)}


def make_pnl_panel(summary: dict) -> Panel:
    if not summary or "n_trades" not in summary:
        return Panel("[dim]No trade data (run backtest_clean.py)[/dim]",
                     title="📊 P&L + Sharpe", border_style="dim")

    total_pnl  = summary["total_pnl"]
    win_rate   = summary["win_rate"]
    n_trades   = summary["n_trades"]
    sharpe     = summary["sharpe90"]
    fragility  = summary.get("fragility", 0)
    mdd        = summary.get("max_dd", 0)

    pnl_col    = "green" if total_pnl > 0 else "red"
    sharpe_col = "green" if sharpe > 1 else "yellow" if sharpe > 0 else "red"
    frag_col   = "green" if fragility < 0 else "yellow" if fragility < 1 else "red"

    text = (
        f"[bold]Trades:[/bold] {n_trades:,}  "
        f"[bold]Win:[/bold] {win_rate:.1%}\n"
        f"[bold]P&L:[/bold] [{pnl_col}]{total_pnl:+.4f}[/{pnl_col}]  "
        f"[bold]Max DD:[/bold] {mdd:+.4f}\n"
        f"[bold]Sharpe (90):[/bold] [{sharpe_col}]{sharpe:+.2f}[/{sharpe_col}]  "
        f"[bold]Fragility:[/bold] [{frag_col}]{fragility:+.2f}[/{frag_col}]"
    )
    return Panel(text, title="📊 P&L + Sharpe (90-trade rolling)", border_style="green")


def make_risk_panel(state: dict) -> Panel:
    bankroll   = state.get("bankroll",    1000.0)
    daily_pnl  = state.get("daily_pnl",  0.0)
    c_losses   = state.get("consecutive_losses", 0)
    can_trade  = state.get("can_trade",   True)
    open_pos   = state.get("open_positions", 0)
    dd_pct     = state.get("drawdown_pct", 0.0)

    color  = "green" if daily_pnl >= 0 else "red"
    status = "[green]ACTIVE[/green]" if can_trade else "[red]HALTED[/red]"

    text = (
        f"[bold]Bankroll:[/bold] ${bankroll:,.2f}  "
        f"[bold]Daily P&L:[/bold] [{color}]${daily_pnl:+.2f}[/{color}]\n"
        f"[bold]Drawdown:[/bold] {dd_pct:.1%}  "
        f"[bold]Consec losses:[/bold] {c_losses}  "
        f"[bold]Open:[/bold] {open_pos}\n"
        f"[bold]Status:[/bold] {status}"
    )
    return Panel(text, title="💰 Risk State", border_style="blue")


def make_whale_table(alerts: list) -> Table:
    t = Table(title="🐳 Whale Alerts", box=box.SIMPLE, show_header=True)
    t.add_column("Market", style="cyan", max_width=35)
    t.add_column("Dir", justify="center")
    t.add_column("Whales", justify="right")
    t.add_column("Exposure", justify="right")
    t.add_column("SMI", justify="right")
    t.add_column("Cluster", justify="center")

    for a in alerts[:8]:
        cluster = "⚡ YES" if a.get("cluster_detected") else "no"
        smi     = float(a.get("smart_money_idx", 50))
        smi_col = "green" if smi > 65 else "red" if smi < 35 else "yellow"
        t.add_row(
            a.get("market_slug","")[:35],
            a.get("direction","").upper(),
            str(a.get("whale_count",0)),
            f"${a.get('total_exposure',0):,.0f}",
            f"[{smi_col}]{smi:.0f}[/{smi_col}]",
            cluster,
        )
    if not alerts:
        t.add_row("No whale data", "–", "–", "–", "–", "–")
    return t


def make_market_panel(fg: dict, btc: dict) -> Panel:
    fg_val   = fg["value"]
    fg_label = fg["label"]
    fg_col   = "red" if fg_val < 25 else "orange1" if fg_val < 45 else "yellow" if fg_val < 55 else "green"
    btc_col  = "green" if btc["change_pct"] >= 0 else "red"
    text = (
        f"[bold]BTC:[/bold] ${btc['price']:,.0f}  "
        f"[{btc_col}]{btc['change_pct']:+.2f}%[/{btc_col}]\n"
        f"[bold]Fear & Greed:[/bold] [{fg_col}]{fg_val} – {fg_label}[/{fg_col}]"
    )
    return Panel(text, title="📊 Market Context", border_style="yellow")


def render_once(console: Console) -> None:
    state    = load_risk_state()
    alerts   = load_whale_alerts()
    fg       = load_fear_greed()
    btc      = load_btc_price()
    summary  = load_trades_summary()
    whale_raw = load_json(WHALE_DIR / "whale_latest.json")

    # Regime from latest whale scan
    regime     = whale_raw.get("regime", "")
    kelly_mult = whale_raw.get("kelly_multiplier", 1.0)

    console.clear()
    console.print(Panel(
        f"[bold cyan]TrapTrade[/bold cyan]  |  "
        f"Backtest Mode  |  {date.today().isoformat()}"
        + (f"  |  Regime: [bold]{regime}[/bold]  Kelly: {kelly_mult:.2f}x" if regime else ""),
        border_style="cyan"
    ))
    console.print(Columns([
        make_pnl_panel(summary),
        make_risk_panel(state),
        make_market_panel(fg, btc),
    ]))
    console.print(make_whale_table(alerts))
    console.print("\n[dim]Press Ctrl+C to exit[/dim]")


def run_live(refresh_seconds: int = 30) -> None:
    console = Console()
    console.print("[cyan]TrapTrade Dashboard starting...[/cyan]")
    try:
        while True:
            render_once(console)
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")


if __name__ == "__main__":
    run_live(30)
