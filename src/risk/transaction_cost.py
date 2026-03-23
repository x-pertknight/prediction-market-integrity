"""
src/risk/transaction_cost.py
Market Impact + Transaction Cost Model.

Ref: RenTech transaction cost model (Zuckerman "The Man Who Solved the Market").
     Almgren-Chriss market impact model (adapted for binary prediction markets).

Total cost breakdown:
  1. Spread cost:  cost of crossing the bid-ask spread
  2. Impact cost:  market impact from position size (price moves against you)
  3. Timing cost:  cost of execution delay / slippage over time

impact = position_size / daily_volume * volatility * sqrt(execution_time)
total_cost = spread_cost + impact_cost + timing_cost

Decision rule:
  If total_cost > expected_edge: DO NOT TRADE
  Self-correcting: find rebalancing trades that minimize total portfolio cost.

Usage:
    from risk.transaction_cost import TransactionCostModel
    tc = TransactionCostModel()
    cost = tc.total_cost(position_size=100, daily_volume=5000, price=0.05)
    if tc.should_trade(expected_edge=0.03, **params): execute()
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Kalshi fee: $0.07 per contract per side (round-trip = $0.14 on $1 nominal)
# BUT markets trade in cents: one contract = $1 nominal. Entry at 5c = $0.05 per contract.
KALSHI_FEE_PER_CONTRACT = 0.07   # dollars (one-way); round-trip = 0.14
SPREAD_COST_MULTIPLIER  = 0.5    # half-spread estimate
DEFAULT_VOLATILITY      = 0.15   # annualized vol proxy for binary markets
EXECUTION_TIME_DAYS     = 0.04   # ~1 hour execution (1/24 of a day)
MIN_DAILY_VOLUME        = 10     # below this: consider illiquid


class TransactionCostModel:
    """
    Predicts total transaction cost for a given trade and decides whether to proceed.
    
    For binary prediction markets:
      - Contracts are priced 0-100c
      - One contract pays $1 if YES, $0 if NO
      - Kalshi charges $0.07/contract/side
      - Market impact is moderate (thin order books)
    """

    def __init__(self,
                 fee_per_contract: float = KALSHI_FEE_PER_CONTRACT,
                 spread_mult: float = SPREAD_COST_MULTIPLIER,
                 default_vol: float = DEFAULT_VOLATILITY,
                 exec_time_days: float = EXECUTION_TIME_DAYS):
        self.fee_per_contract = fee_per_contract
        self.spread_mult      = spread_mult
        self.default_vol      = default_vol
        self.exec_time_days   = exec_time_days

    def spread_cost(self, bid: float, ask: float) -> float:
        """
        Half-spread cost as fraction of entry price.
        If bid/ask not available, use spread_mult * default estimate.
        """
        if bid > 0 and ask > 0 and ask > bid:
            spread = ask - bid
            mid = (ask + bid) / 2
            return spread / (2 * mid) if mid > 0 else 0.0
        return self.spread_mult * 0.02  # default 1c half-spread on 50c market

    def impact_cost(self, position_size: float, daily_volume: float,
                    price: float, volatility=None,
                    exec_time=None) -> float:
        """
        Market impact cost (Almgren-Chriss linear model adapted for binary markets).
        
        impact = (position_size / daily_volume) * volatility * sqrt(exec_time)
        
        Args:
            position_size: number of contracts to trade
            daily_volume:  average daily volume in contracts
            price:         entry price (0-1 scale)
            volatility:    daily price volatility (annualized)
            exec_time:     execution time in trading days
        
        Returns: impact cost as fraction of trade value
        """
        if daily_volume <= MIN_DAILY_VOLUME:
            return 0.10  # 10% impact for illiquid markets

        vol = volatility if volatility is not None else self.default_vol
        t   = exec_time  if exec_time  is not None else self.exec_time_days

        # Participation rate
        participation = min(position_size / max(daily_volume, 1), 1.0)

        # Linear market impact (Almgren-Chriss)
        impact = participation * vol * np.sqrt(t)
        return float(np.clip(impact, 0, 0.50))

    def timing_cost(self, price: float, volatility=None,
                    exec_time=None) -> float:
        """
        Cost from price movement during execution (timing risk).
        timing = 0.5 * volatility^2 * exec_time (expected slippage)
        """
        vol = volatility if volatility is not None else self.default_vol
        t   = exec_time  if exec_time  is not None else self.exec_time_days
        return 0.5 * (vol ** 2) * t

    def kalshi_fee_fraction(self, price: float, n_contracts: int = 1) -> float:
        """
        Kalshi fee as fraction of expected P&L.
        Fee = $0.07/contract/side; P&L = (1 - price) per contract if YES.
        """
        expected_pnl_per_contract = 1 - price  # maximum gain
        if expected_pnl_per_contract <= 0:
            return 1.0  # fee exceeds max gain
        fee_fraction = self.fee_per_contract / expected_pnl_per_contract
        return float(fee_fraction)

    def total_cost(self, position_size: float, daily_volume: float,
                   price: float, bid: float = 0.0, ask: float = 0.0,
                   volatility=None) -> dict:
        """
        Compute total round-trip transaction cost.
        
        Returns dict with:
          spread_cost, impact_cost, timing_cost, fee_cost, total_cost
          (all as fractions of expected P&L)
        """
        s_cost = self.spread_cost(bid, ask)
        i_cost = self.impact_cost(position_size, daily_volume, price, volatility)
        t_cost = self.timing_cost(price, volatility)
        f_cost = self.kalshi_fee_fraction(price)

        total = s_cost + i_cost + t_cost + f_cost

        return {
            "spread_cost": round(s_cost, 5),
            "impact_cost": round(i_cost, 5),
            "timing_cost": round(t_cost, 5),
            "fee_cost":    round(f_cost, 5),
            "total_cost":  round(total, 5),
        }

    def should_trade(self, expected_edge: float, position_size: float,
                     daily_volume: float, price: float,
                     bid: float = 0.0, ask: float = 0.0,
                     volatility=None) -> bool:
        """
        True if expected_edge > total_cost.
        expected_edge = (pred_prob - entry_price) / entry_price
        """
        cost = self.total_cost(position_size, daily_volume, price, bid, ask, volatility)
        return expected_edge > cost["total_cost"]

    def min_edge_required(self, position_size: float, daily_volume: float,
                          price: float, bid: float = 0.0, ask: float = 0.0,
                          volatility=None) -> float:
        """Return the minimum edge required to make the trade profitable after costs."""
        cost = self.total_cost(position_size, daily_volume, price, bid, ask, volatility)
        return cost["total_cost"]

    def optimal_position_size(self, expected_edge: float, daily_volume: float,
                               price: float, volatility=None,
                               max_fraction: float = 0.10) -> float:
        """
        Find optimal position size that maximizes edge after impact cost.
        Uses linear search (not closed-form -- binary market model is simpler).
        
        Returns: optimal fraction of daily_volume to trade
        """
        best_net_edge = 0.0
        best_size     = 0.0

        for frac in np.linspace(0.001, max_fraction, 50):
            n_contracts = frac * daily_volume
            cost = self.total_cost(n_contracts, daily_volume, price, volatility=volatility)
            net_edge = expected_edge - cost["total_cost"]
            if net_edge > best_net_edge:
                best_net_edge = net_edge
                best_size     = frac

        return float(best_size)

    def annotate_trades(self, trades: pd.DataFrame,
                        prob_col: str = "pred_prob_clean",
                        entry_col: str = "entry_price",
                        volume_col: str = "volume_fp") -> pd.DataFrame:
        """
        Add transaction cost analysis columns to trades DataFrame.
        
        Adds:
          tc_total, tc_spread, tc_impact, tc_fee
          tc_net_edge (expected_edge - tc_total)
          tc_viable (True if net_edge > 0)
        """
        out = trades.copy()

        prices  = out[entry_col].fillna(0.05).clip(0.01, 0.99)
        probs   = out.get(prob_col, pd.Series(0.5, index=out.index)).fillna(0.5)
        volumes = out.get(volume_col, pd.Series(1000, index=out.index)).fillna(1000)

        # Expected edge = (pred_prob - entry_price) / entry_price
        out["expected_edge"] = ((probs - prices) / prices.clip(lower=0.01)).clip(-1, 10)

        tc_rows = []
        for price, vol in zip(prices, volumes):
            tc = self.total_cost(
                position_size=min(vol * 0.01, 100),  # 1% of daily volume
                daily_volume=vol,
                price=price,
            )
            tc_rows.append(tc)

        tc_df = pd.DataFrame(tc_rows, index=out.index)
        out["tc_total"]    = tc_df["total_cost"]
        out["tc_spread"]   = tc_df["spread_cost"]
        out["tc_impact"]   = tc_df["impact_cost"]
        out["tc_fee"]      = tc_df["fee_cost"]
        out["tc_net_edge"] = out["expected_edge"] - out["tc_total"]
        out["tc_viable"]   = out["tc_net_edge"] > 0

        return out


def run():
    tc = TransactionCostModel()
    print("[TCM] Transaction Cost Model Examples")
    print("=" * 60)
    
    test_cases = [
        {"price": 0.05, "volume": 500,   "edge": 0.02, "label": "5c illiquid"},
        {"price": 0.05, "volume": 5000,  "edge": 0.02, "label": "5c liquid"},
        {"price": 0.20, "volume": 10000, "edge": 0.08, "label": "20c liquid"},
        {"price": 0.50, "volume": 50000, "edge": 0.15, "label": "50c very liquid"},
    ]
    
    for case in test_cases:
        cost = tc.total_cost(
            position_size=100,
            daily_volume=case["volume"],
            price=case["price"],
        )
        viable = case["edge"] > cost["total_cost"]
        print(f"\n{case['label']}")
        print(f"    Expected edge:  {case['edge']:.3f}")
        print(f"    Total cost:     {cost['total_cost']:.3f} "
              f"(spread={cost['spread_cost']:.3f} impact={cost['impact_cost']:.3f} "
              f"fee={cost['fee_cost']:.3f})")
        print(f"    Decision:       {'TRADE' if viable else 'SKIP'}")


if __name__ == "__main__":
    run()
