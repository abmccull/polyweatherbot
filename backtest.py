#!/usr/bin/env python3
"""Backtest framework: replay resolved BUY trades with Kelly sizing.

Usage:
    python backtest.py --db station_sniper.db --kelly-base 0.50 --kelly-max 0.80
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, Trade
from trading.sizing import kelly_fraction, kelly_multiplier_for_confidence, drawdown_throttle


def run_backtest(
    db_path: str,
    initial_bankroll: float = 200.0,
    kelly_base: float = 0.60,
    kelly_max: float = 0.75,
    kelly_min: float = 0.40,
    kelly_fee: float = 0.02,
    kelly_haircut: float = 0.03,
    kelly_min_edge: float = 0.05,
    kelly_confidence_floor: float = 0.80,
    drawdown_start: float = 0.15,
    drawdown_full: float = 0.30,
    min_bet: float = 5.0,
    max_bet: float = 150.0,
    sell_fee_rate: float = 0.01,
) -> dict:
    """Replay resolved BUY trades chronologically with Kelly sizing.

    Returns JSON-serializable summary dict.
    """
    db_url = f"sqlite:///{Path(db_path).resolve()}"
    engine = create_engine(db_url, echo=False)

    # Run migrations so the DB schema matches the ORM model
    from sqlalchemy import text
    Base.metadata.create_all(engine)
    from db.engine import (
        _migrate_add_precip_columns,
        _migrate_add_fee_column,
        _migrate_add_exit_columns,
    )
    _migrate_add_precip_columns(engine)
    _migrate_add_fee_column(engine)
    _migrate_add_exit_columns(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    # Load all resolved BUY trades ordered by resolved_at
    trades = session.query(Trade).filter(
        Trade.action == "BUY",
        Trade.resolved_correct.isnot(None),
    ).order_by(Trade.resolved_at).all()
    session.close()

    if not trades:
        return {"error": "No resolved BUY trades found", "trade_count": 0}

    # Simulation state
    bankroll = initial_bankroll
    peak = initial_bankroll
    max_drawdown = 0.0
    wins = 0
    losses = 0
    total_fees = 0.0
    total_pnl = 0.0
    bankroll_curve = []
    returns = []

    for trade in trades:
        confidence = trade.confidence
        price = trade.fill_price or trade.price

        # Skip if below confidence floor
        if confidence < kelly_confidence_floor:
            continue

        # Probability with haircut
        q = confidence - kelly_haircut

        # Edge check
        if (q - price) < kelly_min_edge:
            continue

        # Full Kelly
        f_full = kelly_fraction(q, price, kelly_fee)
        if f_full <= 0:
            continue

        # Fractional Kelly multiplier
        k_mult = kelly_multiplier_for_confidence(confidence, kelly_min, kelly_base, kelly_max)

        # Drawdown throttle
        dd_mult = drawdown_throttle(bankroll, peak, drawdown_start, drawdown_full)

        f_adjusted = f_full * k_mult * dd_mult
        size = f_adjusted * bankroll

        # Bounds
        size = max(min_bet, min(max_bet, size))

        # Don't bet more than we have
        if size > bankroll:
            size = bankroll
        if size <= 0:
            continue

        shares = size / price

        # Collect sell info for this trade's token
        # (simplified: use the trade's actual P&L and outcome)
        won = trade.resolved_correct

        if won:
            pnl = (1.0 - price) * shares
            wins += 1
        else:
            pnl = -price * shares
            losses += 1

        # Account for entry fee (maker = 0) -- no fee on GTC entry
        # Account for sell fees on winning trades (simplified)
        fee = 0.0
        if won:
            fee = (1.0 * shares) * sell_fee_rate  # fee on full payout
        total_fees += fee
        net_pnl = pnl - fee
        total_pnl += net_pnl

        bankroll += net_pnl
        if bankroll > peak:
            peak = bankroll

        current_dd = 1.0 - (bankroll / peak) if peak > 0 else 0.0
        if current_dd > max_drawdown:
            max_drawdown = current_dd

        bankroll_curve.append(round(bankroll, 2))
        if bankroll > 0:
            returns.append(net_pnl / size)  # return on this trade

    # Compute Sharpe estimate (annualized, assuming ~2 trades/day)
    sharpe = 0.0
    if returns:
        mean_ret = sum(returns) / len(returns)
        if len(returns) > 1:
            var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            std_ret = math.sqrt(var)
            if std_ret > 0:
                # Annualize: assume ~730 trades/year (2/day)
                sharpe = (mean_ret / std_ret) * math.sqrt(730)

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    return {
        "params": {
            "initial_bankroll": initial_bankroll,
            "kelly_base": kelly_base,
            "kelly_max": kelly_max,
            "kelly_min": kelly_min,
            "kelly_fee": kelly_fee,
            "min_bet": min_bet,
            "max_bet": max_bet,
        },
        "results": {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "total_fees": round(total_fees, 2),
            "final_bankroll": round(bankroll, 2),
            "peak_bankroll": round(peak, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_estimate": round(sharpe, 3),
        },
        "bankroll_curve": bankroll_curve[-50:] if len(bankroll_curve) > 50 else bankroll_curve,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest Station Sniper with Kelly sizing")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--bankroll", type=float, default=200.0, help="Initial bankroll")
    parser.add_argument("--kelly-base", type=float, default=0.60, help="Base Kelly fraction")
    parser.add_argument("--kelly-max", type=float, default=0.75, help="Max Kelly fraction")
    parser.add_argument("--kelly-min", type=float, default=0.40, help="Min Kelly fraction")
    parser.add_argument("--kelly-fee", type=float, default=0.02, help="Fee rate for Kelly odds")
    parser.add_argument("--min-bet", type=float, default=5.0, help="Minimum bet size")
    parser.add_argument("--max-bet", type=float, default=150.0, help="Maximum bet size")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(json.dumps({"error": f"Database not found: {args.db}"}))
        sys.exit(1)

    result = run_backtest(
        db_path=args.db,
        initial_bankroll=args.bankroll,
        kelly_base=args.kelly_base,
        kelly_max=args.kelly_max,
        kelly_min=args.kelly_min,
        kelly_fee=args.kelly_fee,
        min_bet=args.min_bet,
        max_bet=args.max_bet,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
