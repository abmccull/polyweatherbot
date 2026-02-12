"""Metrics snapshot for external observability."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime

from config import Config
from learning.tracker import TradeTracker
from trading.portfolio import Portfolio
from trading.position_manager import PositionManager
from utils.logging import get_logger

log = get_logger("observability")

METRICS_FILE = "metrics.json"


def write_metrics_snapshot(
    config: Config,
    portfolio: Portfolio,
    tracker: TradeTracker,
    position_manager: PositionManager,
    active_markets: int = 0,
) -> None:
    """Collect and atomically write a metrics snapshot to metrics.json."""
    try:
        portfolio_value = portfolio.get_value()
        deployed = portfolio.get_deployed()
        peak = portfolio.peak_value
        drawdown = 1.0 - (portfolio_value / peak) if peak > 0 else 0.0

        stats_all = tracker.get_stats()
        stats_7d = tracker.get_stats(lookback_days=7)
        stats_1d = tracker.get_stats(lookback_days=1)

        open_positions = position_manager.get_open_positions()

        snapshot = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "portfolio": {
                "value": round(portfolio_value, 2),
                "peak": round(peak, 2),
                "drawdown_pct": round(drawdown * 100, 2),
                "deployed": round(deployed, 2),
                "initial_bankroll": config.initial_bankroll,
            },
            "stats_all_time": {
                "total_trades": stats_all.total_trades,
                "resolved": stats_all.resolved_trades,
                "wins": stats_all.wins,
                "losses": stats_all.losses,
                "win_rate": round(stats_all.win_rate, 4),
                "total_pnl": round(stats_all.total_pnl, 2),
                "avg_roi": round(stats_all.avg_roi, 4),
            },
            "stats_7d": {
                "resolved": stats_7d.resolved_trades,
                "win_rate": round(stats_7d.win_rate, 4),
                "pnl": round(stats_7d.total_pnl, 2),
            },
            "stats_1d": {
                "resolved": stats_1d.resolved_trades,
                "win_rate": round(stats_1d.win_rate, 4),
                "pnl": round(stats_1d.total_pnl, 2),
            },
            "positions": {
                "open_count": len(open_positions),
                "tokens": [p.token_id[:12] + "..." for p in open_positions],
            },
            "markets": {
                "active": active_markets,
            },
            "params": {
                "min_confidence": config.min_confidence.value,
                "max_price": config.max_price.value,
                "bet_pct": config.bet_pct.value,
            },
            "circuit_breaker": {
                "active": portfolio._circuit_breaker_until is not None
                and datetime.utcnow() < portfolio._circuit_breaker_until,
                "consecutive_losses": portfolio._consecutive_losses,
            },
            "mode": "dry_run" if config.dry_run else "live",
        }

        # Atomic write: write to temp file, then rename
        dir_name = os.path.dirname(os.path.abspath(METRICS_FILE))
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(snapshot, f, indent=2)
            os.replace(tmp_path, METRICS_FILE)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        log.debug("metrics_snapshot_written")
    except Exception as e:
        log.warning("metrics_snapshot_failed", error=str(e))
