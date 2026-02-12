"""Auto-adjust parameters within safe bounds based on trade performance."""

from __future__ import annotations

from datetime import datetime

from config import Config, TunableParam
from db.engine import get_session
from db.models import ParameterAdjustment
from learning.tracker import TradeTracker, TradeStats
from utils.logging import get_logger

log = get_logger("optimizer")

# Minimum sample size before making adjustments
MIN_SAMPLE_SIZE = 10

# Param name → config attribute mapping
_TUNABLE_NAMES = ("min_confidence", "max_price", "bet_pct")


def restore_tunable_params(config: Config) -> list[str]:
    """Restore tunable parameters from the most recent DB adjustments.

    For each tunable param, reads the latest ParameterAdjustment row and
    sets the config value (clamped to bounds).  Returns descriptions of
    what was restored for logging.
    """
    session = get_session()
    restored: list[str] = []
    try:
        for name in _TUNABLE_NAMES:
            latest = (
                session.query(ParameterAdjustment)
                .filter(ParameterAdjustment.parameter_name == name)
                .order_by(ParameterAdjustment.created_at.desc())
                .first()
            )
            if latest is None:
                continue

            tunable: TunableParam = getattr(config, name)
            clamped = max(tunable.min_val, min(tunable.max_val, latest.new_value))
            old = tunable.value
            tunable.value = round(clamped, 4)
            restored.append(f"{name}: default {old:.4f} → restored {tunable.value:.4f}")
    except Exception as e:
        log.error("restore_params_failed", error=str(e))
    finally:
        session.close()
    return restored


class ParameterOptimizer:
    """Self-adjusting parameter optimization within safe bounds.

    | Parameter       | Default | Min  | Max  | Adjustment Rule                              |
    |-----------------|---------|------|------|----------------------------------------------|
    | min_confidence  | 0.85    | 0.70 | 0.95 | Win >85% → ↓0.02, Win <70% → ↑0.03         |
    | max_price       | 0.25    | 0.10 | 0.40 | ROI >3x → ↑0.02, ROI <1.5x → ↓0.03         |
    | bet_pct         | 0.05    | 0.02 | 0.10 | Win >80% over 30 → +0.005, Win <65% → -0.01 |
    """

    def __init__(self, config: Config, tracker: TradeTracker) -> None:
        self._config = config
        self._tracker = tracker

    def optimize(self) -> list[str]:
        """Run optimization pass. Returns list of adjustments made."""
        stats = self._tracker.get_stats()
        adjustments: list[str] = []

        if stats.resolved_trades < MIN_SAMPLE_SIZE:
            log.info(
                "optimizer_skip",
                reason="insufficient_data",
                resolved=stats.resolved_trades,
                required=MIN_SAMPLE_SIZE,
            )
            return adjustments

        # Adjust min_confidence
        adj = self._adjust_min_confidence(stats)
        if adj:
            adjustments.append(adj)

        # Adjust max_price
        adj = self._adjust_max_price(stats)
        if adj:
            adjustments.append(adj)

        # Adjust bet_pct (needs 30+ trades)
        if stats.resolved_trades >= 30:
            adj = self._adjust_bet_pct(stats)
            if adj:
                adjustments.append(adj)

        return adjustments

    def _adjust_min_confidence(self, stats: TradeStats) -> str | None:
        """Adjust min_confidence based on win rate."""
        param = self._config.min_confidence

        if stats.win_rate > 0.85:
            # Winning a lot → can be less selective to capture more
            return self._apply_adjustment(param, "min_confidence", -0.02, stats,
                                          f"Win rate {stats.win_rate:.1%} > 85% → lowering threshold")
        elif stats.win_rate < 0.70:
            # Losing too much → be more selective
            return self._apply_adjustment(param, "min_confidence", 0.03, stats,
                                          f"Win rate {stats.win_rate:.1%} < 70% → raising threshold")
        return None

    def _adjust_max_price(self, stats: TradeStats) -> str | None:
        """Adjust max_price based on ROI."""
        param = self._config.max_price

        if stats.avg_roi > 3.0:
            # Great ROI → can pay more for shares
            return self._apply_adjustment(param, "max_price", 0.02, stats,
                                          f"ROI {stats.avg_roi:.1f}x > 3x → raising max price")
        elif stats.avg_roi < 1.5:
            # Poor ROI → need cheaper shares
            return self._apply_adjustment(param, "max_price", -0.03, stats,
                                          f"ROI {stats.avg_roi:.1f}x < 1.5x → lowering max price")
        return None

    def _adjust_bet_pct(self, stats: TradeStats) -> str | None:
        """Adjust bet_pct based on sustained win rate."""
        param = self._config.bet_pct

        if stats.win_rate > 0.80 and stats.resolved_trades >= 30:
            return self._apply_adjustment(param, "bet_pct", 0.005, stats,
                                          f"Win rate {stats.win_rate:.1%} > 80% over {stats.resolved_trades} → increasing bet size")
        elif stats.win_rate < 0.65:
            return self._apply_adjustment(param, "bet_pct", -0.01, stats,
                                          f"Win rate {stats.win_rate:.1%} < 65% → decreasing bet size")
        return None

    def _apply_adjustment(
        self, param: TunableParam, name: str, delta: float, stats: TradeStats, reason: str,
    ) -> str | None:
        """Apply an adjustment and log it to DB."""
        old_value = param.value

        if not param.adjust(delta):
            log.info(
                "adjustment_blocked",
                parameter=name,
                delta=delta,
                reason="bounds_or_consecutive_limit",
            )
            return None

        # Log to DB
        session = get_session()
        try:
            record = ParameterAdjustment(
                parameter_name=name,
                old_value=old_value,
                new_value=param.value,
                delta=param.value - old_value,
                reason=reason,
                win_rate=stats.win_rate,
                sample_size=stats.resolved_trades,
            )
            session.add(record)
            session.commit()
        except Exception as e:
            session.rollback()
            log.error("adjustment_log_failed", error=str(e))
        finally:
            session.close()

        msg = f"{name}: {old_value:.4f} → {param.value:.4f} ({reason})"
        log.info("parameter_adjusted", parameter=name, old=old_value, new=param.value, reason=reason)
        return msg
