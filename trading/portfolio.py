"""Balance tracking, exposure limits, and circuit breakers."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, or_

from config import Config
from db.engine import get_session
from db.models import Trade
from db.state import save_state, load_state
from utils.logging import get_logger

log = get_logger("portfolio")

NON_EXECUTED_STATUSES = ("FAILED", "CANCELED", "REJECTED")
TERMINAL_ORDER_STATUSES = ("FILLED", "MATCHED", "FAILED", "CANCELED", "REJECTED")


def _is_executed_order() -> object:
    """SQLAlchemy predicate for trades that should impact economics."""
    return or_(
        Trade.order_status.is_(None),
        Trade.order_status.notin_(NON_EXECUTED_STATUSES),
    )


def _is_open_order() -> object:
    return or_(
        Trade.order_status.is_(None),
        Trade.order_status.notin_(TERMINAL_ORDER_STATUSES),
    )


class Portfolio:
    """Tracks portfolio value, exposure, and enforces risk limits."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._consecutive_losses = 0
        self._circuit_breaker_until: datetime | None = None
        self._peak_value: float = config.initial_bankroll

    def restore_state(self) -> None:
        """Restore persisted state from DB. Call after init_db()."""
        self._peak_value = load_state("portfolio.peak_value", self._config.initial_bankroll)
        self._consecutive_losses = load_state("portfolio.consecutive_losses", 0)
        cb_until = load_state("portfolio.circuit_breaker_until", None)
        if cb_until:
            self._circuit_breaker_until = datetime.fromisoformat(cb_until)
        log.info(
            "portfolio_state_restored",
            peak_value=round(self._peak_value, 2),
            consecutive_losses=self._consecutive_losses,
            circuit_breaker_active=self._circuit_breaker_until is not None,
        )

    @property
    def initial_bankroll(self) -> float:
        return self._config.initial_bankroll

    def get_value(self) -> float:
        """Current portfolio value: initial bankroll + realized P&L."""
        session = get_session()
        try:
            total_pnl = session.query(func.sum(Trade.pnl)).filter(
                Trade.pnl.isnot(None)
            ).scalar() or 0.0

            # Unrealized: sum of costs for open (unresolved) BUY trades
            open_cost = session.query(func.sum(Trade.cost)).filter(
                Trade.action == "BUY",
                Trade.resolved_correct.is_(None),
                _is_executed_order(),
            ).scalar() or 0.0

            value = self._config.initial_bankroll + total_pnl - open_cost
            # Track all-time high
            if value > self._peak_value:
                self._peak_value = value
                save_state("portfolio.peak_value", self._peak_value)
            return value
        finally:
            session.close()

    @property
    def peak_value(self) -> float:
        """All-time high portfolio value (for drawdown throttle)."""
        return self._peak_value

    def get_deployed(self) -> float:
        """Total capital currently deployed in open BUY positions."""
        session = get_session()
        try:
            filled_deployed = session.query(func.sum(Trade.cost)).filter(
                Trade.action == "BUY",
                Trade.resolved_correct.is_(None),
                _is_executed_order(),
            ).scalar() or 0.0

            # Reserve for working BUY orders not fully filled yet.
            pending_rows = session.query(Trade.requested_cost, Trade.cost).filter(
                Trade.action == "BUY",
                Trade.resolved_correct.is_(None),
                or_(Trade.order_status.is_(None), Trade.order_status != "DRY_RUN"),
                _is_open_order(),
                _is_executed_order(),
            ).all()
            pending_reserve = sum(max((req or 0.0) - (filled or 0.0), 0.0) for req, filled in pending_rows)
            return filled_deployed + pending_reserve
        finally:
            session.close()

    def get_market_exposure(self, event_id: str) -> float:
        """Capital deployed in a specific market (BUY trades only)."""
        session = get_session()
        try:
            filled_exposure = session.query(func.sum(Trade.cost)).filter(
                Trade.action == "BUY",
                Trade.event_id == event_id,
                Trade.resolved_correct.is_(None),
                _is_executed_order(),
            ).scalar() or 0.0

            pending_rows = session.query(Trade.requested_cost, Trade.cost).filter(
                Trade.action == "BUY",
                Trade.event_id == event_id,
                Trade.resolved_correct.is_(None),
                or_(Trade.order_status.is_(None), Trade.order_status != "DRY_RUN"),
                _is_open_order(),
                _is_executed_order(),
            ).all()
            pending_reserve = sum(max((req or 0.0) - (filled or 0.0), 0.0) for req, filled in pending_rows)
            return filled_exposure + pending_reserve
        finally:
            session.close()

    def can_trade(self, event_id: str, trade_cost: float) -> tuple[bool, str]:
        """Check if a trade is allowed under risk limits.

        Returns:
            (allowed, reason) tuple
        """
        # Circuit breaker check
        if self._circuit_breaker_until is not None:
            if datetime.utcnow() < self._circuit_breaker_until:
                remaining = (self._circuit_breaker_until - datetime.utcnow()).total_seconds() / 60
                return False, f"Circuit breaker active ({remaining:.0f} min remaining)"
            else:
                self._circuit_breaker_until = None
                self._consecutive_losses = 0
                log.info("circuit_breaker_cleared")

        portfolio_value = self.get_value()

        # Portfolio halt check
        halt_threshold = self._config.initial_bankroll * self._config.halt_portfolio_pct
        if portfolio_value < halt_threshold:
            return False, f"Portfolio ({portfolio_value:.2f}) below halt threshold ({halt_threshold:.2f})"

        # Single market exposure limit
        market_exposure = self.get_market_exposure(event_id)
        max_market = portfolio_value * self._config.max_single_market_exposure
        if market_exposure + trade_cost > max_market:
            return False, f"Market exposure ({market_exposure + trade_cost:.2f}) exceeds limit ({max_market:.2f})"

        # Total deployed limit
        total_deployed = self.get_deployed()
        max_deployed = portfolio_value * self._config.max_total_deployed
        if total_deployed + trade_cost > max_deployed:
            return False, f"Total deployed ({total_deployed + trade_cost:.2f}) exceeds limit ({max_deployed:.2f})"

        return True, "ok"

    def record_outcome(self, won: bool) -> None:
        """Record a trade outcome for circuit breaker logic."""
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._config.consecutive_loss_limit:
                hours = self._config.circuit_breaker_hours
                self._circuit_breaker_until = datetime.utcnow() + timedelta(hours=hours)
                log.warning(
                    "circuit_breaker_triggered",
                    consecutive_losses=self._consecutive_losses,
                    pause_hours=hours,
                )
        # Persist circuit breaker state
        save_state("portfolio.consecutive_losses", self._consecutive_losses)
        cb_iso = self._circuit_breaker_until.isoformat() if self._circuit_breaker_until else None
        save_state("portfolio.circuit_breaker_until", cb_iso)
