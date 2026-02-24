"""Position management: hard stop-loss, profit lock, and trailing stop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy import or_

from config import Config
from db.engine import get_session
from db.models import Trade
from db.state import save_state, load_state
from trading.executor import TradeExecutor
from utils.logging import get_logger

log = get_logger("position_mgr")
NON_EXECUTED_STATUSES = ("FAILED", "CANCELED", "REJECTED")


def _is_executed_order() -> object:
    return or_(
        Trade.order_status.is_(None),
        Trade.order_status.notin_(NON_EXECUTED_STATUSES),
    )


@dataclass
class OpenPosition:
    """Aggregated view of an open position by token_id."""

    token_id: str
    event_id: str
    city: str
    total_shares: float          # net: buys - sells
    avg_entry_price: float       # weighted average of BUY prices
    total_cost: float            # sum of BUY costs
    total_sold_shares: float
    total_sell_proceeds: float
    profit_locked: bool          # has profit lock triggered?
    peak_price_after_lock: float # tracked in memory
    first_buy_id: int | None = None  # ID of first BUY trade for parent reference


class PositionManager:
    """Monitors open positions for profit lock and trailing stop triggers."""

    def __init__(self, executor: TradeExecutor, config: Config) -> None:
        self._executor = executor
        self._config = config
        # In-memory tracking: token_id -> {locked: bool, peak: float}
        self._lock_state: dict[str, dict] = {}

    def restore_state(self) -> None:
        """Restore persisted lock state from DB. Call after init_db()."""
        self._lock_state = load_state("position_manager.lock_state", {})
        if self._lock_state:
            log.info("lock_state_restored", positions=len(self._lock_state))

    def _persist_lock_state(self) -> None:
        """Persist current lock state to DB."""
        save_state("position_manager.lock_state", self._lock_state)

    def get_open_positions(self) -> list[OpenPosition]:
        """Aggregate all open BUY trades minus SELL trades by token_id.

        Uses a single query: fetch all trades (BUY + SELL) for token_ids
        that have unresolved BUY trades, then aggregate in Python.
        """
        session = get_session()
        try:
            # Subquery: token_ids with unresolved BUY trades
            open_token_ids = session.query(Trade.token_id).filter(
                Trade.action == "BUY",
                Trade.resolved_correct.is_(None),
                _is_executed_order(),
            ).distinct().subquery()

            # Single query: all trades for those token_ids
            all_trades = session.query(Trade).filter(
                Trade.token_id.in_(session.query(open_token_ids.c.token_id)),
            ).all()

            if not all_trades:
                return []

            # Separate into buys (unresolved, non-failed) and sells
            positions: dict[str, dict] = {}
            sell_by_token: dict[str, dict] = {}

            for t in all_trades:
                tid = t.token_id
                if t.action == "SELL" and (
                    t.order_status is None or t.order_status not in NON_EXECUTED_STATUSES
                ):
                    if tid not in sell_by_token:
                        sell_by_token[tid] = {"shares": 0.0, "proceeds": 0.0}
                    sell_by_token[tid]["shares"] += t.size
                    sell_by_token[tid]["proceeds"] += t.cost
                elif (
                    t.action == "BUY"
                    and t.resolved_correct is None
                    and (t.order_status is None or t.order_status not in NON_EXECUTED_STATUSES)
                ):
                    if tid not in positions:
                        positions[tid] = {
                            "event_id": t.event_id,
                            "city": t.city,
                            "total_shares": 0.0,
                            "weighted_price_sum": 0.0,
                            "total_cost": 0.0,
                            "first_buy_id": t.id,
                        }
                    fill = t.fill_price or t.price
                    positions[tid]["total_shares"] += t.size
                    positions[tid]["weighted_price_sum"] += fill * t.size
                    positions[tid]["total_cost"] += t.cost
                    if t.id < positions[tid]["first_buy_id"]:
                        positions[tid]["first_buy_id"] = t.id

            result = []
            for tid, pos in positions.items():
                sold = sell_by_token.get(tid, {"shares": 0.0, "proceeds": 0.0})
                net_shares = pos["total_shares"] - sold["shares"]
                if net_shares <= 0.01:  # effectively closed
                    continue

                avg_entry = pos["weighted_price_sum"] / pos["total_shares"] if pos["total_shares"] > 0 else 0.0
                state = self._lock_state.get(tid, {"locked": False, "peak": 0.0})

                result.append(OpenPosition(
                    token_id=tid,
                    event_id=pos["event_id"],
                    city=pos["city"],
                    total_shares=net_shares,
                    avg_entry_price=avg_entry,
                    total_cost=pos["total_cost"],
                    total_sold_shares=sold["shares"],
                    total_sell_proceeds=sold["proceeds"],
                    profit_locked=state["locked"],
                    peak_price_after_lock=state["peak"],
                    first_buy_id=pos["first_buy_id"],
                ))

            return result
        finally:
            session.close()

    async def check_positions(
        self, price_getter: Callable[[str], float | None],
    ) -> list[Trade]:
        """Check each open position for profit lock / trailing stop triggers.

        Args:
            price_getter: Function that returns best bid price for a token_id.

        Returns:
            List of SELL trades executed.
        """
        if not self._config.profit_lock_enabled and not self._config.trailing_stop_enabled:
            return []

        exits: list[Trade] = []

        for pos in self.get_open_positions():
            current_price = price_getter(pos.token_id)
            if current_price is None or current_price <= 0:
                continue

            state = self._lock_state.setdefault(pos.token_id, {"locked": False, "peak": 0.0})

            # Hard downside stop: cut losses before they become tail-risk events.
            if self._config.hard_stop_loss_enabled and pos.avg_entry_price > 0 and pos.total_shares > 0.01:
                stop_price = pos.avg_entry_price * (1.0 - self._config.hard_stop_loss_pct)
                if current_price <= stop_price:
                    log.warning(
                        "hard_stop_loss_triggered",
                        city=pos.city,
                        token_id=pos.token_id,
                        entry=round(pos.avg_entry_price, 3),
                        stop_price=round(stop_price, 3),
                        current=round(current_price, 3),
                        shares=round(pos.total_shares, 2),
                    )
                    trade = await self._executor.execute_sell(
                        token_id=pos.token_id,
                        shares=pos.total_shares,
                        price=current_price,
                        reason="STOP_LOSS",
                        event_id=pos.event_id,
                        city=pos.city,
                        parent_trade_id=pos.first_buy_id,
                    )
                    if trade:
                        exits.append(trade)
                        if pos.token_id in self._lock_state:
                            del self._lock_state[pos.token_id]
                            self._persist_lock_state()
                    continue

            if not state["locked"] and self._config.profit_lock_enabled:
                # Check profit lock trigger
                trigger_price = pos.avg_entry_price * self._config.profit_lock_trigger_ratio
                if current_price >= trigger_price:
                    # Sell enough shares to recoup recoup_multiple x total_cost
                    recoup_target = self._config.profit_lock_recoup_multiple * pos.total_cost
                    # Account for already-sold proceeds
                    remaining_recoup = recoup_target - pos.total_sell_proceeds
                    if remaining_recoup <= 0:
                        # Already recouped enough
                        state["locked"] = True
                        state["peak"] = current_price
                        self._persist_lock_state()
                        continue

                    shares_to_sell = remaining_recoup / current_price
                    shares_to_sell = min(shares_to_sell, pos.total_shares)

                    if shares_to_sell < 0.01:
                        continue

                    log.info(
                        "profit_lock_triggered",
                        city=pos.city,
                        token_id=pos.token_id,
                        entry=round(pos.avg_entry_price, 3),
                        current=round(current_price, 3),
                        shares_to_sell=round(shares_to_sell, 2),
                    )

                    trade = await self._executor.execute_sell(
                        token_id=pos.token_id,
                        shares=shares_to_sell,
                        price=current_price,
                        reason="PROFIT_LOCK",
                        event_id=pos.event_id,
                        city=pos.city,
                        parent_trade_id=pos.first_buy_id,
                    )
                    if trade:
                        exits.append(trade)
                        state["locked"] = True
                        state["peak"] = current_price
                        self._persist_lock_state()

            elif state["locked"] and self._config.trailing_stop_enabled:
                # Update peak
                state["peak"] = max(state["peak"], current_price)

                # Check trailing stop
                stop_price = state["peak"] * (1.0 - self._config.trailing_stop_pct)
                if current_price <= stop_price and pos.total_shares > 0.01:
                    log.info(
                        "trailing_stop_triggered",
                        city=pos.city,
                        token_id=pos.token_id,
                        peak=round(state["peak"], 3),
                        stop_price=round(stop_price, 3),
                        current=round(current_price, 3),
                        remaining_shares=round(pos.total_shares, 2),
                    )

                    trade = await self._executor.execute_sell(
                        token_id=pos.token_id,
                        shares=pos.total_shares,
                        price=current_price,
                        reason="TRAILING_STOP",
                        event_id=pos.event_id,
                        city=pos.city,
                        parent_trade_id=pos.first_buy_id,
                    )
                    if trade:
                        exits.append(trade)
                        # Clean up state since position is fully closed
                        del self._lock_state[pos.token_id]
                        self._persist_lock_state()

        return exits
