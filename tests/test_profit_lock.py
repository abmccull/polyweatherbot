"""Profit lock and trailing stop math tests."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from config import Config
from db.models import Trade
from db.engine import get_session
from trading.position_manager import PositionManager, OpenPosition


def _make_buy(session, token_id="tok1", price=0.10, shares=200.0, **kw) -> Trade:
    """Helper: insert a BUY trade."""
    trade = Trade(
        event_id="evt1", condition_id="cond1", city="Chicago",
        market_date="2025-01-01", icao_station="KORD",
        bucket_value=30, bucket_type="geq", bucket_unit="F",
        token_id=token_id, market_type="temperature",
        action="BUY", price=price, size=shares,
        cost=price * shares, confidence=0.90,
        order_status="DRY_RUN",
        **kw,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def _make_sell(session, token_id="tok1", price=0.50, shares=50.0, parent_id=None) -> Trade:
    """Helper: insert a SELL trade."""
    proceeds = shares * price
    trade = Trade(
        event_id="evt1", condition_id="", city="Chicago",
        market_date="", icao_station="",
        bucket_value=0, bucket_type="", bucket_unit="",
        token_id=token_id, market_type="temperature",
        action="SELL", price=price, size=shares,
        cost=proceeds, confidence=0.0,
        fee_paid=proceeds * 0.01, exit_reason="PROFIT_LOCK",
        parent_trade_id=parent_id, order_status="DRY_RUN",
        pnl=None, resolved_correct=None, resolved_at=None,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


@pytest.fixture
def config():
    return Config(
        dry_run=True,
        initial_bankroll=1000.0,
        profit_lock_trigger_ratio=3.0,
        profit_lock_recoup_multiple=2.0,
        trailing_stop_pct=0.35,
    )


class TestProfitLockTrigger:
    """Profit lock trigger price calculation."""

    def test_trigger_at_correct_multiple(self, config):
        """Trigger when current_price >= trigger_ratio * avg_entry."""
        avg_entry = 0.10
        trigger_price = avg_entry * config.profit_lock_trigger_ratio
        assert trigger_price == pytest.approx(0.30, rel=1e-3)

        # Below trigger: no lock
        assert 0.29 < trigger_price
        # At trigger: lock (use approx for float comparison)
        assert trigger_price == pytest.approx(0.30, abs=1e-9)


class TestRecoupShareCalc:
    """Share calculation for profit lock recoup."""

    def test_recoup_shares_basic(self, config):
        """Shares to sell = recoup_target / current_price."""
        total_cost = 20.0  # 200 shares * $0.10
        recoup_target = config.profit_lock_recoup_multiple * total_cost  # 40.0
        sell_proceeds_so_far = 0.0
        remaining_recoup = recoup_target - sell_proceeds_so_far

        current_price = 0.40
        shares_to_sell = remaining_recoup / current_price
        assert shares_to_sell == pytest.approx(100.0, rel=1e-3)

    def test_recoup_with_prior_sells(self, config):
        """Prior sell proceeds reduce remaining recoup target."""
        total_cost = 20.0
        recoup_target = config.profit_lock_recoup_multiple * total_cost  # 40.0
        sell_proceeds_so_far = 15.0  # already sold some
        remaining_recoup = recoup_target - sell_proceeds_so_far  # 25.0

        current_price = 0.50
        shares_to_sell = remaining_recoup / current_price
        assert shares_to_sell == pytest.approx(50.0, rel=1e-3)

    def test_already_recouped(self, config):
        """If sells already exceed recoup target, remaining <= 0."""
        total_cost = 20.0
        recoup_target = config.profit_lock_recoup_multiple * total_cost  # 40.0
        sell_proceeds_so_far = 45.0
        remaining_recoup = recoup_target - sell_proceeds_so_far
        assert remaining_recoup <= 0


class TestTrailingStopPrice:
    """Trailing stop price calculation."""

    def test_stop_price_from_peak(self, config):
        """Stop triggers at peak * (1 - trailing_stop_pct)."""
        peak = 0.80
        stop_price = peak * (1.0 - config.trailing_stop_pct)
        assert stop_price == pytest.approx(0.52, rel=1e-3)

    def test_above_stop_no_trigger(self, config):
        """Price above stop price -> no trigger."""
        peak = 0.80
        stop_price = peak * (1.0 - config.trailing_stop_pct)
        current = 0.60
        assert current > stop_price

    def test_at_stop_triggers(self, config):
        """Price at or below stop price -> trigger."""
        peak = 0.80
        stop_price = peak * (1.0 - config.trailing_stop_pct)
        current = 0.50
        assert current <= stop_price


class TestPositionAggregation:
    """Position aggregation math from DB trades."""

    def test_net_shares_after_sell(self, db_session, config):
        """Net shares = buy shares - sell shares."""
        from trading.executor import TradeExecutor
        from trading.portfolio import Portfolio

        buy = _make_buy(db_session, price=0.10, shares=200.0)
        _make_sell(db_session, price=0.50, shares=80.0, parent_id=buy.id)

        portfolio = Portfolio(config)
        executor = TradeExecutor(config, portfolio)
        pm = PositionManager(executor, config)

        positions = pm.get_open_positions()
        assert len(positions) == 1
        assert positions[0].total_shares == pytest.approx(120.0, rel=1e-3)
        assert positions[0].total_sold_shares == pytest.approx(80.0, rel=1e-3)
        assert positions[0].total_sell_proceeds == pytest.approx(40.0, rel=1e-3)

    def test_fully_closed_position_excluded(self, db_session, config):
        """Position with net shares <= 0.01 should not appear."""
        from trading.executor import TradeExecutor
        from trading.portfolio import Portfolio

        buy = _make_buy(db_session, price=0.10, shares=100.0)
        _make_sell(db_session, price=0.50, shares=100.0, parent_id=buy.id)

        portfolio = Portfolio(config)
        executor = TradeExecutor(config, portfolio)
        pm = PositionManager(executor, config)

        positions = pm.get_open_positions()
        assert len(positions) == 0


class TestHardStopLoss:
    """Downside protection exits when position drawdown exceeds threshold."""

    @pytest.mark.asyncio
    async def test_hard_stop_loss_sells_full_position(self, db_session):
        cfg = Config(
            dry_run=True,
            initial_bankroll=1000.0,
            profit_lock_enabled=True,
            trailing_stop_enabled=True,
            hard_stop_loss_enabled=True,
            hard_stop_loss_pct=0.65,
        )

        buy = _make_buy(db_session, token_id="tok_stop", price=0.50, shares=100.0)

        class _Executor:
            def __init__(self):
                self.calls = []

            async def execute_sell(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(id=999, exit_reason=kwargs["reason"], size=kwargs["shares"])

        executor = _Executor()
        pm = PositionManager(executor, cfg)

        exits = await pm.check_positions(price_getter=lambda tid: 0.15 if tid == "tok_stop" else None)
        assert len(exits) == 1
        assert exits[0].exit_reason == "STOP_LOSS"

        assert len(executor.calls) == 1
        call = executor.calls[0]
        assert call["token_id"] == "tok_stop"
        assert call["reason"] == "STOP_LOSS"
        assert call["shares"] == pytest.approx(100.0, rel=1e-6)
        assert call["parent_trade_id"] == buy.id

    @pytest.mark.asyncio
    async def test_hard_stop_loss_not_triggered_above_threshold(self, db_session):
        cfg = Config(
            dry_run=True,
            initial_bankroll=1000.0,
            hard_stop_loss_enabled=True,
            hard_stop_loss_pct=0.65,
        )
        _make_buy(db_session, token_id="tok_no_stop", price=0.50, shares=100.0)

        class _Executor:
            def __init__(self):
                self.calls = []

            async def execute_sell(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(id=1000, exit_reason=kwargs["reason"], size=kwargs["shares"])

        executor = _Executor()
        pm = PositionManager(executor, cfg)

        # Threshold is 0.175; current price is above threshold.
        exits = await pm.check_positions(price_getter=lambda tid: 0.20 if tid == "tok_no_stop" else None)
        assert exits == []
        assert executor.calls == []
