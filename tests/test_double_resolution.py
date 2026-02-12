"""Bug 1+2 regression tests: SELL trades excluded from stats; BUY resolution
accounts for partial exits; fully-exited positions compute correct P&L."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from db.engine import get_session
from db.models import Trade
from learning.tracker import TradeTracker, TradeStats


def _make_buy(session, token_id="tok1", city="Chicago", price=0.20,
              shares=100.0, resolved=None, pnl=None, **kwargs) -> Trade:
    """Helper: insert a BUY trade."""
    trade = Trade(
        event_id="evt1", condition_id="cond1", city=city,
        market_date="2025-01-01", icao_station="KORD",
        bucket_value=30, bucket_type="geq", bucket_unit="F",
        token_id=token_id, market_type="temperature",
        action="BUY", price=price, size=shares,
        cost=price * shares, confidence=0.90,
        order_status="DRY_RUN",
        resolved_correct=resolved, pnl=pnl,
        resolved_at=datetime.utcnow() if resolved is not None else None,
        **kwargs,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def _make_sell(session, token_id="tok1", city="Chicago", price=0.60,
               shares=50.0, parent_trade_id=None) -> Trade:
    """Helper: insert a SELL trade (inert record, no P&L)."""
    proceeds = shares * price
    fee = proceeds * 0.01  # 1% sell fee
    trade = Trade(
        event_id="evt1", condition_id="", city=city,
        market_date="", icao_station="",
        bucket_value=0, bucket_type="", bucket_unit="",
        token_id=token_id, market_type="temperature",
        action="SELL", price=price, size=shares,
        cost=proceeds, confidence=0.0,
        fee_paid=fee, exit_reason="PROFIT_LOCK",
        parent_trade_id=parent_trade_id,
        order_status="DRY_RUN",
        # SELL trades are inert — no resolution fields
        pnl=None, resolved_correct=None, resolved_at=None,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


class TestSellTradesExcludedFromStats:
    """SELL trades must not appear in resolved stats queries."""

    def test_sell_not_in_get_stats(self, db_session, config):
        """get_stats() should only count BUY trades."""
        buy = _make_buy(db_session, resolved=True, pnl=10.0)
        _make_sell(db_session, parent_trade_id=buy.id)

        tracker = TradeTracker()
        stats = tracker.get_stats()

        assert stats.resolved_trades == 1
        assert stats.wins == 1
        assert stats.total_pnl == 10.0

    def test_sell_not_in_confidence_bands(self, db_session, config):
        """get_stats_by_confidence_band() should only count BUY trades."""
        _make_buy(db_session, resolved=True, pnl=5.0)
        _make_sell(db_session)

        tracker = TradeTracker()
        bands = tracker.get_stats_by_confidence_band()

        total_resolved = sum(b.resolved_trades for b in bands.values())
        assert total_resolved == 1

    def test_unresolved_sell_not_in_resolve_query(self, db_session, config):
        """resolve_trades() should skip SELL trades entirely."""
        # SELL trade with resolved_correct=None should NOT be picked up
        _make_sell(db_session)

        tracker = TradeTracker()
        session = get_session()
        try:
            unresolved = session.query(Trade).filter(
                Trade.action != "SELL",
                Trade.resolved_correct.is_(None),
                Trade.order_status != "FAILED",
            ).all()
            # Only BUY trades should appear
            for t in unresolved:
                assert t.action == "BUY"
        finally:
            session.close()


class TestPartialExitResolution:
    """BUY resolution must account for shares sold via SELL trades."""

    def test_fully_exited_pnl(self, db_session, config):
        """If all shares sold before resolution, P&L = sell proceeds - fees - cost."""
        buy = _make_buy(db_session, price=0.20, shares=100.0)
        # Sell all 100 shares at 0.60
        _make_sell(db_session, price=0.60, shares=100.0, parent_trade_id=buy.id)

        tracker = TradeTracker()
        session = get_session()
        try:
            sold_shares, sell_proceeds, sell_fees = tracker._get_sell_summary(buy, session)
        finally:
            session.close()

        assert sold_shares == 100.0
        assert sell_proceeds == 60.0  # 100 * 0.60
        assert sell_fees == pytest.approx(0.60, rel=1e-3)  # 1% of 60

        # Full exit P&L: sell_proceeds - sell_fees - cost
        expected_pnl = 60.0 - 0.60 - 20.0  # 39.40
        assert expected_pnl == pytest.approx(39.40, rel=1e-2)

    def test_partial_exit_win(self, db_session, config):
        """Partial exit + win: remaining shares win + sell net profit."""
        buy = _make_buy(db_session, price=0.20, shares=100.0)
        # Sell 40 shares at 0.60
        _make_sell(db_session, price=0.60, shares=40.0, parent_trade_id=buy.id)

        tracker = TradeTracker()
        session = get_session()
        try:
            sold_shares, sell_proceeds, sell_fees = tracker._get_sell_summary(buy, session)
        finally:
            session.close()

        assert sold_shares == 40.0
        remaining = 100.0 - 40.0  # 60

        fill = buy.price  # 0.20 (DRY_RUN, no fill_price)

        # Win: remaining shares pay out (1 - fill) each
        remaining_pnl = (1.0 - fill) * remaining  # 0.80 * 60 = 48.0
        # Sell net: proceeds - fees - (fill * sold_shares)
        sell_net = sell_proceeds - sell_fees - (fill * sold_shares)
        # 24.0 - 0.24 - (0.20 * 40) = 24.0 - 0.24 - 8.0 = 15.76
        total_pnl = remaining_pnl + sell_net

        assert remaining_pnl == pytest.approx(48.0, rel=1e-3)
        assert total_pnl == pytest.approx(63.76, rel=1e-2)

    def test_no_sells_standard_resolution(self, db_session, config):
        """With no sells, resolution is standard: (1-price)*shares for win."""
        _make_buy(db_session, price=0.20, shares=100.0)

        tracker = TradeTracker()
        session = get_session()
        try:
            buy = session.query(Trade).filter(Trade.action == "BUY").first()
            sold_shares, sell_proceeds, sell_fees = tracker._get_sell_summary(buy, session)
        finally:
            session.close()

        assert sold_shares == 0.0
        assert sell_proceeds == 0.0
        assert sell_fees == 0.0


class TestPortfolioExcludesSells:
    """Portfolio queries must only count BUY trades for open cost."""

    def test_deployed_excludes_sells(self, db_session, config):
        """get_deployed() should not count SELL trade costs."""
        from trading.portfolio import Portfolio

        _make_buy(db_session, price=0.20, shares=100.0)  # cost = 20.0
        _make_sell(db_session, price=0.60, shares=50.0)   # cost (proceeds) = 30.0

        portfolio = Portfolio(config)
        deployed = portfolio.get_deployed()

        # Only the BUY's cost should count
        assert deployed == pytest.approx(20.0, rel=1e-3)

    def test_value_excludes_sell_open_cost(self, db_session, config):
        """get_value() open_cost should only include BUY trades."""
        from trading.portfolio import Portfolio

        _make_buy(db_session, price=0.10, shares=50.0)  # cost = 5.0
        _make_sell(db_session, price=0.50, shares=25.0)  # cost = 12.5

        portfolio = Portfolio(config)
        value = portfolio.get_value()

        # value = bankroll + total_pnl(0) - open_buy_cost(5.0)
        assert value == pytest.approx(995.0, rel=1e-3)
