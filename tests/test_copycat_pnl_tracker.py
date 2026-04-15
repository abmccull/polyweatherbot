"""Regression tests for copycat PnL resolution/backfill."""

from __future__ import annotations

from datetime import datetime

import pytest

from db.models import Trade
from learning.tracker import TradeTracker


def _make_copy_trade(
    session,
    *,
    token_id: str,
    condition_id: str,
    status: str = "MATCHED",
    requested_size: float = 10.0,
    requested_cost: float = 4.0,
    size: float = 0.0,
    cost: float = 0.0,
) -> Trade:
    trade = Trade(
        event_id="copy_evt",
        condition_id=condition_id,
        city="copycat",
        market_date="2026-03-02",
        icao_station="COPY",
        bucket_value=0,
        bucket_type="copy",
        bucket_unit="",
        token_id=token_id,
        market_type="copycat",
        action="BUY",
        requested_size=requested_size,
        requested_cost=requested_cost,
        price=0.40,
        size=size,
        cost=cost,
        confidence=1.0,
        order_status=status,
        created_at=datetime.utcnow(),
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


@pytest.mark.asyncio
async def test_copycat_tracker_resolves_winning_trade_from_requested_notional(db_session):
    _make_copy_trade(
        db_session,
        token_id="tok_yes",
        condition_id="cond_yes",
        status="MATCHED",
        requested_size=10.0,
        requested_cost=4.0,
        size=0.0,
        cost=0.0,
    )
    tracker = TradeTracker()

    async def _market(_: Trade):
        return {
            "clobTokenIds": '["tok_yes","tok_no"]',
            "outcomePrices": '["1","0"]',
        }

    async def _positions():
        return {}

    tracker._fetch_copycat_market = _market  # type: ignore[method-assign]
    tracker._fetch_live_copy_positions_map = _positions  # type: ignore[method-assign]

    resolved = await tracker.resolve_trades()
    assert resolved == 1

    db_session.expire_all()
    trade = db_session.query(Trade).filter(Trade.token_id == "tok_yes").one()

    assert trade.size == pytest.approx(10.0, rel=1e-6)
    assert trade.cost == pytest.approx(4.0, rel=1e-6)
    assert trade.resolved_correct is True
    assert trade.pnl == pytest.approx(6.0, rel=1e-6)


@pytest.mark.asyncio
async def test_copycat_tracker_backfills_from_positions_for_live_status(db_session):
    _make_copy_trade(
        db_session,
        token_id="tok_live",
        condition_id="cond_live",
        status="LIVE",
        requested_size=0.0,
        requested_cost=0.0,
        size=0.0,
        cost=0.0,
    )
    tracker = TradeTracker()

    async def _market(_: Trade):
        return {
            "clobTokenIds": ["tok_live", "tok_other"],
            "outcomePrices": ["1", "0"],
        }

    async def _positions():
        return {"tok_live": {"size": 5.0, "avg_price": 0.52}}

    tracker._fetch_copycat_market = _market  # type: ignore[method-assign]
    tracker._fetch_live_copy_positions_map = _positions  # type: ignore[method-assign]

    resolved = await tracker.resolve_trades()
    assert resolved == 1

    db_session.expire_all()
    trade = db_session.query(Trade).filter(Trade.token_id == "tok_live").one()

    assert trade.size == pytest.approx(5.0, rel=1e-6)
    assert trade.cost == pytest.approx(2.6, rel=1e-6)
    assert trade.resolved_correct is True
    assert trade.pnl == pytest.approx(2.4, rel=1e-6)


@pytest.mark.asyncio
async def test_copycat_tracker_skips_unsettled_markets(db_session):
    _make_copy_trade(
        db_session,
        token_id="tok_mid",
        condition_id="cond_mid",
        status="MATCHED",
        requested_size=10.0,
        requested_cost=4.0,
        size=0.0,
        cost=0.0,
    )
    tracker = TradeTracker()

    async def _market(_: Trade):
        return {
            "clobTokenIds": ["tok_mid", "tok_other"],
            "outcomePrices": ["0.61", "0.39"],
        }

    async def _positions():
        return {}

    tracker._fetch_copycat_market = _market  # type: ignore[method-assign]
    tracker._fetch_live_copy_positions_map = _positions  # type: ignore[method-assign]

    resolved = await tracker.resolve_trades()
    assert resolved == 0

    db_session.expire_all()
    trade = db_session.query(Trade).filter(Trade.token_id == "tok_mid").one()

    assert trade.resolved_correct is None
    assert trade.pnl is None
