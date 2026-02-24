"""Tests for realtime market feed + order reconciliation foundations."""

from __future__ import annotations

from config import Config
from db.engine import get_session
from db.models import Trade
from markets.realtime import ClobMarketFeed
from trading.portfolio import Portfolio
from trading.reconciliation import ExecutionReconciler


def _make_trade(
    session,
    *,
    action: str = "BUY",
    token_id: str = "tok1",
    order_id: str = "ord1",
    order_status: str = "OPEN",
    requested_size: float = 100.0,
    requested_cost: float = 40.0,
    size: float = 0.0,
    cost: float = 0.0,
) -> Trade:
    trade = Trade(
        event_id="evt1",
        condition_id="cond1",
        city="Chicago",
        market_date="2025-01-01",
        icao_station="KORD",
        bucket_value=30,
        bucket_type="geq",
        bucket_unit="F",
        token_id=token_id,
        market_type="temperature",
        action=action,
        requested_size=requested_size,
        requested_cost=requested_cost,
        price=0.40,
        size=size,
        cost=cost,
        order_id=order_id,
        order_status=order_status,
        confidence=0.90,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def test_market_feed_ingest_book_event():
    cfg = Config(dry_run=True)
    feed = ClobMarketFeed(cfg)

    feed.ingest_message(
        {
            "event_type": "book",
            "asset_id": "tok_book",
            "bids": [{"price": "0.42", "size": "50"}, {"price": "0.41", "size": "20"}],
            "asks": [{"price": "0.45", "size": "30"}, {"price": "0.46", "size": "10"}],
        }
    )

    px = feed.get_price("tok_book")
    assert px is not None
    assert px.best_bid == 0.42
    assert px.best_ask == 0.45
    assert px.bid_depth == 0.42 * 50 + 0.41 * 20
    assert px.ask_depth == 0.45 * 30 + 0.46 * 10


def test_market_feed_ingest_nested_payload():
    cfg = Config(dry_run=True)
    feed = ClobMarketFeed(cfg)

    feed.ingest_message(
        {
            "type": "snapshot",
            "data": [
                {
                    "token_id": "tok_nested",
                    "bids": [["0.30", "10"]],
                    "asks": [["0.33", "11"]],
                }
            ],
        }
    )

    px = feed.get_price("tok_nested")
    assert px is not None
    assert px.best_bid == 0.30
    assert px.best_ask == 0.33


def test_reconciler_applies_cumulative_fills(db_session):
    _make_trade(db_session, order_id="ord_fill", size=0.0, cost=0.0, order_status="OPEN")
    reconciler = ExecutionReconciler(Config(dry_run=False))

    reconciler.process_event(
        {
            "order_id": "ord_fill",
            "status": "MATCHED",
            "filled_size": "60",
            "fill_price": "0.40",
        }
    )

    s = get_session()
    try:
        t = s.query(Trade).filter(Trade.order_id == "ord_fill").first()
    finally:
        s.close()

    assert t is not None
    assert t.order_status == "FILLED"
    assert t.size == 60.0
    assert t.cost == 24.0
    assert t.fill_price == 0.4

    reconciler.process_event(
        {
            "order_id": "ord_fill",
            "status": "FILLED",
            "filled_size": "100",
            "fill_price": "0.41",
        }
    )

    s = get_session()
    try:
        t = s.query(Trade).filter(Trade.order_id == "ord_fill").first()
    finally:
        s.close()

    assert t is not None
    assert t.order_status == "FILLED"
    assert t.size == 100.0
    assert t.cost == 41.0
    assert t.fill_price == 0.41


def test_reconciler_cancel_unfilled_zeroes_trade(db_session):
    _make_trade(db_session, order_id="ord_cancel", size=0.0, cost=0.0, order_status="OPEN")
    reconciler = ExecutionReconciler(Config(dry_run=False))
    reconciler.process_event({"order_id": "ord_cancel", "status": "CANCELED"})

    s = get_session()
    try:
        t = s.query(Trade).filter(Trade.order_id == "ord_cancel").first()
    finally:
        s.close()

    assert t is not None
    assert t.order_status == "CANCELED"
    assert t.size == 0.0
    assert t.cost == 0.0
    assert (t.fee_paid or 0.0) == 0.0


def test_portfolio_deployed_includes_pending_buy_reserve(db_session):
    cfg = Config(dry_run=False, initial_bankroll=1000.0)
    _make_trade(
        db_session,
        order_id="ord_pending",
        order_status="OPEN",
        requested_size=100.0,
        requested_cost=30.0,
        size=0.0,
        cost=0.0,
    )
    _make_trade(
        db_session,
        order_id="ord_partial",
        order_status="OPEN",
        requested_size=100.0,
        requested_cost=30.0,
        size=10.0,
        cost=5.0,
    )

    portfolio = Portfolio(cfg)
    # pending: 30 + (30 - 5) = 55, plus filled cost 5 -> total 60
    assert portfolio.get_deployed() == 60.0
