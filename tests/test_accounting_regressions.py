"""Regression tests for accounting, attribution, and ops hardening."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from config import Config
from db.engine import cleanup_old_data, get_session
from db.models import DailyPnL, Observation, Trade
from learning.tracker import TradeTracker
from main import StationSniper
from settlement.open_meteo import OpenMeteoClient
from signals.confidence import ConfidenceFactors
from signals.detector import TradeSignal
from trading.executor import TradeExecutor
from trading.portfolio import Portfolio
from trading.position_manager import PositionManager


def _make_buy(
    session,
    token_id: str = "tok1",
    order_status: str = "DRY_RUN",
    size: float = 100.0,
    cost: float = 20.0,
    city: str = "Chicago",
    market_date: str = "2025-01-01",
    resolved: bool | None = None,
    resolved_at: datetime | None = None,
    pnl: float | None = None,
    fee_paid: float | None = None,
    parent_trade_id: int | None = None,
) -> Trade:
    trade = Trade(
        event_id="evt1",
        condition_id="cond1",
        city=city,
        market_date=market_date,
        icao_station="KORD",
        bucket_value=30,
        bucket_type="geq",
        bucket_unit="F",
        token_id=token_id,
        market_type="temperature",
        action="BUY",
        price=0.20,
        size=size,
        cost=cost,
        confidence=0.90,
        order_status=order_status,
        resolved_correct=resolved,
        resolved_at=resolved_at,
        pnl=pnl,
        fee_paid=fee_paid,
        parent_trade_id=parent_trade_id,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def _make_sell(
    session,
    token_id: str = "tok1",
    order_status: str = "DRY_RUN",
    size: float = 50.0,
    proceeds: float = 30.0,
    fee_paid: float = 0.30,
    parent_trade_id: int | None = None,
) -> Trade:
    trade = Trade(
        event_id="evt1",
        condition_id="",
        city="Chicago",
        market_date="",
        icao_station="",
        bucket_value=0,
        bucket_type="",
        bucket_unit="",
        token_id=token_id,
        market_type="temperature",
        action="SELL",
        price=0.60,
        size=size,
        cost=proceeds,
        confidence=0.0,
        order_status=order_status,
        fee_paid=fee_paid,
        exit_reason="PROFIT_LOCK",
        parent_trade_id=parent_trade_id,
        pnl=None,
        resolved_correct=None,
        resolved_at=None,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def test_failed_buy_is_not_counted_in_portfolio(db_session):
    cfg = Config(dry_run=True, initial_bankroll=1000.0)
    _make_buy(db_session, order_status="FAILED", size=25.0, cost=12.5)

    portfolio = Portfolio(cfg)
    assert portfolio.get_value() == pytest.approx(1000.0, rel=1e-6)
    assert portfolio.get_deployed() == pytest.approx(0.0, rel=1e-6)


def test_failed_sell_does_not_close_position(db_session):
    cfg = Config(dry_run=True, initial_bankroll=1000.0)
    buy = _make_buy(db_session, order_status="DRY_RUN", size=100.0, cost=20.0)
    _make_sell(
        db_session,
        token_id=buy.token_id,
        order_status="FAILED",
        size=100.0,
        proceeds=60.0,
        fee_paid=0.6,
    )

    portfolio = Portfolio(cfg)
    executor = TradeExecutor(cfg, portfolio)
    pm = PositionManager(executor, cfg)
    positions = pm.get_open_positions()

    assert len(positions) == 1
    assert positions[0].total_shares == pytest.approx(100.0, rel=1e-6)


def test_sell_summary_prefers_parent_trade_id(db_session):
    buy1 = _make_buy(db_session, token_id="same_tok")
    buy2 = _make_buy(db_session, token_id="same_tok")
    _make_sell(db_session, token_id="same_tok", parent_trade_id=buy1.id, size=40.0, proceeds=24.0, fee_paid=0.24)

    tracker = TradeTracker()
    s = get_session()
    try:
        sold1, proceeds1, fees1 = tracker._get_sell_summary(buy1, s)
        sold2, proceeds2, fees2 = tracker._get_sell_summary(buy2, s)
    finally:
        s.close()

    assert sold1 == pytest.approx(40.0, rel=1e-6)
    assert proceeds1 == pytest.approx(24.0, rel=1e-6)
    assert fees1 == pytest.approx(0.24, rel=1e-6)
    assert sold2 == pytest.approx(0.0, rel=1e-6)
    assert proceeds2 == pytest.approx(0.0, rel=1e-6)
    assert fees2 == pytest.approx(0.0, rel=1e-6)


def test_unlinked_sell_not_attributed_when_multiple_buys(db_session):
    buy1 = _make_buy(db_session, token_id="same_tok_2")
    buy2 = _make_buy(db_session, token_id="same_tok_2")
    _make_sell(db_session, token_id="same_tok_2", parent_trade_id=None, size=10.0, proceeds=6.0, fee_paid=0.06)

    tracker = TradeTracker()
    s = get_session()
    try:
        summary1 = tracker._get_sell_summary(buy1, s)
        summary2 = tracker._get_sell_summary(buy2, s)
    finally:
        s.close()

    assert summary1 == (0.0, 0.0, 0.0)
    assert summary2 == (0.0, 0.0, 0.0)


def test_unlinked_sell_attributed_when_single_buy(db_session):
    buy = _make_buy(db_session, token_id="unique_tok")
    _make_sell(db_session, token_id="unique_tok", parent_trade_id=None, size=15.0, proceeds=9.0, fee_paid=0.09)

    tracker = TradeTracker()
    s = get_session()
    try:
        sold, proceeds, fees = tracker._get_sell_summary(buy, s)
    finally:
        s.close()

    assert sold == pytest.approx(15.0, rel=1e-6)
    assert proceeds == pytest.approx(9.0, rel=1e-6)
    assert fees == pytest.approx(0.09, rel=1e-6)


def test_cleanup_old_data_vacuum_safe(db_session):
    old_obs = Observation(
        station="KORD",
        raw_metar="METAR OLD",
        temp_c=10.0,
        precision="whole",
        obs_time=datetime.utcnow() - timedelta(days=120),
        source="metar_cache",
        created_at=datetime.utcnow() - timedelta(days=120),
    )
    new_obs = Observation(
        station="KORD",
        raw_metar="METAR NEW",
        temp_c=11.0,
        precision="whole",
        obs_time=datetime.utcnow(),
        source="metar_cache",
        created_at=datetime.utcnow(),
    )
    db_session.add(old_obs)
    db_session.add(new_obs)
    db_session.commit()

    deleted = cleanup_old_data(archive_days=90)
    assert deleted == 1

    remaining = db_session.query(Observation).count()
    assert remaining == 1


def test_daily_pnl_uses_resolution_date_utc(db_session):
    now = datetime.utcnow()
    today = now.date().isoformat()
    yesterday = (now - timedelta(days=1)).date().isoformat()

    # Should count for today's rollup (resolved today, market date in the past).
    _make_buy(
        db_session,
        token_id="today_counted",
        market_date=yesterday,
        resolved=True,
        resolved_at=now,
        pnl=10.0,
    )
    # Should not count for today's rollup (resolved yesterday, market date today).
    _make_buy(
        db_session,
        token_id="yday_not_counted",
        market_date=today,
        resolved=True,
        resolved_at=now - timedelta(days=1),
        pnl=7.0,
    )

    # Add explicit fees so we can verify gross/net split.
    s = get_session()
    try:
        t = s.query(Trade).filter(Trade.token_id == "today_counted").first()
        t.fee_paid = 1.0
        s.commit()
    finally:
        s.close()

    tracker = TradeTracker()
    tracker._update_daily_pnl()

    s = get_session()
    try:
        row = s.query(DailyPnL).filter(DailyPnL.date == today).first()
    finally:
        s.close()

    assert row is not None
    assert row.trades_count == 1
    assert row.gross_pnl == pytest.approx(11.0, rel=1e-6)  # net + fees
    assert row.fees == pytest.approx(1.0, rel=1e-6)
    assert row.net_pnl == pytest.approx(10.0, rel=1e-6)


@pytest.mark.asyncio
async def test_open_meteo_uses_passed_timezone(monkeypatch):
    captured: dict = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"daily": {"temperature_2m_max": [12.3]}}

    async def fake_get(url, params):
        captured["params"] = params
        return _FakeResponse()

    client = OpenMeteoClient()
    monkeypatch.setattr(client._client, "get", fake_get)
    try:
        tmax = await client.get_daily_tmax(
            latitude=40.0,
            longitude=-73.0,
            for_date=date(2026, 2, 24),
            timezone="America/New_York",
        )
    finally:
        await client.close()

    assert tmax == pytest.approx(12.3, rel=1e-6)
    assert captured["params"]["timezone"] == "America/New_York"


@pytest.mark.asyncio
async def test_refresh_prices_runs_in_dry_run_without_client():
    class _Bucket:
        token_id = "tok_refresh"

    class _Market:
        buckets = [_Bucket()]

    class _Active:
        market = _Market()

    class _Registry:
        def __init__(self) -> None:
            self.updated = []

        def get_all_active(self):
            return [_Active()]

        def update_prices(self, token_id, best_bid, best_ask, bid_depth, ask_depth):
            self.updated.append((token_id, best_bid, best_ask, bid_depth, ask_depth))

    class _Executor:
        def __init__(self) -> None:
            self._clob_client = None
            self.refreshed = []

        def refresh_prices(self, token_id: str):
            self.refreshed.append(token_id)
            return 0.41, 0.44, 120.0, 80.0

    fake_bot = SimpleNamespace(
        config=SimpleNamespace(dry_run=True),
        executor=_Executor(),
        registry=_Registry(),
    )

    await StationSniper._refresh_prices(fake_bot)

    assert fake_bot.executor.refreshed == ["tok_refresh"]
    assert len(fake_bot.registry.updated) == 1


@pytest.mark.asyncio
async def test_execute_buy_failure_returns_none_and_zero_impact(db_session):
    cfg = Config(dry_run=False, initial_bankroll=1000.0, kelly_mode=False)
    portfolio = Portfolio(cfg)
    executor = TradeExecutor(cfg, portfolio)
    executor._place_order = lambda *args, **kwargs: None

    signal = TradeSignal(
        event_id="evt_buy_fail",
        condition_id="cond1",
        city="Chicago",
        market_date=date(2026, 2, 24),
        icao_station="KORD",
        token_id="tok_buy_fail",
        bucket_type="geq",
        bucket_value=30,
        unit="F",
        market_type="temperature",
        confidence=ConfidenceFactors(
            base=0.6,
            precision_bonus=0.1,
            margin_bonus=0.0,
            wu_lag_bonus=0.0,
            peak_hours_bonus=0.1,
            recency_bonus=0.1,
            historical_blend=0.0,
            calibration_adjustment=0.0,
            total=0.9,
        ),
        metar_temp_c=20.0,
        metar_precision="tenths",
        wu_displayed_high=None,
        margin_from_boundary=1.0,
        local_hour_val=14,
        best_ask=0.5,
        ask_depth=1000.0,
    )

    trade = await executor.execute(signal)
    assert trade is None

    s = get_session()
    try:
        row = s.query(Trade).filter(Trade.token_id == "tok_buy_fail").first()
    finally:
        s.close()

    assert row is not None
    assert row.order_status == "FAILED"
    assert row.size == pytest.approx(0.0, rel=1e-6)
    assert row.cost == pytest.approx(0.0, rel=1e-6)
    assert (row.fee_paid or 0.0) == pytest.approx(0.0, rel=1e-6)


@pytest.mark.asyncio
async def test_execute_sell_failure_returns_none_and_zero_impact(db_session):
    cfg = Config(dry_run=False, initial_bankroll=1000.0)
    portfolio = Portfolio(cfg)
    executor = TradeExecutor(cfg, portfolio)
    executor._place_order = lambda *args, **kwargs: None

    trade = await executor.execute_sell(
        token_id="tok_sell_fail",
        shares=100.0,
        price=0.5,
        reason="TRAILING_STOP",
        event_id="evt1",
        city="Chicago",
        parent_trade_id=1,
    )
    assert trade is None

    s = get_session()
    try:
        row = s.query(Trade).filter(Trade.token_id == "tok_sell_fail").first()
    finally:
        s.close()

    assert row is not None
    assert row.order_status == "FAILED"
    assert row.size == pytest.approx(0.0, rel=1e-6)
    assert row.cost == pytest.approx(0.0, rel=1e-6)
    assert (row.fee_paid or 0.0) == pytest.approx(0.0, rel=1e-6)
