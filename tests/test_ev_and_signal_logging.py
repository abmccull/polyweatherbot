"""Tests for EV gate behavior and signal candidate logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from types import SimpleNamespace

import pytest
import pytz

from config import Config
from db.engine import get_session
from db.models import SignalCandidate, Trade
from signals.confidence import ConfidenceFactors
from signals.detector import SignalDetector, TradeSignal
from trading.executor import TradeExecutor
from trading.portfolio import Portfolio
from weather.metar_feed import DailyHigh
from weather.temperature import PreciseTemp, Precision


def _signal(price: float, q: float, best_bid: float | None = None, ask_depth: float = 1000.0) -> TradeSignal:
    return TradeSignal(
        event_id="evt1",
        condition_id="cond1",
        city="Chicago",
        market_date=date(2026, 2, 24),
        icao_station="KORD",
        token_id="tok1",
        bucket_type="geq",
        bucket_value=30,
        unit="F",
        market_type="temperature",
        confidence=ConfidenceFactors(
            base=q,
            precision_bonus=0.0,
            margin_bonus=0.0,
            wu_lag_bonus=0.0,
            peak_hours_bonus=0.0,
            recency_bonus=0.0,
            historical_blend=0.0,
            calibration_adjustment=0.0,
            total=q,
        ),
        metar_temp_c=20.0,
        metar_precision="tenths",
        wu_displayed_high=None,
        margin_from_boundary=1.0,
        local_hour_val=14,
        best_ask=price,
        ask_depth=ask_depth,
        best_bid=best_bid,
    )


@pytest.mark.asyncio
async def test_ev_gate_blocks_low_edge(db_session):
    cfg = Config(
        dry_run=True,
        initial_bankroll=1000.0,
        kelly_mode=False,
        enable_ev_gate=True,
        min_expected_edge=0.03,
        min_expected_profit=0.5,
    )
    portfolio = Portfolio(cfg)
    executor = TradeExecutor(cfg, portfolio, tracker=None)

    sig = _signal(price=0.60, q=0.62, best_bid=0.59, ask_depth=1000.0)
    trade = await executor.execute(sig)
    assert trade is None

    s = get_session()
    try:
        count = s.query(Trade).filter(Trade.token_id == "tok1").count()
    finally:
        s.close()
    assert count == 0


@pytest.mark.asyncio
async def test_ev_gate_allows_high_edge_and_records_expectancy(db_session):
    cfg = Config(
        dry_run=True,
        initial_bankroll=1000.0,
        kelly_mode=False,
        enable_ev_gate=True,
        min_expected_edge=0.01,
        min_expected_profit=0.1,
    )
    portfolio = Portfolio(cfg)
    executor = TradeExecutor(cfg, portfolio, tracker=None)

    sig = _signal(price=0.40, q=0.90, best_bid=0.39, ask_depth=1000.0)
    trade = await executor.execute(sig)
    assert trade is not None
    assert trade.order_status == "DRY_RUN"
    assert trade.expected_edge is not None and trade.expected_edge > 0
    assert trade.expected_profit is not None and trade.expected_profit > 0
    assert trade.expected_slippage is not None and trade.expected_slippage >= 0
    assert trade.calibrated_probability == pytest.approx(sig.confidence.total)


@pytest.mark.asyncio
async def test_ev_gate_uses_calibrated_win_probability(db_session):
    cfg = Config(
        dry_run=True,
        initial_bankroll=1000.0,
        kelly_mode=False,
        enable_ev_gate=True,
        min_expected_edge=0.12,
        min_expected_profit=0.1,
    )
    portfolio = Portfolio(cfg)
    executor = TradeExecutor(cfg, portfolio, tracker=None)

    sig = _signal(price=0.60, q=0.62, best_bid=0.59, ask_depth=1000.0)
    # Raw confidence would be blocked; calibrated model probability should pass.
    sig.win_probability = 0.92
    trade = await executor.execute(sig)
    assert trade is not None
    assert trade.expected_edge is not None and trade.expected_edge > 0.12
    assert trade.calibrated_probability == pytest.approx(0.92)


def test_dynamic_slippage_uses_tracker_estimate(db_session):
    class _Tracker:
        def get_recent_buy_slippage_bps(self) -> float:
            return 120.0

        def get_stats(self):
            return SimpleNamespace(resolved_trades=0)

    cfg = Config(
        dry_run=True,
        kelly_mode=False,
        ev_base_slippage_bps=20.0,
        ev_dynamic_slippage=True,
    )
    portfolio = Portfolio(cfg)
    executor = TradeExecutor(cfg, portfolio, tracker=_Tracker())

    slip = executor._estimate_slippage_per_share(
        price=0.50,
        size_usd=20.0,
        ask_depth=1000.0,
        best_bid=0.49,
    )

    # 120 bps baseline => at least 0.006/share before spread/depth add-ons.
    assert slip >= 0.006


@pytest.mark.asyncio
async def test_detector_persists_signal_candidates(db_session):
    @dataclass
    class _Bucket:
        bucket_type: str
        bucket_value: int
        unit: str

    @dataclass
    class _BucketData:
        token_id: str
        condition_id: str
        bucket: _Bucket

    market = SimpleNamespace(
        event_id="evt2",
        icao="KORD",
        timezone="America/Chicago",
        market_type="temperature",
        info=SimpleNamespace(city="Chicago", market_date=date(2026, 2, 24)),
        buckets=[
            _BucketData(token_id="tok_emit", condition_id="c1", bucket=_Bucket("geq", 68, "F")),
            _BucketData(token_id="tok_low_conf", condition_id="c2", bucket=_Bucket("geq", 80, "F")),
        ],
    )
    active = SimpleNamespace(
        market=market,
        bucket_prices={
            "tok_emit": SimpleNamespace(best_bid=0.41, best_ask=0.43, bid_depth=200.0, ask_depth=150.0),
            "tok_low_conf": SimpleNamespace(best_bid=0.41, best_ask=0.43, bid_depth=200.0, ask_depth=150.0),
        },
    )

    class _Registry:
        def get_all_active(self):
            return [active]

    class _Metar:
        def get_daily_high(self, _icao):
            dh = DailyHigh(station="KORD", date=date(2026, 2, 24))
            dh.high = PreciseTemp(celsius=20.0, precision=Precision.TENTHS)  # ~68F
            dh.last_obs_time = datetime.utcnow().replace(tzinfo=pytz.utc)
            return dh

    cfg = Config(
        dry_run=True,
        min_price=0.30,
        geq_min_hour=0,
    )
    cfg.min_confidence.value = 0.75

    detector = SignalDetector(cfg, _Metar(), _Registry(), tracker=None, calibrator=None)
    signals = await detector.detect()
    assert len(signals) == 1
    assert signals[0].token_id == "tok_emit"

    s = get_session()
    try:
        rows = s.query(SignalCandidate).filter(SignalCandidate.event_id == "evt2").all()
    finally:
        s.close()

    statuses = {r.token_id: r.status for r in rows}
    assert statuses["tok_emit"] == "EMITTED"
    assert statuses["tok_low_conf"] in {"BUCKET_MISS", "CONFIDENCE_BLOCKED"}
    emitted = [r for r in rows if r.token_id == "tok_emit"][0]
    assert emitted.calibrated_probability is not None
