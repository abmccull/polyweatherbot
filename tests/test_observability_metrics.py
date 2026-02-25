"""Tests for expectancy/slippage drift analytics and metrics snapshot output."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from config import Config
from db.models import SignalCandidate, Trade
from learning.tracker import ExpectancyDriftStats, SlippageDriftStats, TradeTracker
from observability import write_metrics_snapshot
from trading.portfolio import Portfolio


def _buy_trade(
    *,
    token_id: str,
    expected_edge: float,
    expected_slippage: float,
    fill_price: float,
    pnl: float,
    resolved_correct: bool,
) -> Trade:
    now = datetime.utcnow()
    return Trade(
        event_id="evt1",
        condition_id="cond1",
        city="Chicago",
        market_date="2026-02-23",
        icao_station="KORD",
        bucket_value=70,
        bucket_type="geq",
        bucket_unit="F",
        token_id=token_id,
        market_type="temperature",
        action="BUY",
        requested_size=100.0,
        requested_cost=40.0,
        price=0.40,
        size=100.0,
        cost=40.0,
        order_status="DRY_RUN",
        fill_price=fill_price,
        confidence=0.90,
        expected_edge=expected_edge,
        expected_profit=expected_edge * 100.0,
        expected_slippage=expected_slippage,
        calibrated_probability=0.9,
        resolved_correct=resolved_correct,
        pnl=pnl,
        resolved_at=now - timedelta(hours=1),
        created_at=now - timedelta(hours=2),
    )


def test_tracker_expectancy_and_slippage_drift_stats(db_session):
    db_session.add_all(
        [
            _buy_trade(
                token_id="tok_win",
                expected_edge=0.08,
                expected_slippage=0.005,
                fill_price=0.41,
                pnl=22.0,
                resolved_correct=True,
            ),
            _buy_trade(
                token_id="tok_loss",
                expected_edge=0.06,
                expected_slippage=0.004,
                fill_price=0.405,
                pnl=-8.0,
                resolved_correct=False,
            ),
        ]
    )
    db_session.commit()

    tracker = TradeTracker()
    expectancy = tracker.get_expectancy_drift_stats(lookback_days=30)
    assert expectancy.samples == 2
    assert expectancy.avg_expected_edge > 0
    assert expectancy.avg_realized_edge == ((22.0 / 100.0) + (-8.0 / 100.0)) / 2

    slippage = tracker.get_slippage_drift_stats(lookback_days=30)
    assert slippage.samples == 2
    assert slippage.avg_realized_slippage > 0
    assert slippage.p75_positive_slippage_bps > 0


def test_write_metrics_snapshot_includes_drift_and_calibration_sections(db_session, monkeypatch, tmp_path):
    class _PositionManager:
        def get_open_positions(self):
            return []

    class _Calibrator:
        def get_model_diagnostics(self):
            return {
                "sample_size": 42,
                "slope": 4.5,
                "intercept": -2.0,
                "log_loss": 0.31,
                "brier_score": 0.16,
                "ece": 0.04,
            }

    class _SelfLearner:
        def get_diagnostics(self):
            return {
                "enabled": True,
                "trained_samples": 64,
                "global_probability": 0.83,
                "city_segments": 5,
                "hour_segments": 8,
                "bucket_segments": 2,
                "precision_segments": 2,
                "updated_at": datetime.utcnow().isoformat(),
            }

    # Add one candidate so funnel/status output is non-empty.
    db_session.add(
        SignalCandidate(
            event_id="evt1",
            token_id="tok_emit",
            city="Chicago",
            market_date="2026-02-24",
            market_type="temperature",
            bucket_type="geq",
            bucket_value=70,
            bucket_unit="F",
            status="EMITTED",
            reason="all_gates_passed",
        )
    )
    db_session.commit()

    tracker = TradeTracker()
    monkeypatch.setattr(
        tracker,
        "get_expectancy_drift_stats",
        lambda lookback_days=30: ExpectancyDriftStats(
            samples=12,
            avg_expected_edge=0.07,
            avg_realized_edge=0.05,
            avg_edge_drift=-0.02,
            positive_realized_edge_rate=0.66,
        ),
    )
    monkeypatch.setattr(
        tracker,
        "get_slippage_drift_stats",
        lambda lookback_days=30: SlippageDriftStats(
            samples=12,
            avg_expected_slippage=0.006,
            avg_realized_slippage=0.009,
            avg_slippage_drift=0.003,
            p75_positive_slippage_bps=95.0,
        ),
    )
    monkeypatch.setattr(tracker, "get_recent_buy_slippage_bps", lambda lookback_trades=100: 88.0)

    metrics_path = tmp_path / "metrics.json"
    monkeypatch.setattr("observability.METRICS_FILE", str(metrics_path))

    cfg = Config(dry_run=True, initial_bankroll=1000.0)
    portfolio = Portfolio(cfg)
    write_metrics_snapshot(
        cfg,
        portfolio,
        tracker,
        _PositionManager(),
        active_markets=3,
        calibrator=_Calibrator(),
        self_learner=_SelfLearner(),
    )

    snapshot = json.loads(metrics_path.read_text())
    assert "expectancy" in snapshot
    assert snapshot["expectancy"]["samples"] == 12
    assert "slippage" in snapshot
    assert snapshot["slippage"]["recent_buy_slippage_p75_bps"] == 88.0
    assert "signal_funnel_24h" in snapshot
    assert "calibration_model" in snapshot
    assert snapshot["calibration_model"]["sample_size"] == 42
    assert "self_learning" in snapshot
    assert snapshot["self_learning"]["trained_samples"] == 64
