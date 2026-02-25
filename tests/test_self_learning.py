"""Tests for contextual self-learning probability model."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config import Config
from db.models import Trade
from learning.self_learner import SelfLearner


def _resolved_trade(
    *,
    token_id: str,
    city: str,
    hour: int,
    bucket_type: str,
    precision: str,
    won: bool,
) -> Trade:
    now = datetime.utcnow()
    return Trade(
        event_id="evt_self",
        condition_id="cond_self",
        city=city,
        market_date="2026-02-24",
        icao_station="KORD",
        bucket_value=70,
        bucket_type=bucket_type,
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
        fill_price=0.40,
        confidence=0.90,
        metar_precision=precision,
        local_hour=hour,
        resolved_correct=won,
        pnl=10.0 if won else -10.0,
        resolved_at=now - timedelta(hours=1),
        created_at=now - timedelta(hours=2),
    )


def test_self_learner_retrain_and_context_prediction(db_session):
    trades: list[Trade] = []
    for i in range(30):
        trades.append(
            _resolved_trade(
                token_id=f"chi_{i}",
                city="Chicago",
                hour=14,
                bucket_type="geq",
                precision="tenths",
                won=i < 27,
            )
        )
    for i in range(30):
        trades.append(
            _resolved_trade(
                token_id=f"phx_{i}",
                city="Phoenix",
                hour=14,
                bucket_type="geq",
                precision="tenths",
                won=i < 12,
            )
        )
    db_session.add_all(trades)
    db_session.commit()

    cfg = Config(
        dry_run=True,
        self_learning_min_samples=20,
        self_learning_min_segment_samples=3,
        self_learning_reliability_samples=50,
    )
    learner = SelfLearner(cfg)
    diagnostics = learner.retrain()

    assert diagnostics["trained_samples"] == 60
    assert diagnostics["city_segments"] >= 2

    chicago = learner.get_context_signal("Chicago", 14, "geq", "tenths")
    phoenix = learner.get_context_signal("Phoenix", 14, "geq", "tenths")

    assert chicago is not None
    assert phoenix is not None
    assert chicago.context_probability > phoenix.context_probability
    assert chicago.confidence_adjustment > 0
    assert phoenix.confidence_adjustment < 0

    base_probability = diagnostics["global_probability"]
    assert learner.blend_probability(base_probability, chicago) > base_probability
    assert learner.blend_probability(base_probability, phoenix) < base_probability


def test_self_learner_persists_and_restores_state(db_session):
    for i in range(12):
        db_session.add(
            _resolved_trade(
                token_id=f"persist_{i}",
                city="Chicago",
                hour=15,
                bucket_type="leq",
                precision="whole",
                won=i < 9,
            )
        )
    db_session.commit()

    cfg = Config(
        dry_run=True,
        self_learning_min_samples=5,
        self_learning_min_segment_samples=1,
    )
    learner = SelfLearner(cfg)
    learner.retrain()
    expected = learner.get_context_signal("Chicago", 15, "leq", "whole")
    assert expected is not None

    restored = SelfLearner(cfg)
    assert restored.restore() is True
    actual = restored.get_context_signal("Chicago", 15, "leq", "whole")
    assert actual is not None
    assert actual.context_probability == pytest.approx(expected.context_probability, abs=1e-8)
    assert actual.confidence_adjustment == pytest.approx(expected.confidence_adjustment, abs=1e-8)


def test_self_learner_respects_min_sample_gate(db_session):
    for i in range(6):
        db_session.add(
            _resolved_trade(
                token_id=f"min_gate_{i}",
                city="Dallas",
                hour=13,
                bucket_type="geq",
                precision="tenths",
                won=True,
            )
        )
    db_session.commit()

    cfg = Config(
        dry_run=True,
        self_learning_min_samples=10,
        self_learning_min_segment_samples=1,
    )
    learner = SelfLearner(cfg)
    learner.retrain()
    assert learner.get_context_signal("Dallas", 13, "geq", "tenths") is None
