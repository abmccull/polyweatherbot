"""Tests for shadow expansion mode controls."""

from __future__ import annotations

from unittest.mock import MagicMock

from config import Config
from signals.detector import SignalDetector


def test_shadow_signal_gates_relax_detection_thresholds():
    cfg = Config(
        dry_run=True,
        min_price=0.30,
        metar_max_age_minutes=30.0,
        geq_min_hour=12,
        leq_min_hour=17,
        shadow_expansion_enabled=True,
        shadow_min_confidence=0.78,
        shadow_min_price=0.20,
        shadow_max_price=0.75,
        shadow_metar_max_age_minutes=45.0,
        shadow_geq_min_hour=9,
        shadow_leq_min_hour=15,
    )
    cfg.min_confidence.value = 0.85
    cfg.max_price.value = 0.65

    detector = SignalDetector(cfg, MagicMock(), MagicMock(), tracker=None, calibrator=None)
    gates = detector._effective_signal_gates()

    assert gates == (0.20, 0.75, 0.78, 45.0, 9, 15)


def test_shadow_signal_gates_default_when_disabled():
    cfg = Config(
        dry_run=True,
        min_price=0.30,
        metar_max_age_minutes=30.0,
        geq_min_hour=12,
        leq_min_hour=17,
        shadow_expansion_enabled=False,
    )
    cfg.min_confidence.value = 0.85
    cfg.max_price.value = 0.65

    detector = SignalDetector(cfg, MagicMock(), MagicMock(), tracker=None, calibrator=None)
    gates = detector._effective_signal_gates()

    assert gates == (0.30, 0.65, 0.85, 30.0, 12, 17)


def test_exact_bucket_shadow_pilot_requires_explicit_enable():
    cfg = Config(
        dry_run=True,
        shadow_expansion_enabled=True,
        shadow_exact_enabled=False,
        shadow_exact_min_hour=15,
    )
    detector = SignalDetector(cfg, MagicMock(), MagicMock(), tracker=None, calibrator=None)

    allowed, reason = detector._exact_bucket_allowed(local_hour_val=16, metar_precision="tenths")
    assert allowed is False
    assert reason == "exact_disabled"


def test_exact_bucket_shadow_pilot_gates_by_hour_and_precision():
    cfg = Config(
        dry_run=True,
        shadow_expansion_enabled=True,
        shadow_exact_enabled=True,
        shadow_exact_min_hour=15,
    )
    detector = SignalDetector(cfg, MagicMock(), MagicMock(), tracker=None, calibrator=None)

    too_early, early_reason = detector._exact_bucket_allowed(local_hour_val=14, metar_precision="tenths")
    assert too_early is False
    assert early_reason == "exact_before_15"

    whole_precision, precision_reason = detector._exact_bucket_allowed(local_hour_val=16, metar_precision="whole")
    assert whole_precision is False
    assert precision_reason == "exact_requires_tenths_precision"

    allowed, reason = detector._exact_bucket_allowed(local_hour_val=16, metar_precision="tenths")
    assert allowed is True
    assert reason == "exact_shadow_enabled"


def test_adaptive_time_gate_relaxation_and_reset():
    cfg = Config(
        dry_run=True,
        geq_min_hour=12,
        leq_min_hour=17,
        adaptive_time_gates_enabled=True,
        adaptive_time_gate_no_emit_cycles=1,
        adaptive_time_gate_step_hours=1,
        adaptive_time_gate_min_geq_hour=8,
        adaptive_time_gate_min_leq_hour=14,
        adaptive_time_gate_reset_on_emit=True,
    )
    cfg.min_confidence.value = 0.85
    cfg.max_price.value = 0.65
    detector = SignalDetector(cfg, MagicMock(), MagicMock(), tracker=None, calibrator=None)

    detector._update_adaptive_time_gates(emitted_count=0)
    gates = detector._effective_signal_gates()
    assert gates[4] == 11
    assert gates[5] == 16

    detector._update_adaptive_time_gates(emitted_count=1)
    reset_gates = detector._effective_signal_gates()
    assert reset_gates[4] == 12
    assert reset_gates[5] == 17
