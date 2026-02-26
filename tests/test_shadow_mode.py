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
