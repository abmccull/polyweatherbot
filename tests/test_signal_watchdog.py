"""Tests for signal funnel KPI watchdog fallback actions."""

from __future__ import annotations

from config import Config
from signals.watchdog import SignalWatchdog


def test_signal_watchdog_does_not_trigger_with_low_candidate_volume():
    cfg = Config(
        dry_run=True,
        signal_kpi_enabled=True,
        signal_kpi_min_candidates=300,
        signal_kpi_min_emitted=1,
    )
    wd = SignalWatchdog(cfg)
    result = wd.evaluate_and_apply(
        {
            "candidates": 120,
            "emitted": 0,
            "status_counts": {"TIME_GATE_BLOCKED": 90, "SKIP_EXACT": 20},
        }
    )
    assert result.triggered is False
    assert result.reason == "insufficient_candidates"


def test_signal_watchdog_applies_fallback_actions_once():
    cfg = Config(
        dry_run=True,
        signal_kpi_enabled=True,
        signal_kpi_min_candidates=200,
        signal_kpi_min_emitted=1,
        signal_kpi_max_time_gate_ratio=0.60,
        signal_kpi_max_exact_ratio=0.60,
        signal_kpi_auto_shadow_expand=True,
        signal_kpi_auto_exact_pilot=True,
        signal_kpi_auto_adaptive_gates=True,
        shadow_expansion_enabled=False,
        shadow_exact_enabled=False,
        adaptive_time_gates_enabled=False,
        adaptive_time_gate_no_emit_cycles=90,
    )
    wd = SignalWatchdog(cfg)

    funnel = {
        "candidates": 500,
        "emitted": 0,
        "status_counts": {
            "TIME_GATE_BLOCKED": 320,
            "SKIP_EXACT": 360,
        },
    }
    first = wd.evaluate_and_apply(funnel)
    assert first.triggered is True
    assert "enable_shadow_expansion" in first.actions
    assert "enable_shadow_exact_pilot" in first.actions
    assert "accelerate_adaptive_time_gates" in first.actions
    assert cfg.shadow_expansion_enabled is True
    assert cfg.shadow_exact_enabled is True
    assert cfg.adaptive_time_gates_enabled is True
    assert cfg.adaptive_time_gate_no_emit_cycles == 20

    second = wd.evaluate_and_apply(funnel)
    assert second.triggered is True
    assert second.actions == []
