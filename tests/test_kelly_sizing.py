"""Kelly criterion sizing tests."""

from __future__ import annotations

import pytest

from config import Config
from trading.sizing import (
    compute_size_kelly,
    effective_max_bet,
    kelly_fraction,
    kelly_multiplier_for_confidence,
    drawdown_throttle,
)


@pytest.fixture
def config():
    return Config(dry_run=True, initial_bankroll=1000.0, dynamic_max_bet_enabled=False)


class TestKellyFraction:
    """Raw Kelly fraction computation."""

    def test_no_edge_zero(self):
        """When q <= price, no edge -> 0 or negative."""
        # q=0.20, price=0.20: no edge (fee makes it slightly negative)
        f = kelly_fraction(0.20, 0.20, 0.02)
        assert f <= 0.0

    def test_strong_edge_positive(self):
        """Strong edge produces substantial fraction."""
        # q=0.90, price=0.20: huge edge
        f = kelly_fraction(0.90, 0.20, 0.02)
        assert f > 0.5

    def test_negative_edge_zero(self):
        """Negative edge -> 0."""
        # q=0.10, price=0.50: losing proposition
        f = kelly_fraction(0.10, 0.50, 0.02)
        assert f <= 0.0


class TestComputeSizeKelly:
    """Full Kelly sizing pipeline."""

    def test_below_confidence_floor_zero(self, config):
        """Confidence below floor -> $0."""
        size = compute_size_kelly(
            config, 1000.0, confidence=0.75, ask_depth=500.0,
            price=0.20, resolved_trades=50, peak_value=1000.0,
        )
        assert size == 0.0

    def test_insufficient_edge_zero(self, config):
        """When q - price < min_edge -> $0."""
        # confidence=0.85, haircut=0.03 -> q=0.82, price=0.80 -> edge=0.02 < 0.05
        size = compute_size_kelly(
            config, 1000.0, confidence=0.85, ask_depth=500.0,
            price=0.80, resolved_trades=50, peak_value=1000.0,
        )
        assert size == 0.0

    def test_strong_edge_substantial(self, config):
        """Strong edge with full ramp -> substantial position."""
        size = compute_size_kelly(
            config, 1000.0, confidence=0.95, ask_depth=5000.0,
            price=0.15, resolved_trades=50, peak_value=1000.0,
        )
        assert size >= config.min_bet
        assert size <= config.max_bet

    def test_bounded_by_max_bet(self, config):
        """Size must not exceed max_bet."""
        size = compute_size_kelly(
            config, 10000.0, confidence=0.99, ask_depth=50000.0,
            price=0.10, resolved_trades=100, peak_value=10000.0,
        )
        assert size <= config.max_bet


class TestAdaptiveSizing:
    """Dynamic max-bet and aggression behavior."""

    def test_dynamic_max_bet_scales_with_bankroll(self):
        cfg = Config(
            dry_run=True,
            initial_bankroll=1000.0,
            dynamic_max_bet_enabled=True,
            dynamic_max_bet_pct=0.10,
            dynamic_max_bet_floor=150.0,
            dynamic_max_bet_cap=5000.0,
            aggression_enabled=False,
        )
        # 10% of bankroll (500) should set max-bet above floor
        cap = effective_max_bet(cfg, portfolio_value=5000.0, peak_value=5000.0)
        assert cap == pytest.approx(500.0, rel=1e-6)

        size = compute_size_kelly(
            cfg,
            portfolio_value=5000.0,
            confidence=0.97,
            ask_depth=100000.0,
            price=0.12,
            resolved_trades=200,
            peak_value=5000.0,
        )
        assert size <= cap
        assert size > 150.0

    def test_aggression_boost_expands_max_bet_when_win_rate_is_strong(self):
        cfg = Config(
            dry_run=True,
            initial_bankroll=1000.0,
            dynamic_max_bet_enabled=True,
            dynamic_max_bet_pct=0.05,
            dynamic_max_bet_floor=150.0,
            dynamic_max_bet_cap=5000.0,
            aggression_enabled=True,
            aggression_min_samples=40,
            aggression_target_win_rate=0.90,
            aggression_max_boost=0.75,
            aggression_drawdown_guard=0.20,
        )

        base_size = compute_size_kelly(
            cfg,
            portfolio_value=10000.0,
            confidence=0.99,
            ask_depth=100000.0,
            price=0.10,
            resolved_trades=300,
            peak_value=10000.0,
            performance_win_rate=None,
            performance_samples=0,
        )
        boosted_size = compute_size_kelly(
            cfg,
            portfolio_value=10000.0,
            confidence=0.99,
            ask_depth=100000.0,
            price=0.10,
            resolved_trades=300,
            peak_value=10000.0,
            performance_win_rate=0.95,
            performance_samples=300,
        )
        assert boosted_size > base_size

    def test_aggression_boost_disabled_in_large_drawdown(self):
        cfg = Config(
            dry_run=True,
            initial_bankroll=1000.0,
            dynamic_max_bet_enabled=True,
            dynamic_max_bet_pct=0.05,
            dynamic_max_bet_floor=150.0,
            dynamic_max_bet_cap=5000.0,
            aggression_enabled=True,
            aggression_min_samples=40,
            aggression_target_win_rate=0.90,
            aggression_max_boost=0.75,
            aggression_drawdown_guard=0.10,
        )

        boosted = compute_size_kelly(
            cfg,
            portfolio_value=10000.0,
            confidence=0.99,
            ask_depth=100000.0,
            price=0.10,
            resolved_trades=300,
            peak_value=10000.0,
            performance_win_rate=0.95,
            performance_samples=300,
        )
        drawdown_blocked = compute_size_kelly(
            cfg,
            portfolio_value=7000.0,  # 30% DD from peak
            confidence=0.99,
            ask_depth=100000.0,
            price=0.10,
            resolved_trades=300,
            peak_value=10000.0,
            performance_win_rate=0.95,
            performance_samples=300,
        )
        assert drawdown_blocked < boosted


class TestDrawdownThrottle:
    """Drawdown-based multiplier."""

    def test_no_drawdown_full(self):
        """No drawdown -> 1.0."""
        assert drawdown_throttle(1000, 1000, 0.15, 0.30) == 1.0

    def test_below_start_full(self):
        """Drawdown below start threshold -> 1.0."""
        # 10% drawdown, start=15%
        assert drawdown_throttle(900, 1000, 0.15, 0.30) == 1.0

    def test_at_full_throttle(self):
        """At full drawdown threshold -> 0.5."""
        # 30% drawdown
        assert drawdown_throttle(700, 1000, 0.15, 0.30) == 0.5

    def test_midpoint(self):
        """Between start and full -> between 0.5 and 1.0."""
        # 22.5% drawdown (midpoint of 15%-30%)
        result = drawdown_throttle(775, 1000, 0.15, 0.30)
        assert 0.5 < result < 1.0
        assert result == pytest.approx(0.75, rel=1e-2)

    def test_beyond_full(self):
        """Beyond full drawdown -> 0.5 (not lower)."""
        assert drawdown_throttle(500, 1000, 0.15, 0.30) == 0.5


class TestKellyMultiplier:
    """Confidence -> fractional Kelly mapping."""

    def test_low_confidence_min_fraction(self):
        """Below 0.85 -> min_fraction."""
        m = kelly_multiplier_for_confidence(0.82, 0.40, 0.60, 0.75)
        assert m == 0.40

    def test_mid_confidence_base_fraction(self):
        """0.85-0.90 -> base_fraction."""
        m = kelly_multiplier_for_confidence(0.87, 0.40, 0.60, 0.75)
        assert m == 0.60

    def test_high_confidence_max_fraction(self):
        """0.95+ -> max_fraction."""
        m = kelly_multiplier_for_confidence(0.97, 0.40, 0.60, 0.75)
        assert m == 0.75

    def test_interpolation_range(self):
        """0.90-0.95 -> linear between base and max."""
        m = kelly_multiplier_for_confidence(0.925, 0.40, 0.60, 0.75)
        expected = 0.60 + 0.5 * (0.75 - 0.60)  # midpoint
        assert m == pytest.approx(expected, rel=1e-3)
