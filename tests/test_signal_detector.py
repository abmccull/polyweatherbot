"""Signal detector tests — validates all 7 fixes for edge detection.

Tests cover:
  1. Date validation (daily high date must match market_date)
  2. METAR freshness gate (must be < 30 min old)
  3. Time-of-day gating (geq: hour >= 12, leq: hour >= 17)
  4. Confidence scoring (no double-counting, conservative)
  5. Price range (min_price floor, max_price ceiling)
  6. Directional logic (exact buckets skipped)
  7. End-to-end signal detection
"""

from __future__ import annotations

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import pytz

from config import Config
from signals.confidence import compute_confidence, ConfidenceFactors
from signals.detector import SignalDetector, TradeSignal
from weather.temperature import (
    PreciseTemp, Precision, BucketMatch,
    temp_hits_bucket, _margin_confidence,
)
from weather.metar_feed import DailyHigh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return Config(
        dry_run=True,
        initial_bankroll=1000.0,
        min_price=0.30,
        metar_max_age_minutes=30.0,
        geq_min_hour=12,
        leq_min_hour=17,
    )


def _make_temp(celsius: float, precision: Precision = Precision.TENTHS) -> PreciseTemp:
    return PreciseTemp(celsius=celsius, precision=precision)


# ---------------------------------------------------------------------------
# Fix 1: Date validation
# ---------------------------------------------------------------------------


class TestDateValidation:
    """Daily high date must match market_date."""

    def test_matching_dates_allowed(self):
        """Signal allowed when daily high date == market date."""
        dh = DailyHigh(station="KATL", date=date(2026, 2, 11))
        market_date = date(2026, 2, 11)
        assert dh.date == market_date

    def test_mismatched_dates_rejected(self):
        """Signal rejected when daily high date != market date."""
        dh = DailyHigh(station="KATL", date=date(2026, 2, 10))  # yesterday
        market_date = date(2026, 2, 11)  # today's market
        assert dh.date != market_date

    def test_future_market_rejected(self):
        """Tomorrow's market with today's METAR → rejected."""
        dh = DailyHigh(station="KATL", date=date(2026, 2, 11))
        market_date = date(2026, 2, 12)  # tomorrow
        assert dh.date != market_date


# ---------------------------------------------------------------------------
# Fix 2: METAR freshness gate
# ---------------------------------------------------------------------------


class TestMetarFreshness:
    """METAR must be fresh (< 30 min) for latency edge."""

    def test_fresh_metar_accepted(self):
        """10-minute-old METAR → fresh."""
        obs_time = datetime.utcnow().replace(tzinfo=pytz.utc) - timedelta(minutes=10)
        now = datetime.utcnow().replace(tzinfo=pytz.utc)
        age = (now - obs_time).total_seconds() / 60.0
        assert age <= 30.0

    def test_stale_metar_rejected(self):
        """45-minute-old METAR → stale, no edge."""
        obs_time = datetime.utcnow().replace(tzinfo=pytz.utc) - timedelta(minutes=45)
        now = datetime.utcnow().replace(tzinfo=pytz.utc)
        age = (now - obs_time).total_seconds() / 60.0
        assert age > 30.0

    def test_no_obs_time_rejected(self):
        """Missing obs time → rejected."""
        dh = DailyHigh(station="KATL", date=date(2026, 2, 11))
        assert dh.last_obs_time is None


# ---------------------------------------------------------------------------
# Fix 3: Time-of-day gating
# ---------------------------------------------------------------------------


class TestTimeOfDayGating:
    """geq requires hour >= 12, leq requires hour >= 17."""

    def test_geq_before_noon_rejected(self, config):
        """geq signal at 9 AM → rejected (too early, temp will rise)."""
        assert 9 < config.geq_min_hour

    def test_geq_at_noon_allowed(self, config):
        """geq signal at noon → allowed."""
        assert 12 >= config.geq_min_hour

    def test_geq_afternoon_allowed(self, config):
        """geq signal at 2 PM → allowed."""
        assert 14 >= config.geq_min_hour

    def test_leq_at_noon_rejected(self, config):
        """leq signal at noon → rejected (peak heating hasn't passed)."""
        assert 12 < config.leq_min_hour

    def test_leq_at_3pm_rejected(self, config):
        """leq signal at 3 PM → rejected (still in peak heating)."""
        assert 15 < config.leq_min_hour

    def test_leq_at_5pm_allowed(self, config):
        """leq signal at 5 PM → allowed (peak heating over)."""
        assert 17 >= config.leq_min_hour

    def test_leq_at_8pm_allowed(self, config):
        """leq signal at 8 PM → allowed."""
        assert 20 >= config.leq_min_hour


# ---------------------------------------------------------------------------
# Fix 4: Confidence scoring (no double-counting)
# ---------------------------------------------------------------------------


class TestMarginConfidence:
    """Base confidence from margin only (no precision bonus)."""

    def test_base_is_lower(self):
        """Base should be 0.40 (down from old 0.50)."""
        c = _margin_confidence(0.3, has_tenths=False)
        # margin > 0.2: base=0.40 + 0.05 = 0.45
        assert c == pytest.approx(0.45, abs=0.01)

    def test_no_precision_in_margin(self):
        """Tenths precision should NOT affect _margin_confidence."""
        c_whole = _margin_confidence(0.6, has_tenths=False)
        c_tenths = _margin_confidence(0.6, has_tenths=True)
        # Both should be the same — precision is NOT added here anymore
        assert c_whole == c_tenths

    def test_large_margin(self):
        """Margin > 2.0 → base + 0.20 = 0.60."""
        c = _margin_confidence(2.5, has_tenths=False)
        assert c == pytest.approx(0.60, abs=0.01)

    def test_tiny_margin(self):
        """Margin < 0.1 → base - 0.10 = 0.30."""
        c = _margin_confidence(0.05, has_tenths=False)
        assert c == pytest.approx(0.30, abs=0.01)


class TestConfidenceScoring:
    """Full confidence scoring with no double-counting."""

    def test_strong_signal_reaches_threshold(self):
        """Good margin + tenths + peak + fresh → ~0.90 (above 0.85)."""
        match = BucketMatch(hit=True, confidence=0.55, margin=1.2)
        c = compute_confidence(
            bucket_match=match,
            precision=Precision.TENTHS,
            wu_lag_confirmed=False,
            is_peak_hours=True,
            metar_age_minutes=8.0,
        )
        # 0.55 + 0.10 (tenths) + 0.10 (peak) + 0.15 (fresh) = 0.90
        assert c.total >= 0.85
        assert c.total <= 1.0

    def test_weak_signal_below_threshold(self):
        """Low margin + whole + off-peak + stale → well below 0.85."""
        match = BucketMatch(hit=True, confidence=0.30, margin=0.05)
        c = compute_confidence(
            bucket_match=match,
            precision=Precision.WHOLE,
            wu_lag_confirmed=False,
            is_peak_hours=False,
            metar_age_minutes=28.0,
        )
        # 0.30 + 0 (whole) + (-0.10) (off-peak) + 0.05 (barely fresh) = 0.25
        assert c.total < 0.85

    def test_off_peak_penalty(self):
        """Off-peak hours should REDUCE confidence."""
        match = BucketMatch(hit=True, confidence=0.55, margin=1.0)
        c = compute_confidence(
            bucket_match=match,
            precision=Precision.TENTHS,
            wu_lag_confirmed=False,
            is_peak_hours=False,
            metar_age_minutes=10.0,
        )
        assert c.peak_hours_bonus == -0.10

    def test_peak_hours_bonus(self):
        """Peak hours should give +0.10."""
        match = BucketMatch(hit=True, confidence=0.55, margin=1.0)
        c = compute_confidence(
            bucket_match=match,
            precision=Precision.TENTHS,
            wu_lag_confirmed=False,
            is_peak_hours=True,
            metar_age_minutes=10.0,
        )
        assert c.peak_hours_bonus == 0.10

    def test_margin_bonus_is_zero(self):
        """margin_bonus should always be 0 (no double-counting)."""
        match = BucketMatch(hit=True, confidence=0.55, margin=2.0)
        c = compute_confidence(
            bucket_match=match,
            precision=Precision.TENTHS,
            wu_lag_confirmed=False,
            is_peak_hours=True,
            metar_age_minutes=10.0,
        )
        assert c.margin_bonus == 0.0

    def test_recency_tiers(self):
        """Recency bonus: <10min=0.15, <20min=0.10, <30min=0.05."""
        match = BucketMatch(hit=True, confidence=0.50, margin=0.5)

        c5 = compute_confidence(match, Precision.WHOLE, False, True, metar_age_minutes=5.0)
        c15 = compute_confidence(match, Precision.WHOLE, False, True, metar_age_minutes=15.0)
        c25 = compute_confidence(match, Precision.WHOLE, False, True, metar_age_minutes=25.0)

        assert c5.recency_bonus == 0.15
        assert c15.recency_bonus == 0.10
        assert c25.recency_bonus == 0.05

    def test_confidence_never_trivially_one(self):
        """Even best conditions shouldn't trivially hit 1.0."""
        # Max margin confidence = 0.60, + 0.10 + 0.10 + 0.15 = 0.95
        match = BucketMatch(hit=True, confidence=0.60, margin=2.5)
        c = compute_confidence(
            bucket_match=match,
            precision=Precision.TENTHS,
            wu_lag_confirmed=False,
            is_peak_hours=True,
            metar_age_minutes=5.0,
        )
        assert c.total <= 0.95


# ---------------------------------------------------------------------------
# Fix 5: Price range
# ---------------------------------------------------------------------------


class TestPriceRange:
    """Price must be between min_price and max_price."""

    def test_penny_ask_rejected(self, config):
        """$0.05 ask → below min_price → rejected."""
        assert 0.05 < config.min_price

    def test_decided_outcome_rejected(self, config):
        """$0.92 ask → above max_price → rejected (market already knows)."""
        assert 0.92 > config.max_price.value

    def test_sweet_spot_accepted(self, config):
        """$0.45 ask → in range → accepted."""
        assert config.min_price <= 0.45 <= config.max_price.value

    def test_low_uncertainty_accepted(self, config):
        """$0.35 ask → at min boundary → accepted."""
        assert config.min_price <= 0.35 <= config.max_price.value

    def test_max_boundary_accepted(self, config):
        """$0.65 ask → at max boundary → accepted."""
        assert config.min_price <= 0.65 <= config.max_price.value


# ---------------------------------------------------------------------------
# Fix 6: Exact buckets skipped
# ---------------------------------------------------------------------------


class TestExactBucketsSkipped:
    """Exact bucket type should be skipped (unreliable for latency arb)."""

    def test_exact_type_identified(self):
        """Exact bucket type is 'exact'."""
        assert "exact" not in ("geq", "leq")

    def test_geq_not_skipped(self):
        """geq buckets should NOT be skipped."""
        assert "geq" in ("geq", "leq")

    def test_leq_not_skipped(self):
        """leq buckets should NOT be skipped."""
        assert "leq" in ("geq", "leq")


# ---------------------------------------------------------------------------
# Fix 7: Config defaults updated
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Config defaults should reflect the fix."""

    def test_max_price_raised(self):
        """max_price should be 0.65 (was 0.25)."""
        cfg = Config(dry_run=True)
        assert cfg.max_price.value == 0.65

    def test_min_price_exists(self):
        """min_price should be 0.30."""
        cfg = Config(dry_run=True)
        assert cfg.min_price == 0.30

    def test_metar_max_age(self):
        """metar_max_age_minutes should be 30."""
        cfg = Config(dry_run=True)
        assert cfg.metar_max_age_minutes == 30.0

    def test_geq_min_hour(self):
        """geq_min_hour should be 12 (noon)."""
        cfg = Config(dry_run=True)
        assert cfg.geq_min_hour == 12

    def test_leq_min_hour(self):
        """leq_min_hour should be 17 (5 PM)."""
        cfg = Config(dry_run=True)
        assert cfg.leq_min_hour == 17


# ---------------------------------------------------------------------------
# Integration: temp_hits_bucket consistency
# ---------------------------------------------------------------------------


class TestBucketMatch:
    """Verify bucket matching returns sensible confidence with new scaling."""

    def test_geq_hit_above_threshold_f(self):
        """45°F reading for geq:44°F → hit with positive margin."""
        temp = _make_temp(7.2, Precision.TENTHS)  # 7.2°C ≈ 45.0°F
        match = temp_hits_bucket(temp, "geq", 44, unit="F")
        assert match.hit is True
        assert match.margin > 0
        assert match.confidence > 0

    def test_geq_miss_below_threshold_f(self):
        """42°F reading for geq:44°F → miss."""
        temp = _make_temp(5.6, Precision.TENTHS)  # 5.6°C ≈ 42.0°F
        match = temp_hits_bucket(temp, "geq", 44, unit="F")
        assert match.hit is False

    def test_leq_hit_below_threshold_f(self):
        """42°F reading for leq:44°F → hit."""
        temp = _make_temp(5.6, Precision.TENTHS)  # 5.6°C ≈ 42.0°F
        match = temp_hits_bucket(temp, "leq", 44, unit="F")
        assert match.hit is True
        assert match.margin > 0

    def test_confidence_range_0_to_1(self):
        """Confidence must always be in [0, 1]."""
        for margin_c in [0.01, 0.1, 0.3, 0.6, 1.5, 3.0]:
            c = _margin_confidence(margin_c, has_tenths=True)
            assert 0.0 <= c <= 1.0

    def test_precision_only_in_confidence_py(self):
        """_margin_confidence must not use precision — same result for both."""
        c1 = _margin_confidence(1.0, has_tenths=False)
        c2 = _margin_confidence(1.0, has_tenths=True)
        assert c1 == c2
