"""Multi-factor confidence scoring for trade signals.

Confidence components (no double-counting):
  - base: from _margin_confidence() in temperature.py (margin-only, 0.30-0.60)
  - precision_bonus: T-group tenths precision (single source, +0.10)
  - peak_hours_bonus: during peak heating +0.10, off-peak -0.10
  - recency_bonus: fresh METAR = WU hasn't caught up (+0.05 to +0.15)
  - historical_blend: learning module adjustment
  - calibration_adjustment: calibrator adjustment

A strong signal (margin>1, tenths, peak, fresh<10min) → ~0.55+0.10+0.10+0.15 = 0.90
A weak signal (margin<0.2, whole, off-peak, stale) → ~0.30+0+(-0.10)+0 = 0.20
Threshold is 0.85, so you need strong evidence across multiple factors.
"""

from __future__ import annotations

from dataclasses import dataclass

from weather.temperature import BucketMatch, Precision


@dataclass
class ConfidenceFactors:
    """Breakdown of confidence score components."""

    base: float
    precision_bonus: float
    margin_bonus: float  # kept at 0.0 — margin is in base now
    wu_lag_bonus: float  # deprecated, always 0.0
    peak_hours_bonus: float
    recency_bonus: float
    historical_blend: float
    calibration_adjustment: float
    total: float


def compute_confidence(
    bucket_match: BucketMatch,
    precision: Precision,
    wu_lag_confirmed: bool,
    is_peak_hours: bool,
    historical_accuracy: float | None = None,
    calibration_adjustment: float = 0.0,
    metar_age_minutes: float | None = None,
) -> ConfidenceFactors:
    """Compute multi-factor confidence score.

    Args:
        bucket_match: Result from temp_hits_bucket (base includes margin)
        precision: Temperature measurement precision
        wu_lag_confirmed: Deprecated — kept for API compat
        is_peak_hours: Whether current local time is 1-5 PM
        historical_accuracy: Historical win rate from learning module (0.0-1.0)
        calibration_adjustment: From calibrator module
        metar_age_minutes: Age of METAR observation in minutes

    Returns:
        ConfidenceFactors with breakdown and total score
    """
    # Base already incorporates margin (from _margin_confidence in temperature.py)
    base = bucket_match.confidence

    # T-group precision bonus (single source — NOT in _margin_confidence)
    precision_bonus = 0.10 if precision == Precision.TENTHS else 0.0

    # Margin bonus: REMOVED — already baked into base from _margin_confidence()
    margin_bonus = 0.0

    # WU lag bonus — deprecated (replaced by recency_bonus)
    wu_lag_bonus = 0.0

    # Peak heating hours: strong factor for edge quality
    # During peak: temp is actively changing, METAR edge is highest
    # Off-peak: temp is stable, market likely already reflects reality
    if is_peak_hours:
        peak_hours_bonus = 0.10
    else:
        peak_hours_bonus = -0.10

    # Recency bonus: fresh METAR = WU likely hasn't caught up yet
    # This is the CORE edge indicator for latency arbitrage
    recency_bonus = 0.0
    if metar_age_minutes is not None:
        if metar_age_minutes <= 10:
            recency_bonus = 0.15
        elif metar_age_minutes <= 20:
            recency_bonus = 0.10
        elif metar_age_minutes <= 30:
            recency_bonus = 0.05

    # Historical accuracy blend (30% weight when available)
    historical_blend = 0.0
    if historical_accuracy is not None and historical_accuracy > 0:
        hist_adj = (historical_accuracy - 0.85) * 0.30  # 85% is neutral
        historical_blend = hist_adj

    raw_total = (
        base
        + precision_bonus
        + margin_bonus
        + peak_hours_bonus
        + recency_bonus
        + historical_blend
        + calibration_adjustment
    )
    total = max(0.0, min(1.0, raw_total))

    return ConfidenceFactors(
        base=base,
        precision_bonus=precision_bonus,
        margin_bonus=margin_bonus,
        wu_lag_bonus=wu_lag_bonus,
        peak_hours_bonus=peak_hours_bonus,
        recency_bonus=recency_bonus,
        historical_blend=historical_blend,
        calibration_adjustment=calibration_adjustment,
        total=total,
    )
