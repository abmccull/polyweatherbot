"""Multi-factor confidence scoring for trade signals."""

from __future__ import annotations

from dataclasses import dataclass

from weather.temperature import BucketMatch, Precision


@dataclass
class ConfidenceFactors:
    """Breakdown of confidence score components."""

    base: float
    precision_bonus: float
    margin_bonus: float
    wu_lag_bonus: float
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
        bucket_match: Result from temp_hits_bucket
        precision: Temperature measurement precision
        wu_lag_confirmed: Whether WU is showing a different (lower) high
        is_peak_hours: Whether current local time is 1-5 PM
        historical_accuracy: Historical win rate from learning module (0.0-1.0)

    Returns:
        ConfidenceFactors with breakdown and total score
    """
    base = bucket_match.confidence

    # T-group precision bonus
    precision_bonus = 0.15 if precision == Precision.TENTHS else 0.0

    # Margin from boundary
    margin = bucket_match.margin
    if margin > 1.0:
        margin_bonus = 0.20
    elif margin > 0.5:
        margin_bonus = 0.10
    elif margin > 0.2:
        margin_bonus = 0.0
    elif margin < 0.1:
        margin_bonus = -0.15
    else:
        margin_bonus = -0.05

    # WU lag bonus — kept at 0.0 for backwards compat (replaced by recency_bonus)
    wu_lag_bonus = 0.0

    # Peak heating hours
    peak_hours_bonus = 0.05 if is_peak_hours else 0.0

    # Recency bonus: fresh METAR obs = WU likely hasn't caught up yet
    recency_bonus = 0.0
    if metar_age_minutes is not None:
        if metar_age_minutes <= 20:
            recency_bonus = 0.10
        elif metar_age_minutes <= 45:
            recency_bonus = 0.05

    # Historical accuracy blend (30% weight when available)
    historical_blend = 0.0
    if historical_accuracy is not None and historical_accuracy > 0:
        # Blend: adjust base toward historical rate
        hist_adj = (historical_accuracy - 0.85) * 0.30  # 85% is neutral
        historical_blend = hist_adj

    raw_total = base + precision_bonus + margin_bonus + peak_hours_bonus + recency_bonus + historical_blend + calibration_adjustment
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
