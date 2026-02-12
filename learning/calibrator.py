"""Confidence calibration: compare predicted confidence bands to actual win rates."""

from __future__ import annotations

from dataclasses import dataclass

from learning.tracker import TradeTracker
from utils.logging import get_logger

log = get_logger("calibrator")

# Minimum samples required per band before computing an offset
MIN_BAND_SAMPLES = 5

# Dampening factor to prevent over-correction
DAMPENING = 0.50

# Maximum absolute adjustment (cap)
MAX_ADJUSTMENT = 0.10

# Band definitions: label → midpoint
BAND_MIDPOINTS: dict[str, float] = {
    "0.70-0.80": 0.750,
    "0.80-0.85": 0.825,
    "0.85-0.90": 0.875,
    "0.90-0.95": 0.925,
    "0.95-1.00": 0.975,
}


@dataclass
class CalibrationOffset:
    """Calibration data for a single confidence band."""

    band: str
    predicted_midpoint: float
    actual_win_rate: float
    sample_size: int
    offset: float


class ConfidenceCalibrator:
    """Compares predicted confidence bands to actual outcomes and provides
    calibration adjustments for the confidence scoring pipeline."""

    def __init__(self, tracker: TradeTracker) -> None:
        self._tracker = tracker
        self._offsets: dict[str, CalibrationOffset] = {}

    def calibrate(self) -> dict[str, CalibrationOffset]:
        """Recompute calibration offsets from current trade data.

        Calls tracker.get_stats_by_confidence_band(), computes offset for
        bands with sufficient samples, and stores them.

        Returns the offsets dict (band label → CalibrationOffset).
        """
        band_stats = self._tracker.get_stats_by_confidence_band()
        self._offsets.clear()

        for band, stats in band_stats.items():
            midpoint = BAND_MIDPOINTS.get(band)
            if midpoint is None:
                continue

            if stats.resolved_trades < MIN_BAND_SAMPLES:
                continue

            offset = stats.win_rate - midpoint

            self._offsets[band] = CalibrationOffset(
                band=band,
                predicted_midpoint=midpoint,
                actual_win_rate=stats.win_rate,
                sample_size=stats.resolved_trades,
                offset=offset,
            )

            log.debug(
                "calibration_band",
                band=band,
                midpoint=midpoint,
                actual=round(stats.win_rate, 3),
                offset=round(offset, 3),
                samples=stats.resolved_trades,
            )

        return self._offsets

    def get_adjustment_for_confidence(self, raw_confidence: float) -> float:
        """Map a raw confidence score to its calibration adjustment.

        Returns offset * DAMPENING, capped at +/- MAX_ADJUSTMENT.
        Returns 0.0 if no calibration data for the confidence band.
        """
        band = self._confidence_to_band(raw_confidence)
        if band is None:
            return 0.0

        cal = self._offsets.get(band)
        if cal is None:
            return 0.0

        adj = cal.offset * DAMPENING
        return max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adj))

    @staticmethod
    def _confidence_to_band(confidence: float) -> str | None:
        """Map a confidence value to its band label."""
        if confidence < 0.70:
            return None
        elif confidence < 0.80:
            return "0.70-0.80"
        elif confidence < 0.85:
            return "0.80-0.85"
        elif confidence < 0.90:
            return "0.85-0.90"
        elif confidence < 0.95:
            return "0.90-0.95"
        else:
            return "0.95-1.00"
