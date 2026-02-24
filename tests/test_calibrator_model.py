"""Tests for confidence probability calibration model."""

from __future__ import annotations

from learning.calibrator import ConfidenceCalibrator
from learning.tracker import TradeStats


class _TrackerStub:
    def __init__(self, points: list[tuple[float, int]], band_stats: dict[str, TradeStats]) -> None:
        self._points = points
        self._band_stats = band_stats

    def get_confidence_outcomes(self, lookback_days: int | None = None) -> list[tuple[float, int]]:
        return list(self._points)

    def get_stats_by_confidence_band(self) -> dict[str, TradeStats]:
        return dict(self._band_stats)


def _stats(win_rate: float, resolved: int) -> TradeStats:
    wins = int(round(win_rate * resolved))
    losses = resolved - wins
    return TradeStats(
        total_trades=resolved,
        resolved_trades=resolved,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=0.0,
        avg_roi=0.0,
    )


def test_calibrator_fits_logistic_model_when_enough_data():
    # Monotonic-ish calibration set: higher confidence should map to higher win probability.
    points = [
        (0.72, 0), (0.74, 0), (0.75, 0), (0.76, 0), (0.77, 0),
        (0.78, 0), (0.79, 0), (0.80, 0), (0.81, 1), (0.82, 0),
        (0.83, 1), (0.84, 1), (0.85, 1), (0.86, 1), (0.87, 1),
        (0.88, 1), (0.89, 1), (0.90, 1), (0.91, 1), (0.92, 1),
        (0.93, 1), (0.94, 1), (0.95, 1), (0.96, 1), (0.97, 1),
    ]
    tracker = _TrackerStub(
        points=points,
        band_stats={
            "0.70-0.80": _stats(0.35, 10),
            "0.80-0.85": _stats(0.60, 10),
            "0.85-0.90": _stats(0.80, 10),
            "0.90-0.95": _stats(0.95, 10),
            "0.95-1.00": _stats(0.98, 10),
        },
    )
    calibrator = ConfidenceCalibrator(tracker)
    calibrator.calibrate()

    diagnostics = calibrator.get_model_diagnostics()
    assert diagnostics is not None
    assert diagnostics["sample_size"] == len(points)

    low = calibrator.get_calibrated_probability(0.75)
    high = calibrator.get_calibrated_probability(0.95)
    assert low is not None
    assert high is not None
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_calibrator_falls_back_to_band_offsets_without_model():
    tracker = _TrackerStub(
        points=[(0.75, 1), (0.76, 0), (0.77, 1)],  # insufficient for logistic fit
        band_stats={
            "0.70-0.80": _stats(0.90, 10),
        },
    )
    calibrator = ConfidenceCalibrator(tracker)
    calibrator.calibrate()

    assert calibrator.get_model_diagnostics() is None
    p = calibrator.get_calibrated_probability(0.76)
    assert p is not None
    assert p > 0.76


def test_calibrator_returns_none_when_no_calibration_data():
    tracker = _TrackerStub(points=[], band_stats={})
    calibrator = ConfidenceCalibrator(tracker)
    calibrator.calibrate()

    assert calibrator.get_model_diagnostics() is None
    assert calibrator.get_calibrated_probability(0.88) is None
