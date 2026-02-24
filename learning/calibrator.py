"""Confidence calibration for mapping heuristic scores to win probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from learning.tracker import TradeTracker
from utils.logging import get_logger

log = get_logger("calibrator")

# Minimum samples required per confidence band before computing an offset.
MIN_BAND_SAMPLES = 5

# Dampening factor to prevent over-correction.
DAMPENING = 0.50

# Maximum absolute adjustment (cap).
MAX_ADJUSTMENT = 0.10

# Minimum samples for fitting a logistic probability model.
MIN_MODEL_SAMPLES = 20
MODEL_LOOKBACK_DAYS = 120
MODEL_MAX_ITERS = 400
MODEL_LEARNING_RATE = 0.25
MODEL_L2_REG = 0.02

# Band definitions: label -> midpoint.
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


@dataclass
class CalibrationModel:
    """Fitted Platt-style logistic model diagnostics."""

    slope: float
    intercept: float
    sample_size: int
    log_loss: float
    brier_score: float
    expected_calibration_error: float


class ConfidenceCalibrator:
    """Produces both band offsets and calibrated win probabilities."""

    def __init__(self, tracker: TradeTracker) -> None:
        self._tracker = tracker
        self._offsets: dict[str, CalibrationOffset] = {}
        self._model: CalibrationModel | None = None

    def calibrate(self) -> dict[str, CalibrationOffset]:
        """Recompute calibration offsets and logistic probability model."""
        band_stats = self._tracker.get_stats_by_confidence_band()
        self._offsets.clear()

        for band, stats in band_stats.items():
            midpoint = BAND_MIDPOINTS.get(band)
            if midpoint is None or stats.resolved_trades < MIN_BAND_SAMPLES:
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

        self._fit_probability_model()
        return self._offsets

    def get_adjustment_for_confidence(self, raw_confidence: float) -> float:
        """Map a raw confidence score to an additive calibration adjustment."""
        band = self._confidence_to_band(raw_confidence)
        if band is None:
            return 0.0

        cal = self._offsets.get(band)
        if cal is None:
            return 0.0

        adj = cal.offset * DAMPENING
        return max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adj))

    def get_calibrated_probability(self, confidence: float) -> float | None:
        """Return calibrated win probability for a confidence score.

        Preference order:
        1) fitted logistic model, if available
        2) band-offset-adjusted confidence
        3) None when no calibration data exists yet
        """
        clipped = max(0.0, min(1.0, confidence))
        if self._model is not None:
            return _sigmoid(self._model.slope * clipped + self._model.intercept)

        if self._offsets:
            adj = self.get_adjustment_for_confidence(clipped)
            return max(0.0, min(1.0, clipped + adj))

        return None

    def get_model_diagnostics(self) -> dict[str, float | int] | None:
        """Return latest logistic model diagnostics for observability."""
        if self._model is None:
            return None
        return {
            "sample_size": self._model.sample_size,
            "slope": round(self._model.slope, 4),
            "intercept": round(self._model.intercept, 4),
            "log_loss": round(self._model.log_loss, 5),
            "brier_score": round(self._model.brier_score, 5),
            "ece": round(self._model.expected_calibration_error, 5),
        }

    def _fit_probability_model(self) -> None:
        """Fit a one-feature logistic model p(y=1|confidence)."""
        points = self._tracker.get_confidence_outcomes(lookback_days=MODEL_LOOKBACK_DAYS)
        if len(points) < MIN_MODEL_SAMPLES:
            self._model = None
            return

        n = len(points)
        wins = sum(y for _, y in points)

        if wins == 0 or wins == n:
            # Degenerate case: no class variance. Fit constant model with Laplace smoothing.
            p = (wins + 1.0) / (n + 2.0)
            slope = 0.0
            intercept = _logit(p)
        else:
            # Start near identity mapping around 0.5.
            slope = 5.0
            intercept = -2.5

            for _ in range(MODEL_MAX_ITERS):
                grad_slope = 0.0
                grad_intercept = 0.0

                for x, y in points:
                    p = _sigmoid(slope * x + intercept)
                    err = p - y
                    grad_slope += err * x
                    grad_intercept += err

                inv_n = 1.0 / n
                grad_slope = grad_slope * inv_n + MODEL_L2_REG * slope
                grad_intercept = grad_intercept * inv_n + MODEL_L2_REG * intercept

                slope -= MODEL_LEARNING_RATE * grad_slope
                intercept -= MODEL_LEARNING_RATE * grad_intercept

                if abs(grad_slope) + abs(grad_intercept) < 1e-7:
                    break

        log_loss, brier, ece = _calibration_metrics(points, slope, intercept)
        self._model = CalibrationModel(
            slope=slope,
            intercept=intercept,
            sample_size=n,
            log_loss=log_loss,
            brier_score=brier,
            expected_calibration_error=ece,
        )
        log.info(
            "calibration_model_fit",
            sample_size=n,
            slope=round(slope, 4),
            intercept=round(intercept, 4),
            log_loss=round(log_loss, 5),
            brier=round(brier, 5),
            ece=round(ece, 5),
        )

    @staticmethod
    def _confidence_to_band(confidence: float) -> str | None:
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


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _calibration_metrics(points: list[tuple[float, int]], slope: float, intercept: float) -> tuple[float, float, float]:
    eps = 1e-12
    n = len(points)
    log_loss = 0.0
    brier = 0.0

    bins: dict[int, list[tuple[float, int]]] = {}
    for x, y in points:
        p = _sigmoid(slope * x + intercept)
        p = max(eps, min(1.0 - eps, p))
        log_loss += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
        brier += (p - y) ** 2

        idx = min(9, int(p * 10))
        bins.setdefault(idx, []).append((p, y))

    ece = 0.0
    for values in bins.values():
        if not values:
            continue
        avg_p = sum(v[0] for v in values) / len(values)
        avg_y = sum(v[1] for v in values) / len(values)
        ece += abs(avg_p - avg_y) * (len(values) / n)

    return log_loss / n, brier / n, ece
