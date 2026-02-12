"""Precision temperature engine: T-group parsing, F↔C conversion, bucket matching."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum


class Precision(Enum):
    TENTHS = "tenths"  # From T-group in METAR remarks (0.1°C)
    WHOLE = "whole"  # From main body temp field (1°C)


@dataclass
class PreciseTemp:
    """A temperature reading with known precision."""

    celsius: float
    precision: Precision

    @property
    def fahrenheit(self) -> float:
        return self.celsius * 9.0 / 5.0 + 32.0

    @property
    def has_tenths(self) -> bool:
        return self.precision == Precision.TENTHS

    @property
    def wu_display_c(self) -> int:
        """Predict what WU will display (whole-degree C, standard rounding)."""
        return round(self.celsius)

    @property
    def wu_display_f(self) -> int:
        """Predict what WU will display in whole-degree Fahrenheit."""
        return round(self.fahrenheit)


@dataclass
class BucketMatch:
    """Result of checking a temperature against a market bucket."""

    hit: bool
    confidence: float
    margin: float  # distance from boundary (positive = safely inside)


def parse_t_group(raw_metar: str) -> PreciseTemp | None:
    """Extract tenth-degree temperature from METAR T-group in remarks.

    T-group format: T{sign1}{temp3}{sign2}{dewpt3}
    sign: 0 = positive, 1 = negative
    temp3/dewpt3: temperature in tenths of °C (e.g., 0156 = 15.6°C)

    Examples:
        T01560055 → temp = +15.6°C, dewpt = +5.5°C
        T10121008 → temp = -1.2°C, dewpt = -0.8°C
    """
    match = re.search(r'\bT(\d)(\d{3})(\d)(\d{3})\b', raw_metar)
    if not match:
        return None

    sign = -1 if match.group(1) == '1' else 1
    temp_tenths = int(match.group(2))
    temp_c = sign * temp_tenths / 10.0

    return PreciseTemp(celsius=temp_c, precision=Precision.TENTHS)


def parse_metar_temp(raw_metar: str) -> PreciseTemp | None:
    """Parse temperature from METAR, preferring T-group precision.

    Falls back to main body temp/dewpoint field (e.g., '15/10' or 'M02/M05').
    """
    # Try T-group first (tenth-degree precision)
    t_group = parse_t_group(raw_metar)
    if t_group is not None:
        return t_group

    # Fallback: main body temp field
    match = re.search(r'\b(M?\d{2})/(M?\d{2})\b', raw_metar)
    if not match:
        return None

    temp_str = match.group(1)
    sign = -1 if temp_str.startswith('M') else 1
    temp_c = sign * int(temp_str.lstrip('M'))

    return PreciseTemp(celsius=float(temp_c), precision=Precision.WHOLE)


def temp_hits_bucket(
    temp: PreciseTemp,
    bucket_type: str,
    bucket_value: int,
    unit: str = "C",
) -> BucketMatch:
    """Check if a temperature reading hits a market bucket.

    Args:
        temp: The precise temperature reading
        bucket_type: "exact", "geq" (>=), or "leq" (<=)
        bucket_value: The bucket boundary in the market's unit
        unit: "C" or "F" — which unit the market uses

    Returns:
        BucketMatch with hit status, confidence, and margin.
    """
    if unit == "F":
        wu_display = temp.wu_display_f
        # For margin calculation in F, convert to C-equivalent for confidence
        # so thresholds remain consistent (°F margins are ~1.8x wider)
        actual = temp.fahrenheit
        margin_scale = 5.0 / 9.0  # normalize F margin to C-equivalent
    else:
        wu_display = temp.wu_display_c
        actual = temp.celsius
        margin_scale = 1.0

    if bucket_type == "geq":
        hit = wu_display >= bucket_value
        margin_raw = actual - (bucket_value - 0.5)
        margin = margin_raw * margin_scale
        confidence = _margin_confidence(margin, temp.has_tenths) if hit else 0.0

    elif bucket_type == "leq":
        hit = wu_display <= bucket_value
        margin_raw = (bucket_value + 0.5) - actual
        margin = margin_raw * margin_scale
        confidence = _margin_confidence(margin, temp.has_tenths) if hit else 0.0

    elif bucket_type == "exact":
        hit = wu_display == bucket_value
        margin_raw = 0.5 - abs(actual - bucket_value)
        margin = margin_raw * margin_scale
        confidence = _margin_confidence(margin, temp.has_tenths) if hit else 0.0

    else:
        return BucketMatch(hit=False, confidence=0.0, margin=0.0)

    return BucketMatch(hit=hit, confidence=confidence, margin=margin)


def _margin_confidence(margin: float, has_tenths: bool) -> float:
    """Compute confidence based on margin from boundary and precision.

    Margin = distance from the rounding boundary, normalized to °C-equivalent.
    Positive margin means safely inside the bucket.
    """
    base = 0.50

    # Precision bonus
    if has_tenths:
        base += 0.15

    # Margin-based adjustments
    if margin > 1.0:
        base += 0.25
    elif margin > 0.5:
        base += 0.15
    elif margin > 0.2:
        base += 0.05
    elif margin < 0.1:
        base -= 0.15

    return max(0.0, min(1.0, base))


def c_to_f(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9.0 / 5.0 + 32.0


def f_to_c(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32.0) * 5.0 / 9.0
