"""METAR precipitation parsing and monthly accumulation tracking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, date


@dataclass
class PrecipReading:
    """Precipitation amounts extracted from a single METAR remark group."""

    inches_1h: float | None = None   # P-group (hourly precip)
    inches_6h: float | None = None   # 6-group (3/6-hour precip)
    inches_24h: float | None = None  # 7-group (24-hour precip)
    raw: str = ""


@dataclass
class MonthlyPrecipAccumulator:
    """Running monthly precipitation total for a station."""

    station: str
    month: str  # "2026-02"
    total_inches: float = 0.0
    last_24h_value: float | None = None  # last seen 7-group value to detect resets
    last_24h_obs_time: datetime | None = None
    readings: list[PrecipReading] = field(default_factory=list)

    def update(self, reading: PrecipReading, obs_time: datetime) -> None:
        """Update the accumulator with a new reading.

        Uses the 7-group (24-hour accumulation) to track daily totals.
        Detects resets (value drops below previous) to accumulate day-by-day.
        Falls back to 1-hour P-group if no 7-group is available.
        """
        self.readings.append(reading)

        if reading.inches_24h is not None:
            if self.last_24h_value is not None:
                if reading.inches_24h < self.last_24h_value:
                    # Reset detected: the previous value was the day's total.
                    # It was already accumulated when we saw it grow,
                    # so just update the baseline to the new post-reset value.
                    self.total_inches += reading.inches_24h
                else:
                    # Still same accumulation period — add the delta
                    delta = reading.inches_24h - self.last_24h_value
                    if delta > 0:
                        self.total_inches += delta
            else:
                # First reading this month — treat current 24h value as starting point
                self.total_inches += reading.inches_24h

            self.last_24h_value = reading.inches_24h
            self.last_24h_obs_time = obs_time

        elif reading.inches_1h is not None and reading.inches_1h > 0:
            # Fallback: if we only have P-group, add hourly amount
            # (only if we have no 7-group tracking active)
            if self.last_24h_value is None:
                self.total_inches += reading.inches_1h


def parse_precip_remarks(raw_metar: str) -> PrecipReading | None:
    """Parse precipitation remark groups from raw METAR text.

    Groups:
        P0013  → 0.13 inches in past hour (1-hour precip, P-group)
        60009  → 0.09 inches in past 3/6 hours (6-group)
        70045  → 0.45 inches in past 24 hours (7-group)

    Returns PrecipReading or None if no precip groups found.
    """
    inches_1h = None
    inches_6h = None
    inches_24h = None

    # P-group: P followed by 4 digits (hundredths of inch)
    match = re.search(r'\bP(\d{4})\b', raw_metar)
    if match:
        inches_1h = int(match.group(1)) / 100.0

    # 6-group: starts with 6, followed by 4 digits
    match = re.search(r'\b6(\d{4})\b', raw_metar)
    if match:
        inches_6h = int(match.group(1)) / 100.0

    # 7-group: starts with 7, followed by 4 digits
    match = re.search(r'\b7(\d{4})\b', raw_metar)
    if match:
        inches_24h = int(match.group(1)) / 100.0

    if inches_1h is None and inches_6h is None and inches_24h is None:
        return None

    return PrecipReading(
        inches_1h=inches_1h,
        inches_6h=inches_6h,
        inches_24h=inches_24h,
        raw=raw_metar,
    )
