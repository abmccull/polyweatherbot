"""Timezone handling, market-local time, and peak heating detection."""

from __future__ import annotations

from datetime import datetime, date, time

import pytz


def utc_now() -> datetime:
    """Current time in UTC."""
    return datetime.now(pytz.utc)


def to_local(dt: datetime, timezone: str) -> datetime:
    """Convert a UTC datetime to station-local time."""
    tz = pytz.timezone(timezone)
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(tz)


def to_utc(dt: datetime, timezone: str) -> datetime:
    """Convert a local datetime to UTC."""
    tz = pytz.timezone(timezone)
    if dt.tzinfo is None:
        dt = tz.localize(dt)
    return dt.astimezone(pytz.utc)


def local_date(timezone: str) -> date:
    """Current date in the given timezone."""
    return to_local(utc_now(), timezone).date()


def local_hour(timezone: str) -> int:
    """Current hour (0-23) in the given timezone."""
    return to_local(utc_now(), timezone).hour


def is_peak_heating(timezone: str) -> bool:
    """Is current local time within peak heating hours (13:00-17:00)?"""
    hour = local_hour(timezone)
    return 13 <= hour <= 17


def local_midnight_utc(timezone: str, for_date: date | None = None) -> datetime:
    """Return UTC time of local midnight for the given date."""
    tz = pytz.timezone(timezone)
    if for_date is None:
        for_date = local_date(timezone)
    local_dt = tz.localize(datetime.combine(for_date, time.min))
    return local_dt.astimezone(pytz.utc)
