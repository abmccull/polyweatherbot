"""Helpers for redemption scheduling and local quiet-hour windows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytz


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_timezone(timezone_name: str):
    try:
        return pytz.timezone(timezone_name)
    except Exception:
        return pytz.timezone("America/Denver")


def is_within_quiet_hours(
    now_utc: datetime,
    *,
    timezone_name: str,
    quiet_start_hour: int,
    quiet_end_hour: int,
) -> bool:
    """Return true if local hour is inside the configured quiet window.

    Quiet window semantics:
    - If start < end, window is same-day interval [start, end).
    - If start > end, window wraps midnight (e.g. 23 -> 6).
    - If start == end, quiet hours are disabled.
    """
    start = int(quiet_start_hour) % 24
    end = int(quiet_end_hour) % 24
    if start == end:
        return False

    local_now = _as_utc(now_utc).astimezone(_resolve_timezone(timezone_name))
    hour = local_now.hour

    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def next_redemption_run_utc(
    now_utc: datetime,
    *,
    interval_seconds: int,
    quiet_hours_enabled: bool,
    timezone_name: str,
    quiet_start_hour: int,
    quiet_end_hour: int,
) -> datetime:
    """Compute the next UTC execution time for redemption checks."""
    now = _as_utc(now_utc)
    step = max(60, int(interval_seconds))

    next_epoch = ((int(now.timestamp()) // step) + 1) * step
    candidate = datetime.fromtimestamp(next_epoch, tz=timezone.utc)

    if not quiet_hours_enabled:
        return candidate

    max_steps = max(1, (24 * 3600) // step + 2)
    for _ in range(max_steps):
        if not is_within_quiet_hours(
            candidate,
            timezone_name=timezone_name,
            quiet_start_hour=quiet_start_hour,
            quiet_end_hour=quiet_end_hour,
        ):
            return candidate
        candidate += timedelta(seconds=step)

    # Fallback safety: never loop forever on malformed configs.
    return candidate
