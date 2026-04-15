from __future__ import annotations

from datetime import datetime, timezone

from utils.redemption_schedule import is_within_quiet_hours, next_redemption_run_utc


def test_is_within_quiet_hours_wraps_midnight():
    # 2026-01-15 07:30 UTC == 00:30 America/Denver (MST).
    now = datetime(2026, 1, 15, 7, 30, tzinfo=timezone.utc)
    assert is_within_quiet_hours(
        now,
        timezone_name="America/Denver",
        quiet_start_hour=23,
        quiet_end_hour=6,
    )

    # 2026-01-15 15:00 UTC == 08:00 America/Denver (MST).
    day_time = datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc)
    assert not is_within_quiet_hours(
        day_time,
        timezone_name="America/Denver",
        quiet_start_hour=23,
        quiet_end_hour=6,
    )


def test_next_redemption_run_skips_quiet_window():
    # 2026-01-15 05:20 UTC == 22:20 America/Denver.
    now = datetime(2026, 1, 15, 5, 20, tzinfo=timezone.utc)
    next_run = next_redemption_run_utc(
        now,
        interval_seconds=3600,
        quiet_hours_enabled=True,
        timezone_name="America/Denver",
        quiet_start_hour=23,
        quiet_end_hour=6,
    )
    # 13:00 UTC == 06:00 America/Denver, first non-quiet hourly slot.
    assert next_run == datetime(2026, 1, 15, 13, 0, tzinfo=timezone.utc)


def test_next_redemption_run_no_quiet_hours():
    now = datetime(2026, 1, 15, 15, 37, tzinfo=timezone.utc)
    next_run = next_redemption_run_utc(
        now,
        interval_seconds=3600,
        quiet_hours_enabled=False,
        timezone_name="America/Denver",
        quiet_start_hour=23,
        quiet_end_hour=6,
    )
    assert next_run == datetime(2026, 1, 15, 16, 0, tzinfo=timezone.utc)


def test_equal_start_end_disables_quiet_window():
    now = datetime(2026, 1, 15, 7, 30, tzinfo=timezone.utc)
    assert not is_within_quiet_hours(
        now,
        timezone_name="America/Denver",
        quiet_start_hour=23,
        quiet_end_hour=23,
    )
