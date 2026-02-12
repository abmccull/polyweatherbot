"""NOAA NCEI Data Service v1 client for precipitation and temperature data."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date

import httpx

from utils.logging import get_logger

log = get_logger("noaa")

BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

# PRCP values are in tenths of mm; conversion factor to inches
_TENTHS_MM_TO_INCHES = 1.0 / 254.0


@dataclass
class MonthlyPrecipResult:
    """Result of a monthly precipitation query."""

    station_id: str
    year: int
    month: int
    total_inches: float
    days_with_data: int
    total_days: int


class NOAAClient:
    """Async client for NOAA NCEI Data Service (no auth required)."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

    async def get_monthly_precip(
        self, ghcnd_station: str, year: int, month: int,
    ) -> MonthlyPrecipResult | None:
        """Fetch daily precipitation summaries and compute monthly total.

        Args:
            ghcnd_station: GHCND station ID (e.g. "USW00014732")
            year: 4-digit year
            month: 1-12

        Returns:
            MonthlyPrecipResult or None on error/no data.
        """
        total_days = calendar.monthrange(year, month)[1]
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month:02d}-{total_days:02d}"

        params = {
            "dataset": "daily-summaries",
            "stations": ghcnd_station,
            "dataTypes": "PRCP",
            "startDate": start_date,
            "endDate": end_date,
            "format": "json",
        }

        try:
            resp = await self._client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            log.warning("noaa_http_error", status=e.response.status_code, station=ghcnd_station)
            return None
        except Exception as e:
            log.warning("noaa_request_error", error=str(e), station=ghcnd_station)
            return None

        if not data:
            log.debug("noaa_no_data", station=ghcnd_station, year=year, month=month)
            return None

        # Sum up PRCP values (tenths of mm)
        total_tenths_mm = 0.0
        days_with_data = 0
        for record in data:
            prcp = record.get("PRCP")
            if prcp is not None:
                try:
                    total_tenths_mm += float(prcp)
                    days_with_data += 1
                except (ValueError, TypeError):
                    continue

        total_inches = total_tenths_mm * _TENTHS_MM_TO_INCHES

        log.debug(
            "noaa_precip_fetched",
            station=ghcnd_station,
            year=year,
            month=month,
            total_inches=round(total_inches, 3),
            days_with_data=days_with_data,
        )

        return MonthlyPrecipResult(
            station_id=ghcnd_station,
            year=year,
            month=month,
            total_inches=total_inches,
            days_with_data=days_with_data,
            total_days=total_days,
        )

    async def get_daily_tmax(self, ghcnd_station: str, for_date: date) -> float | None:
        """Fetch daily maximum temperature in Celsius for a specific date.

        Args:
            ghcnd_station: GHCND station ID (e.g. "USW00014732")
            for_date: The date to query

        Returns:
            Daily max temperature in Celsius, or None on error/no data.
        """
        date_str = for_date.isoformat()
        params = {
            "dataset": "daily-summaries",
            "stations": ghcnd_station,
            "dataTypes": "TMAX",
            "startDate": date_str,
            "endDate": date_str,
            "format": "json",
        }

        try:
            resp = await self._client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            log.warning("noaa_tmax_http_error", status=e.response.status_code, station=ghcnd_station)
            return None
        except Exception as e:
            log.warning("noaa_tmax_request_error", error=str(e), station=ghcnd_station)
            return None

        if not data:
            log.debug("noaa_tmax_no_data", station=ghcnd_station, date=date_str)
            return None

        # TMAX is in tenths of °C
        for record in data:
            tmax = record.get("TMAX")
            if tmax is not None:
                try:
                    tmax_c = float(tmax) / 10.0
                    log.debug("noaa_tmax_fetched", station=ghcnd_station, date=date_str, tmax_c=tmax_c)
                    return tmax_c
                except (ValueError, TypeError):
                    continue

        return None

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
