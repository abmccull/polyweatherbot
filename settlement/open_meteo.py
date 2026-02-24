"""Open-Meteo Archive API client for international city temperature resolution."""

from __future__ import annotations

from datetime import date

import httpx

from utils.logging import get_logger

log = get_logger("open_meteo")

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoClient:
    """Free historical weather API — no auth required."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

    async def get_daily_tmax(
        self,
        latitude: float,
        longitude: float,
        for_date: date,
        timezone: str = "UTC",
    ) -> float | None:
        """Fetch daily maximum temperature in Celsius for a specific date.

        Args:
            latitude: Station latitude
            longitude: Station longitude
            for_date: The date to query
            timezone: IANA timezone used for daily aggregation boundaries

        Returns:
            Daily max temperature in Celsius, or None on error/no data.
        """
        date_str = for_date.isoformat()
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date_str,
            "end_date": date_str,
            "daily": "temperature_2m_max",
            "timezone": timezone,
        }

        try:
            resp = await self._client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            log.warning("open_meteo_http_error", status=e.response.status_code, lat=latitude, lon=longitude)
            return None
        except Exception as e:
            log.warning("open_meteo_request_error", error=str(e), lat=latitude, lon=longitude)
            return None

        try:
            daily = data.get("daily", {})
            temps = daily.get("temperature_2m_max", [])
            if temps and temps[0] is not None:
                tmax_c = float(temps[0])
                log.debug("open_meteo_tmax_fetched", lat=latitude, lon=longitude, date=date_str, tmax_c=tmax_c)
                return tmax_c
        except (ValueError, TypeError, KeyError) as e:
            log.warning("open_meteo_parse_error", error=str(e))

        return None

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
