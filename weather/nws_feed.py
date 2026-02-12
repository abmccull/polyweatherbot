"""TERTIARY: NWS API backup for US stations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx

from utils.logging import get_logger
from weather.temperature import PreciseTemp, Precision

log = get_logger("nws_feed")

NWS_BASE = "https://api.weather.gov"
USER_AGENT = "StationSniper/1.0 (weather-monitoring)"


@dataclass
class NWSObservation:
    station: str
    temp_c: float
    obs_time: datetime
    source: str = "nws"


class NWSFeed:
    """Polls NWS API for latest observations. US stations only, ~5-20 min lag."""

    def __init__(self) -> None:
        transport = httpx.AsyncHTTPTransport(retries=2)
        self._client = httpx.AsyncClient(
            timeout=15.0,
            transport=transport,
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
        )

    async def get_latest(self, icao: str) -> NWSObservation | None:
        """Fetch latest observation for a US station from NWS API."""
        # NWS only covers US stations (K-prefix ICAO codes)
        if not icao.startswith("K"):
            return None

        url = f"{NWS_BASE}/stations/{icao}/observations/latest"

        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("nws_fetch_failed", station=icao, error=str(e))
            return None

        try:
            props = data["properties"]
            temp_data = props["temperature"]

            if temp_data["value"] is None:
                return None

            temp_c = float(temp_data["value"])
            # NWS sometimes reports in different units
            if temp_data.get("unitCode") == "wmoUnit:degF":
                temp_c = (temp_c - 32.0) * 5.0 / 9.0

            obs_time_str = props["timestamp"]
            obs_time = datetime.fromisoformat(obs_time_str.replace("Z", "+00:00"))

            return NWSObservation(
                station=icao,
                temp_c=temp_c,
                obs_time=obs_time,
            )
        except (KeyError, ValueError, TypeError) as e:
            log.warning("nws_parse_failed", station=icao, error=str(e))
            return None

    def to_precise_temp(self, obs: NWSObservation) -> PreciseTemp:
        """Convert NWS observation to PreciseTemp (whole-degree)."""
        return PreciseTemp(celsius=obs.temp_c, precision=Precision.WHOLE)

    async def close(self) -> None:
        await self._client.aclose()
