"""SECONDARY: Synoptic Data API for 1-minute ASOS observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx

from utils.logging import get_logger
from weather.temperature import PreciseTemp, Precision

log = get_logger("synoptic_feed")

SYNOPTIC_BASE = "https://api.synopticdata.com/v2"


@dataclass
class SynopticObservation:
    station: str
    temp_c: float
    obs_time: datetime
    source: str = "synoptic"


class SynopticFeed:
    """Polls Synoptic Data API for 1-minute ASOS data (whole-degree only)."""

    def __init__(self, api_token: str) -> None:
        self._token = api_token
        transport = httpx.AsyncHTTPTransport(retries=2)
        self._client = httpx.AsyncClient(timeout=15.0, transport=transport)
        self._enabled = bool(api_token)

    async def get_latest(self, icao: str) -> SynopticObservation | None:
        """Fetch latest observation for a station from Synoptic Data API.

        Uses the 1-minute ASOS data by appending '1M' to the station ID.
        """
        if not self._enabled:
            return None

        # 1-minute ASOS station IDs have "1M" suffix
        stid = f"{icao}1M"
        url = f"{SYNOPTIC_BASE}/stations/latest"
        params = {
            "stid": stid,
            "token": self._token,
            "vars": "air_temp",
            "units": "temp|C",
        }

        try:
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("synoptic_fetch_failed", station=icao, error=str(e))
            return None

        try:
            station_data = data["STATION"][0]
            temp_c = float(station_data["OBSERVATIONS"]["air_temp_value_1"]["value"])
            obs_time_str = station_data["OBSERVATIONS"]["air_temp_value_1"]["date_time"]
            obs_time = datetime.fromisoformat(obs_time_str.replace("Z", "+00:00"))

            return SynopticObservation(
                station=icao,
                temp_c=temp_c,
                obs_time=obs_time,
            )
        except (KeyError, IndexError, ValueError) as e:
            log.warning("synoptic_parse_failed", station=icao, error=str(e))
            return None

    def to_precise_temp(self, obs: SynopticObservation) -> PreciseTemp:
        """Convert Synoptic observation to PreciseTemp (whole-degree only)."""
        return PreciseTemp(celsius=obs.temp_c, precision=Precision.WHOLE)

    async def close(self) -> None:
        await self._client.aclose()
