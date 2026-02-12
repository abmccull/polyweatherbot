"""PRIMARY: aviationweather.gov METAR cache file poller."""

from __future__ import annotations

import gzip
import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, date

import httpx
import pytz

from utils.logging import get_logger
from weather.station_map import all_icao_codes, lookup_icao
from weather.temperature import PreciseTemp, parse_metar_temp, parse_t_group
from weather.precipitation import PrecipReading, MonthlyPrecipAccumulator, parse_precip_remarks

log = get_logger("metar_feed")

METAR_CACHE_URL = "https://aviationweather.gov/data/cache/metars.cache.xml.gz"


@dataclass
class MetarObservation:
    station: str
    raw: str
    temp: PreciseTemp | None
    obs_time: datetime
    precip: PrecipReading | None = None
    source: str = "metar_cache"


@dataclass
class DailyHigh:
    """Track running daily high for a station."""

    station: str
    date: date
    high: PreciseTemp | None = None
    last_obs_time: datetime | None = None

    def update(self, obs: MetarObservation) -> bool:
        """Update daily high if observation is a new max. Returns True if updated."""
        if obs.temp is None:
            return False
        if self.high is None or obs.temp.celsius > self.high.celsius:
            self.high = obs.temp
            self.last_obs_time = obs.obs_time
            return True
        return False


class MetarFeed:
    """Polls the global METAR cache and tracks daily highs for configured stations."""

    def __init__(self, max_retries: int = 3) -> None:
        transport = httpx.AsyncHTTPTransport(retries=max_retries)
        self._client = httpx.AsyncClient(timeout=30.0, transport=transport)
        self._tracked_stations: set[str] = set(all_icao_codes())
        # station ICAO → DailyHigh
        self._daily_highs: dict[str, DailyHigh] = {}
        # station ICAO → latest observation
        self._latest_obs: dict[str, MetarObservation] = {}
        # station ICAO → MonthlyPrecipAccumulator
        self._monthly_precip: dict[str, MonthlyPrecipAccumulator] = {}

    async def poll(self) -> list[MetarObservation]:
        """Download and parse the global METAR cache. Returns observations for tracked stations."""
        try:
            resp = await self._client.get(METAR_CACHE_URL)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.error("metar_cache_fetch_failed", error=str(e))
            return []

        try:
            raw_xml = gzip.decompress(resp.content)
        except Exception as e:
            log.error("metar_cache_decompress_failed", error=str(e))
            return []

        observations = self._parse_xml(raw_xml)
        self._update_daily_highs(observations)
        self._update_monthly_precip(observations)
        return observations

    def _parse_xml(self, raw_xml: bytes) -> list[MetarObservation]:
        """Parse METAR XML and filter to tracked stations."""
        observations = []
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as e:
            log.error("metar_xml_parse_failed", error=str(e))
            return []

        for metar_el in root.iter("METAR"):
            station_id = metar_el.findtext("station_id", "")
            if station_id not in self._tracked_stations:
                continue

            raw_text = metar_el.findtext("raw_text", "")
            obs_time_str = metar_el.findtext("observation_time", "")

            if not raw_text or not obs_time_str:
                continue

            # Parse observation time
            try:
                obs_time = datetime.fromisoformat(obs_time_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Parse temperature (prefer T-group from raw text)
            temp = parse_metar_temp(raw_text)

            # Parse precipitation from remarks
            precip = parse_precip_remarks(raw_text)

            obs = MetarObservation(
                station=station_id,
                raw=raw_text,
                temp=temp,
                obs_time=obs_time,
                precip=precip,
            )
            observations.append(obs)
            self._latest_obs[station_id] = obs

        log.info("metar_cache_parsed", tracked_obs=len(observations))
        return observations

    def _update_daily_highs(self, observations: list[MetarObservation]) -> None:
        """Update running daily highs from new observations."""
        for obs in observations:
            station_info = lookup_icao(obs.station)
            if station_info is None:
                continue

            # Determine local date for this observation
            tz = pytz.timezone(station_info.timezone)
            local_dt = obs.obs_time.astimezone(tz)
            local_d = local_dt.date()

            # Reset if new day
            if obs.station not in self._daily_highs or self._daily_highs[obs.station].date != local_d:
                self._daily_highs[obs.station] = DailyHigh(station=obs.station, date=local_d)

            updated = self._daily_highs[obs.station].update(obs)
            if updated and obs.temp:
                log.info(
                    "daily_high_updated",
                    station=obs.station,
                    temp_c=obs.temp.celsius,
                    precision=obs.temp.precision.value,
                    date=str(local_d),
                )

    def _update_monthly_precip(self, observations: list[MetarObservation]) -> None:
        """Update monthly precipitation accumulators from new observations."""
        for obs in observations:
            if obs.precip is None:
                continue

            station_info = lookup_icao(obs.station)
            if station_info is None:
                continue

            # Determine local month for this observation
            tz = pytz.timezone(station_info.timezone)
            local_dt = obs.obs_time.astimezone(tz)
            month_key = local_dt.strftime("%Y-%m")

            # Reset accumulator if new month
            if obs.station not in self._monthly_precip or self._monthly_precip[obs.station].month != month_key:
                self._monthly_precip[obs.station] = MonthlyPrecipAccumulator(
                    station=obs.station,
                    month=month_key,
                )

            acc = self._monthly_precip[obs.station]
            old_total = acc.total_inches
            acc.update(obs.precip, obs.obs_time)

            if acc.total_inches != old_total:
                log.info(
                    "monthly_precip_updated",
                    station=obs.station,
                    month=month_key,
                    total_inches=round(acc.total_inches, 2),
                )

    def get_daily_high(self, icao: str) -> DailyHigh | None:
        """Get current daily high for a station."""
        return self._daily_highs.get(icao)

    def get_latest_obs(self, icao: str) -> MetarObservation | None:
        """Get most recent observation for a station."""
        return self._latest_obs.get(icao)

    def get_monthly_precip(self, icao: str) -> MonthlyPrecipAccumulator | None:
        """Get current monthly precipitation accumulator for a station."""
        return self._monthly_precip.get(icao)

    def add_station(self, icao: str) -> None:
        """Add a station to track."""
        self._tracked_stations.add(icao)

    async def close(self) -> None:
        await self._client.aclose()
