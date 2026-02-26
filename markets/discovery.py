"""Gamma API scanner for Polymarket temperature and precipitation markets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from markets.parser import (
    MarketInfo,
    BucketInfo,
    PrecipMarketInfo,
    PrecipBucketInfo,
    parse_event_title,
    parse_bucket_question,
    parse_precip_event_title,
    parse_precip_bucket_question,
)
from utils.logging import get_logger
from weather.station_map import lookup_city

log = get_logger("discovery")

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
GAMMA_EVENTS_FALLBACK_URL = "https://gamma-api.polymarket.com/events"


@dataclass
class DiscoveredBucket:
    """A single YES/NO bucket within a temperature market."""

    token_id: str
    condition_id: str
    bucket: BucketInfo
    outcome: str  # "Yes" or "No"


@dataclass
class DiscoveredPrecipBucket:
    """A single YES/NO bucket within a precipitation market."""

    token_id: str
    condition_id: str
    bucket: PrecipBucketInfo
    outcome: str  # "Yes" or "No"


@dataclass
class DiscoveredMarket:
    """A temperature event with all its buckets."""

    event_id: str
    info: MarketInfo
    icao: str
    timezone: str
    wu_path: str
    market_type: str = "temperature"  # "temperature" or "precipitation"
    display_unit: str = "C"  # "C" or "F"
    buckets: list[DiscoveredBucket] = field(default_factory=list)


@dataclass
class DiscoveredPrecipMarket:
    """A precipitation event with all its buckets."""

    event_id: str
    info: PrecipMarketInfo
    icao: str
    timezone: str
    market_type: str = "precipitation"
    buckets: list[DiscoveredPrecipBucket] = field(default_factory=list)


class MarketDiscovery:
    """Scans Gamma API for active temperature and precipitation markets on Polymarket."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        self._last_scan_error: str | None = None

    @property
    def last_scan_error(self) -> str | None:
        return self._last_scan_error

    async def scan(self) -> list[DiscoveredMarket] | None:
        """Query Gamma API for open temperature events and parse them."""
        markets: list[DiscoveredMarket] = []

        try:
            all_events = await self._fetch_all_events()
            self._last_scan_error = None
            log.info("gamma_events_fetched", total=len(all_events))

            for event in all_events:
                market = self._parse_temp_event(event)
                if market is not None:
                    markets.append(market)

        except httpx.HTTPError as e:
            self._last_scan_error = str(e)
            log.error("gamma_scan_failed", error=str(e))
            return None

        log.info("temperature_markets_found", count=len(markets))
        return markets

    async def scan_precip(self) -> list[DiscoveredPrecipMarket] | None:
        """Query Gamma API for open precipitation events and parse them."""
        markets: list[DiscoveredPrecipMarket] = []

        try:
            all_events = await self._fetch_all_events()
            self._last_scan_error = None

            for event in all_events:
                market = self._parse_precip_event(event)
                if market is not None:
                    markets.append(market)

        except httpx.HTTPError as e:
            self._last_scan_error = str(e)
            log.error("gamma_precip_scan_failed", error=str(e))
            return None

        log.info("precipitation_markets_found", count=len(markets))
        return markets

    async def _fetch_all_events(self) -> list[dict]:
        """Fetch all open events from Gamma API, paginating as needed."""
        offset = 0
        limit = 100
        all_events: list[dict] = []
        endpoints = [GAMMA_EVENTS_URL]
        if GAMMA_EVENTS_FALLBACK_URL not in endpoints:
            endpoints.append(GAMMA_EVENTS_FALLBACK_URL)
        params_variants = (
            {"closed": "false"},
            {"active": "true"},
            {},
        )

        while True:
            events = await self._fetch_events_page(
                endpoints=endpoints,
                params_variants=params_variants,
                limit=limit,
                offset=offset,
            )
            if not events:
                break
            all_events.extend(events)
            if len(events) < limit:
                break
            offset += limit

        return all_events

    async def _fetch_events_page(
        self,
        endpoints: list[str],
        params_variants: tuple[dict[str, str], ...],
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        """Fetch one paginated events page with fallback params/endpoints."""
        last_error: Exception | None = None
        for endpoint in endpoints:
            for params_base in params_variants:
                params: dict[str, str | int] = {
                    "limit": limit,
                    "offset": offset,
                    **params_base,
                }
                try:
                    resp = await self._client.get(endpoint, params=params)
                    resp.raise_for_status()
                    payload = resp.json()
                    if isinstance(payload, list):
                        if params_base != {"closed": "false"} or endpoint != GAMMA_EVENTS_URL:
                            log.warning(
                                "gamma_events_fallback_used",
                                endpoint=endpoint,
                                params=params_base,
                                offset=offset,
                            )
                        return payload
                    raise httpx.HTTPError(
                        f"Unexpected events payload type: {type(payload).__name__}",
                    )
                except httpx.HTTPStatusError as e:
                    last_error = e
                    # 404s are often transient on one query shape; try fallbacks.
                    if e.response.status_code == 404:
                        continue
                    raise
                except httpx.HTTPError as e:
                    last_error = e
                    continue

        if last_error is not None:
            raise last_error
        raise httpx.HTTPError("Gamma events page fetch failed with unknown error")

    def _parse_temp_event(self, event: dict) -> DiscoveredMarket | None:
        """Parse a single Gamma API event into a DiscoveredMarket (temperature)."""
        title = event.get("title", "")
        event_id = event.get("id", "")

        # Check if this is a temperature market
        info = parse_event_title(title)
        if info is None:
            return None

        # Look up station
        station = lookup_city(info.city)
        if station is None:
            log.debug("unknown_city", city=info.city, title=title)
            return None

        market = DiscoveredMarket(
            event_id=event_id,
            info=info,
            icao=station.icao,
            timezone=station.timezone,
            wu_path=station.wu_path,
            market_type="temperature",
            display_unit=station.display_unit,
        )

        # Parse nested markets (buckets)
        for sub_market in event.get("markets", []):
            self._parse_temp_buckets(sub_market, market)

        if not market.buckets:
            log.debug("no_buckets_parsed", event_id=event_id, title=title)
            return None

        log.info(
            "market_discovered",
            event_id=event_id,
            city=info.city,
            date=str(info.market_date),
            station=station.icao,
            unit=station.display_unit,
            buckets=len(market.buckets),
        )
        return market

    def _parse_temp_buckets(self, sub_market: dict, market: DiscoveredMarket) -> None:
        """Extract bucket info from a sub-market within a temperature event."""
        if not self._is_tradable_sub_market(sub_market):
            return

        question = sub_market.get("question", "")
        condition_id = sub_market.get("conditionId", "")

        bucket = parse_bucket_question(question)
        if bucket is None:
            return

        # Extract token IDs for YES outcome
        tokens = sub_market.get("clobTokenIds", "")
        outcomes = sub_market.get("outcomes", "")

        tokens = self._parse_json_or_csv(tokens)
        outcomes = self._parse_json_or_csv(outcomes)

        # Find the YES token
        for i, outcome in enumerate(outcomes):
            if outcome.lower() == "yes" and i < len(tokens):
                market.buckets.append(
                    DiscoveredBucket(
                        token_id=tokens[i],
                        condition_id=condition_id,
                        bucket=bucket,
                        outcome="Yes",
                    )
                )
                break

    def _parse_precip_event(self, event: dict) -> DiscoveredPrecipMarket | None:
        """Parse a single Gamma API event into a DiscoveredPrecipMarket."""
        title = event.get("title", "")
        event_id = event.get("id", "")

        info = parse_precip_event_title(title)
        if info is None:
            return None

        # Look up station by city name
        station = lookup_city(info.city)
        if station is None:
            log.debug("unknown_precip_city", city=info.city, title=title)
            return None

        market = DiscoveredPrecipMarket(
            event_id=event_id,
            info=info,
            icao=station.icao,
            timezone=station.timezone,
        )

        # Parse nested markets (buckets)
        for sub_market in event.get("markets", []):
            self._parse_precip_buckets(sub_market, market)

        if not market.buckets:
            log.debug("no_precip_buckets_parsed", event_id=event_id, title=title)
            return None

        log.info(
            "precip_market_discovered",
            event_id=event_id,
            city=info.city,
            month=info.month,
            buckets=len(market.buckets),
        )
        return market

    def _parse_precip_buckets(self, sub_market: dict, market: DiscoveredPrecipMarket) -> None:
        """Extract bucket info from a sub-market within a precipitation event."""
        if not self._is_tradable_sub_market(sub_market):
            return

        question = sub_market.get("question", "")
        condition_id = sub_market.get("conditionId", "")

        bucket = parse_precip_bucket_question(question)
        if bucket is None:
            return

        tokens = sub_market.get("clobTokenIds", "")
        outcomes = sub_market.get("outcomes", "")

        tokens = self._parse_json_or_csv(tokens)
        outcomes = self._parse_json_or_csv(outcomes)

        for i, outcome in enumerate(outcomes):
            if outcome.lower() == "yes" and i < len(tokens):
                market.buckets.append(
                    DiscoveredPrecipBucket(
                        token_id=tokens[i],
                        condition_id=condition_id,
                        bucket=bucket,
                        outcome="Yes",
                    )
                )
                break

    @staticmethod
    def _parse_json_or_csv(value: str | list) -> list[str]:
        """Parse a value that might be a JSON string or comma-separated."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                import json
                return json.loads(value)
            except (ValueError, TypeError):
                return [v.strip() for v in value.split(",") if v.strip()]
        return []

    @staticmethod
    def _is_tradable_sub_market(sub_market: dict) -> bool:
        """Return True when a market bucket appears currently tradable on CLOB."""
        if MarketDiscovery._as_bool(sub_market.get("closed"), default=False):
            return False
        if MarketDiscovery._as_bool(sub_market.get("archived"), default=False):
            return False
        if not MarketDiscovery._as_bool(sub_market.get("active"), default=True):
            return False

        accepting = sub_market.get("acceptingOrders")
        if accepting is None:
            accepting = sub_market.get("accepting_orders")
        if accepting is not None and not MarketDiscovery._as_bool(accepting, default=True):
            return False

        orderbook_enabled = sub_market.get("enableOrderBook")
        if orderbook_enabled is None:
            orderbook_enabled = sub_market.get("enable_order_book")
        if orderbook_enabled is not None and not MarketDiscovery._as_bool(orderbook_enabled, default=True):
            return False

        return True

    @staticmethod
    def _as_bool(value, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            val = value.strip().lower()
            if val in ("true", "1", "yes", "y", "t"):
                return True
            if val in ("false", "0", "no", "n", "f", ""):
                return False
        return default

    async def close(self) -> None:
        await self._client.aclose()
