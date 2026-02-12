"""In-memory active market cache with price refresh."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from markets.discovery import DiscoveredMarket, DiscoveredBucket, DiscoveredPrecipMarket, MarketDiscovery
from utils.logging import get_logger

log = get_logger("registry")


@dataclass
class BucketPrice:
    """Current pricing for a bucket."""

    token_id: str
    best_bid: float | None = None
    best_ask: float | None = None
    bid_depth: float = 0.0  # total size on bid side
    ask_depth: float = 0.0  # total size on ask side
    last_refreshed: datetime | None = None


@dataclass
class ActiveMarket:
    """A market being actively monitored."""

    market: DiscoveredMarket
    bucket_prices: dict[str, BucketPrice] = field(default_factory=dict)
    last_refreshed: datetime | None = None


@dataclass
class ActivePrecipMarket:
    """A precipitation market being actively monitored."""

    market: DiscoveredPrecipMarket
    bucket_prices: dict[str, BucketPrice] = field(default_factory=dict)
    last_refreshed: datetime | None = None


class MarketRegistry:
    """Maintains cache of active temperature and precipitation markets with pricing."""

    def __init__(self, discovery: MarketDiscovery) -> None:
        self._discovery = discovery
        # event_id → ActiveMarket
        self._markets: dict[str, ActiveMarket] = {}
        # event_id → ActivePrecipMarket
        self._precip_markets: dict[str, ActivePrecipMarket] = {}

    async def refresh(self) -> None:
        """Re-scan for markets and update the registry."""
        discovered = await self._discovery.scan()
        now = datetime.utcnow()

        # Build new set of event IDs
        new_ids = {m.event_id for m in discovered}

        # Remove markets that are no longer active
        removed = [eid for eid in self._markets if eid not in new_ids]
        for eid in removed:
            del self._markets[eid]
            log.info("market_removed", event_id=eid)

        # Add/update markets
        for market in discovered:
            if market.event_id not in self._markets:
                self._markets[market.event_id] = ActiveMarket(market=market, last_refreshed=now)
                log.info(
                    "market_added",
                    event_id=market.event_id,
                    city=market.info.city,
                    date=str(market.info.market_date),
                )
            else:
                self._markets[market.event_id].market = market
                self._markets[market.event_id].last_refreshed = now

    async def refresh_precip(self) -> None:
        """Re-scan for precipitation markets and update the registry."""
        discovered = await self._discovery.scan_precip()
        now = datetime.utcnow()

        new_ids = {m.event_id for m in discovered}

        removed = [eid for eid in self._precip_markets if eid not in new_ids]
        for eid in removed:
            del self._precip_markets[eid]
            log.info("precip_market_removed", event_id=eid)

        for market in discovered:
            if market.event_id not in self._precip_markets:
                self._precip_markets[market.event_id] = ActivePrecipMarket(market=market, last_refreshed=now)
                log.info(
                    "precip_market_added",
                    event_id=market.event_id,
                    city=market.info.city,
                    month=market.info.month,
                )
            else:
                self._precip_markets[market.event_id].market = market
                self._precip_markets[market.event_id].last_refreshed = now

    def update_prices(self, token_id: str, best_bid: float | None, best_ask: float | None,
                      bid_depth: float = 0.0, ask_depth: float = 0.0) -> None:
        """Update pricing for a specific bucket token (temperature or precip)."""
        # Check temperature markets
        for active in self._markets.values():
            for bucket in active.market.buckets:
                if bucket.token_id == token_id:
                    active.bucket_prices[token_id] = BucketPrice(
                        token_id=token_id,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        bid_depth=bid_depth,
                        ask_depth=ask_depth,
                        last_refreshed=datetime.utcnow(),
                    )
                    return

        # Check precipitation markets
        for active in self._precip_markets.values():
            for bucket in active.market.buckets:
                if bucket.token_id == token_id:
                    active.bucket_prices[token_id] = BucketPrice(
                        token_id=token_id,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        bid_depth=bid_depth,
                        ask_depth=ask_depth,
                        last_refreshed=datetime.utcnow(),
                    )
                    return

    def get_all_active(self) -> list[ActiveMarket]:
        """Return all active temperature markets."""
        return list(self._markets.values())

    def get_all_active_precip(self) -> list[ActivePrecipMarket]:
        """Return all active precipitation markets."""
        return list(self._precip_markets.values())

    def get_market(self, event_id: str) -> ActiveMarket | None:
        """Get a specific temperature market by event ID."""
        return self._markets.get(event_id)

    def get_precip_market(self, event_id: str) -> ActivePrecipMarket | None:
        """Get a specific precipitation market by event ID."""
        return self._precip_markets.get(event_id)

    def get_bucket_price(self, token_id: str) -> BucketPrice | None:
        """Get pricing for a specific bucket token."""
        for active in self._markets.values():
            if token_id in active.bucket_prices:
                return active.bucket_prices[token_id]
        for active in self._precip_markets.values():
            if token_id in active.bucket_prices:
                return active.bucket_prices[token_id]
        return None

    @property
    def count(self) -> int:
        return len(self._markets)

    @property
    def precip_count(self) -> int:
        return len(self._precip_markets)
