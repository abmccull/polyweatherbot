"""Tests for market registry resilience against transient discovery failures."""

from __future__ import annotations

from datetime import date

import pytest

from markets.discovery import DiscoveredMarket
from markets.parser import MarketInfo
from markets.registry import MarketRegistry


class _FlakyDiscovery:
    def __init__(self, market: DiscoveredMarket) -> None:
        self._market = market
        self.calls = 0
        self.last_scan_error = "temporary gamma failure"

    async def scan(self):
        self.calls += 1
        if self.calls == 1:
            return [self._market]
        return None

    async def scan_precip(self):
        return []


def _market(event_id: str) -> DiscoveredMarket:
    return DiscoveredMarket(
        event_id=event_id,
        info=MarketInfo(
            city="Chicago",
            market_date=date(2026, 2, 26),
            raw_title="Highest temperature in Chicago on February 26?",
        ),
        icao="KORD",
        timezone="America/Chicago",
        wu_path="/weather/us/il/chicago",
        buckets=[],
    )


@pytest.mark.asyncio
async def test_refresh_keeps_existing_markets_when_scan_fails():
    discovery = _FlakyDiscovery(_market("evt1"))
    registry = MarketRegistry(discovery)

    await registry.refresh()
    assert registry.count == 1
    assert registry.get_market("evt1") is not None

    # Next refresh fails; registry should keep the last known-good market set.
    await registry.refresh()
    assert registry.count == 1
    assert registry.get_market("evt1") is not None
