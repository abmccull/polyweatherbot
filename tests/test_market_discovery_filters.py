"""Tests for tradability filtering in market discovery."""

from __future__ import annotations

from datetime import date
import httpx
import pytest

from markets.discovery import MarketDiscovery, DiscoveredMarket
from markets.parser import MarketInfo


def _market() -> DiscoveredMarket:
    return DiscoveredMarket(
        event_id="evt1",
        info=MarketInfo(city="Chicago", market_date=date(2026, 2, 25), raw_title="x"),
        icao="KORD",
        timezone="America/Chicago",
        wu_path="/weather/us/il/chicago",
    )


def _sub_market(**overrides) -> dict:
    base = {
        "question": "Will the highest temperature in Chicago be 30°F or below on February 25?",
        "conditionId": "cond1",
        "clobTokenIds": "[\"tok_yes\",\"tok_no\"]",
        "outcomes": "[\"Yes\",\"No\"]",
        "active": True,
        "closed": False,
        "archived": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
    }
    base.update(overrides)
    return base


def test_parse_temp_bucket_skips_closed_market():
    discovery = MarketDiscovery()
    market = _market()
    discovery._parse_temp_buckets(_sub_market(closed=True), market)
    assert len(market.buckets) == 0


def test_parse_temp_bucket_skips_not_accepting_orders():
    discovery = MarketDiscovery()
    market = _market()
    discovery._parse_temp_buckets(_sub_market(acceptingOrders=False), market)
    assert len(market.buckets) == 0


def test_parse_temp_bucket_accepts_tradable_market():
    discovery = MarketDiscovery()
    market = _market()
    discovery._parse_temp_buckets(_sub_market(), market)
    assert len(market.buckets) == 1
    assert market.buckets[0].token_id == "tok_yes"


@pytest.mark.asyncio
async def test_fetch_all_events_falls_back_to_active_param_on_404():
    discovery = MarketDiscovery()
    calls: list[dict] = []

    async def _fake_get(url, params):
        calls.append(dict(params))
        request = httpx.Request("GET", url, params=params)
        if params.get("closed") == "false":
            resp = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("not found", request=request, response=resp)
        return httpx.Response(200, request=request, json=[])

    discovery._client.get = _fake_get  # type: ignore[assignment]
    events = await discovery._fetch_all_events()
    assert events == []
    assert any(c.get("closed") == "false" for c in calls)
    assert any(c.get("active") == "true" for c in calls)
