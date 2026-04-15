"""Lightweight async client for Polymarket Data/Gamma APIs used by copycat."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx


DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"


class PolymarketDataClient:
    """Async wrapper with small convenience helpers and simple event cache."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        self._event_cache: dict[str, tuple[datetime, dict]] = {}

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_trades(self, wallet: str, limit: int = 100, offset: int = 0) -> list[dict]:
        resp = await self._client.get(
            f"{DATA_API}/trades",
            params={"user": wallet, "limit": limit, "offset": offset},
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    async def fetch_positions(self, wallet: str) -> list[dict]:
        resp = await self._client.get(
            f"{DATA_API}/positions",
            params={"user": wallet},
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    async def fetch_leaderboard(
        self,
        period: str,
        *,
        category: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict]:
        params: dict[str, str | int] = {"timePeriod": period, "limit": limit, "offset": offset}
        if category:
            params["category"] = category
        resp = await self._client.get(f"{DATA_API}/v1/leaderboard", params=params)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    async def fetch_event_by_slug(self, event_slug: str) -> dict | None:
        """Fetch Gamma event by slug with short in-memory cache."""
        slug = (event_slug or "").strip().lower()
        if not slug:
            return None

        cached = self._event_cache.get(slug)
        if cached is not None:
            ts, event = cached
            if (datetime.now(timezone.utc) - ts).total_seconds() <= 900:
                return event

        resp = await self._client.get(
            f"{GAMMA_API}/events",
            params={"slug": slug, "limit": 1, "offset": 0},
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or not payload:
            return None
        event = payload[0]
        self._event_cache[slug] = (datetime.now(timezone.utc), event)
        return event
