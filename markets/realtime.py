"""Realtime market data cache via Polymarket CLOB websocket market channel."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config import Config
from utils.logging import get_logger

log = get_logger("market_ws")


@dataclass
class RealtimePrice:
    token_id: str
    best_bid: float | None
    best_ask: float | None
    bid_depth: float
    ask_depth: float
    updated_at: datetime


class ClobMarketFeed:
    """Maintains a token->best bid/ask cache from CLOB websocket events."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._prices: dict[str, RealtimePrice] = {}
        self._desired_tokens: set[str] = set()
        self._running = False
        self._task: asyncio.Task | None = None
        self._ws = None

    async def start(self) -> None:
        """Start the background websocket loop."""
        if self._running or not self._config.enable_market_ws:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="market_ws_loop")

    async def stop(self) -> None:
        """Stop the background websocket loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._ws = None

    def set_tokens(self, token_ids: list[str]) -> None:
        """Set token universe to subscribe for market updates."""
        self._desired_tokens = {tid for tid in token_ids if tid}
        if self._ws is not None and self._desired_tokens:
            # Fire-and-forget refresh subscribe with the latest token set.
            asyncio.create_task(self._send_subscribe())

    def get_price(self, token_id: str) -> RealtimePrice | None:
        return self._prices.get(token_id)

    def is_fresh(self, token_id: str, max_age_seconds: int) -> bool:
        price = self._prices.get(token_id)
        if price is None:
            return False
        age = (datetime.utcnow() - price.updated_at).total_seconds()
        return age <= max_age_seconds

    async def _run_loop(self) -> None:
        """Reconnecting websocket receiver loop."""
        try:
            import websockets
        except Exception as e:
            log.warning("market_ws_dependency_missing", error=str(e), package="websockets")
            self._running = False
            return

        while self._running:
            if not self._desired_tokens:
                await asyncio.sleep(1.0)
                continue

            try:
                async with websockets.connect(
                    self._config.clob_market_ws_url,
                    ping_interval=self._config.ws_ping_interval,
                    ping_timeout=self._config.ws_ping_interval,
                    max_size=2**22,
                ) as ws:
                    self._ws = ws
                    await self._send_subscribe()
                    log.info("market_ws_connected", tokens=len(self._desired_tokens))

                    while self._running:
                        raw = await ws.recv()
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        self.ingest_message(payload)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("market_ws_error", error=str(e))
                await asyncio.sleep(self._config.ws_reconnect_seconds)
            finally:
                self._ws = None

    async def _send_subscribe(self) -> None:
        """Subscribe to market stream for desired tokens."""
        if self._ws is None or not self._desired_tokens:
            return
        payload = {
            "type": "market",
            "assets_ids": sorted(self._desired_tokens),
        }
        await self._ws.send(json.dumps(payload))

    def ingest_message(self, payload: Any) -> None:
        """Parse a websocket payload and update local price cache."""
        for event in self._iter_events(payload):
            token_id = (
                event.get("asset_id")
                or event.get("token_id")
                or event.get("tokenId")
            )
            if not token_id:
                continue

            bids = event.get("bids")
            asks = event.get("asks")
            if bids is None and asks is None:
                continue

            best_bid, bid_depth = self._parse_book_side(bids, side="bid")
            best_ask, ask_depth = self._parse_book_side(asks, side="ask")

            self._prices[token_id] = RealtimePrice(
                token_id=token_id,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                updated_at=datetime.utcnow(),
            )

    def _iter_events(self, payload: Any):
        if isinstance(payload, list):
            for item in payload:
                yield from self._iter_events(item)
            return
        if not isinstance(payload, dict):
            return

        # Direct event payload.
        if any(k in payload for k in ("asset_id", "token_id", "tokenId")) and (
            "bids" in payload or "asks" in payload
        ):
            yield payload

        # Common nested wrappers used by websocket payloads.
        for key in ("data", "events", "books", "payload", "message"):
            nested = payload.get(key)
            if nested is not None:
                yield from self._iter_events(nested)

    @staticmethod
    def _parse_book_side(levels: Any, side: str) -> tuple[float | None, float]:
        if not levels:
            return None, 0.0

        best: float | None = None
        depth = 0.0

        for lvl in levels:
            price, size = ClobMarketFeed._parse_level(lvl)
            if price is None or size is None or price <= 0 or size <= 0:
                continue
            depth += price * size

            if side == "bid":
                if best is None or price > best:
                    best = price
            else:
                if best is None or price < best:
                    best = price

        return best, depth

    @staticmethod
    def _parse_level(level: Any) -> tuple[float | None, float | None]:
        if isinstance(level, dict):
            p = level.get("price", level.get("p"))
            s = level.get("size", level.get("s", level.get("amount")))
            try:
                return float(p), float(s)
            except (TypeError, ValueError):
                return None, None

        if isinstance(level, (list, tuple)) and len(level) >= 2:
            try:
                return float(level[0]), float(level[1])
            except (TypeError, ValueError):
                return None, None

        return None, None
