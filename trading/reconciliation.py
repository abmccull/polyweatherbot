"""Realtime order/fill reconciliation from CLOB user websocket events."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from sqlalchemy import or_

from config import Config
from db.engine import get_session
from db.models import Trade
from utils.logging import get_logger

log = get_logger("reconciler")

TERMINAL_STATUSES = ("FILLED", "MATCHED", "FAILED", "CANCELED", "REJECTED")
NON_EXECUTED_STATUSES = ("FAILED", "CANCELED", "REJECTED")


class ExecutionReconciler:
    """Consumes user websocket events and keeps DB trades fill-accurate."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._running = False
        self._task: asyncio.Task | None = None
        self._ws = None

    async def start(self) -> None:
        """Start user websocket reconciliation loop."""
        if self._running or self._config.dry_run or not self._config.enable_user_ws:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="user_ws_reconciler")

    async def stop(self) -> None:
        """Stop user websocket reconciliation loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._ws = None

    async def _run_loop(self) -> None:
        """Reconnect and consume user events forever while running."""
        if not (self._config.poly_api_key and self._config.poly_api_secret and self._config.poly_api_passphrase):
            log.warning("user_ws_disabled_missing_api_creds")
            self._running = False
            return

        try:
            import websockets
        except Exception as e:
            log.warning("user_ws_dependency_missing", error=str(e), package="websockets")
            self._running = False
            return

        while self._running:
            try:
                async with websockets.connect(
                    self._config.clob_user_ws_url,
                    ping_interval=self._config.ws_ping_interval,
                    ping_timeout=self._config.ws_ping_interval,
                    max_size=2**22,
                ) as ws:
                    self._ws = ws
                    await self._send_subscribe()
                    log.info("user_ws_connected")

                    while self._running:
                        raw = await ws.recv()
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        self.process_payload(payload)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("user_ws_error", error=str(e))
                await asyncio.sleep(self._config.ws_reconnect_seconds)
            finally:
                self._ws = None

    async def _send_subscribe(self) -> None:
        if self._ws is None:
            return
        payload = {
            "type": "user",
            "auth": {
                "apiKey": self._config.poly_api_key,
                "secret": self._config.poly_api_secret,
                "passphrase": self._config.poly_api_passphrase,
            },
        }
        await self._ws.send(json.dumps(payload))

    def process_payload(self, payload: Any) -> None:
        """Parse websocket payload and apply all recognized order/trade events."""
        for event in self._iter_events(payload):
            self.process_event(event)

    def process_event(self, event: dict[str, Any]) -> None:
        """Apply one order/trade event to the best-matching trade row."""
        order_id = self._extract_order_id(event)
        status = self._normalize_status(self._extract_status(event))
        token_id = self._extract_token_id(event)
        side = self._extract_side(event)
        fill_size = self._extract_size(event)
        fill_price = self._extract_price(event)
        fee_paid = self._extract_fee(event)

        if order_id is None and token_id is None:
            return

        session = get_session()
        try:
            trade = None
            if order_id:
                trade = (
                    session.query(Trade)
                    .filter(Trade.order_id == order_id)
                    .order_by(Trade.created_at.desc())
                    .first()
                )

            if trade is None and token_id:
                action = None
                if side in ("BUY", "BID"):
                    action = "BUY"
                elif side in ("SELL", "ASK"):
                    action = "SELL"

                q = session.query(Trade).filter(
                    Trade.token_id == token_id,
                    Trade.resolved_correct.is_(None),
                )
                if action:
                    q = q.filter(Trade.action == action)
                if order_id is None:
                    q = q.filter(
                        or_(
                            Trade.order_status.is_(None),
                            Trade.order_status.notin_(TERMINAL_STATUSES),
                        )
                    )
                trade = q.order_by(Trade.created_at.desc()).first()

            if trade is None:
                return

            # Apply fill updates. Size/cost are fill-truth for economic accounting.
            if fill_size is not None and fill_size > 0:
                current_size = trade.size or 0.0
                current_cost = trade.cost or 0.0

                # Prefer cumulative semantics when event size is >= current size.
                if fill_size >= current_size:
                    new_size = fill_size
                    px = fill_price or trade.fill_price or trade.price
                    new_cost = px * new_size
                else:
                    # Fall back to delta semantics.
                    delta = fill_size
                    px = fill_price or trade.fill_price or trade.price
                    new_size = current_size + delta
                    new_cost = current_cost + (px * delta)

                trade.size = round(new_size, 8)
                trade.cost = round(new_cost, 8)
                if trade.size > 0:
                    trade.fill_price = round(trade.cost / trade.size, 8)

            elif fill_price is not None and (trade.size or 0.0) > 0:
                trade.fill_price = fill_price
                trade.cost = round(fill_price * trade.size, 8)

            if fee_paid is not None and fee_paid >= 0:
                trade.fee_paid = fee_paid

            if status:
                trade.order_status = status
                if status in NON_EXECUTED_STATUSES and (trade.size or 0.0) <= 0:
                    trade.size = 0.0
                    trade.cost = 0.0
                    trade.fill_price = None
                    trade.fee_paid = 0.0

            session.commit()
        except Exception as e:
            session.rollback()
            log.warning("reconcile_event_failed", error=str(e), event=str(event)[:400])
        finally:
            session.close()

    def _iter_events(self, payload: Any):
        if isinstance(payload, list):
            for item in payload:
                yield from self._iter_events(item)
            return

        if not isinstance(payload, dict):
            return

        # Direct event
        if any(k in payload for k in ("order_id", "orderID", "asset_id", "token_id", "tokenId")):
            yield payload

        for key in ("data", "events", "orders", "trades", "payload", "message"):
            nested = payload.get(key)
            if nested is not None:
                yield from self._iter_events(nested)

    @staticmethod
    def _extract_order_id(event: dict[str, Any]) -> str | None:
        val = event.get("order_id", event.get("orderID", event.get("id")))
        if isinstance(val, str) and val:
            return val
        return None

    @staticmethod
    def _extract_token_id(event: dict[str, Any]) -> str | None:
        val = event.get("asset_id", event.get("token_id", event.get("tokenId")))
        if isinstance(val, str) and val:
            return val
        return None

    @staticmethod
    def _extract_status(event: dict[str, Any]) -> str | None:
        val = event.get("status", event.get("order_status", event.get("state")))
        if isinstance(val, str) and val:
            return val
        return None

    @staticmethod
    def _normalize_status(status: str | None) -> str | None:
        if status is None:
            return None
        status_up = status.upper()
        aliases = {
            "DONE": "FILLED",
            "EXECUTED": "FILLED",
            "MATCHED": "FILLED",
            "CANCELLED": "CANCELED",
        }
        return aliases.get(status_up, status_up)

    @staticmethod
    def _extract_side(event: dict[str, Any]) -> str | None:
        side = event.get("side")
        if isinstance(side, str) and side:
            return side.upper()
        return None

    @staticmethod
    def _extract_size(event: dict[str, Any]) -> float | None:
        candidates = (
            event.get("filled_size"),
            event.get("filledSize"),
            event.get("size_matched"),
            event.get("sizeMatched"),
            event.get("matched_size"),
            event.get("size"),
        )
        for val in candidates:
            try:
                if val is not None:
                    return float(val)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_price(event: dict[str, Any]) -> float | None:
        candidates = (
            event.get("fill_price"),
            event.get("fillPrice"),
            event.get("matched_price"),
            event.get("avg_price"),
            event.get("avgPrice"),
            event.get("price"),
        )
        for val in candidates:
            try:
                if val is not None:
                    return float(val)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_fee(event: dict[str, Any]) -> float | None:
        candidates = (
            event.get("fee_paid"),
            event.get("feePaid"),
            event.get("fee"),
            event.get("fees"),
        )
        for val in candidates:
            try:
                if val is not None:
                    return float(val)
            except (TypeError, ValueError):
                continue
        return None
