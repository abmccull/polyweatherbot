"""Tests for orderbook 404 cooldown behavior."""

from __future__ import annotations

from config import Config
from trading.executor import TradeExecutor
from trading.portfolio import Portfolio


class _MissingBookClient:
    def __init__(self) -> None:
        self.calls = 0

    def get_order_book(self, token_id: str):
        self.calls += 1
        raise RuntimeError("PolyApiException[status_code=404, error_message={'error': 'No orderbook exists for the requested token id'}]")


def test_orderbook_404_cooldown_skips_repeated_calls():
    cfg = Config(dry_run=True, orderbook_404_cooldown_seconds=300)
    executor = TradeExecutor(cfg, Portfolio(cfg), tracker=None)
    client = _MissingBookClient()
    executor._clob_client = client

    token = "tok_missing"

    first = executor.get_order_book(token)
    second = executor.get_order_book(token)

    assert first is None
    assert second is None
    assert client.calls == 1
